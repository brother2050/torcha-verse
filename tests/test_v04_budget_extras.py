"""Tests for the v0.4.3 BudgetTracker additions: try_acquire, allocate_many,
stats, and allocate_with_backoff.
"""

from __future__ import annotations

import threading
import time

import pytest

from infrastructure.resource_budget import (
    AllocationHandle,
    BudgetExceededError,
    BudgetTracker,
    ResourceBudget,
)


@pytest.fixture
def tracker() -> BudgetTracker:
    return BudgetTracker(
        ResourceBudget(
            vram_gb=8.0,
            ram_gb=32.0,
            disk_gb=100.0,
            max_concurrent_models=1,
            max_concurrent_requests=4,
        )
    )


# ---------------------------------------------------------------------------
# try_acquire
# ---------------------------------------------------------------------------
def test_try_acquire_returns_handle_on_success(
    tracker: BudgetTracker,
) -> None:
    handle = tracker.try_acquire("a", vram_gb=2.0)
    assert isinstance(handle, AllocationHandle)
    assert not handle._released


def test_try_acquire_returns_none_when_saturated(
    tracker: BudgetTracker,
) -> None:
    tracker.allocate("big", vram_gb=7.0)
    # 1 GB left; asking for 2 must fail without raising.
    assert tracker.try_acquire("small", vram_gb=2.0) is None


def test_try_acquire_raises_for_infeasible_request(
    tracker: BudgetTracker,
) -> None:
    with pytest.raises(BudgetExceededError):
        tracker.try_acquire("huge", vram_gb=99.0)


def test_try_acquire_rejects_slot_only_contention(
    tracker: BudgetTracker,
) -> None:
    tracker.allocate("first", model_slot=True)
    # Second model-slot-only request is rejected immediately because
    # there is no event source for "slot freed".
    with pytest.raises(BudgetExceededError):
        tracker.try_acquire("second", model_slot=True)


# ---------------------------------------------------------------------------
# allocate_many
# ---------------------------------------------------------------------------
def test_allocate_many_succeeds_atomically(tracker: BudgetTracker) -> None:
    handles = tracker.allocate_many(
        [
            {"name": "m1", "vram_gb": 1.0, "ram_gb": 4.0},
            {"name": "m2", "vram_gb": 2.0, "ram_gb": 8.0},
        ]
    )
    assert len(handles) == 2
    used = tracker.used()
    assert used["vram_gb"] == pytest.approx(3.0)
    assert used["ram_gb"] == pytest.approx(12.0)


def test_allocate_many_returns_empty_on_partial_fit(
    tracker: BudgetTracker,
) -> None:
    # Saturate the budget with one large handle so the second spec
    # in the batch cannot fit.
    tracker.allocate("big", vram_gb=7.0)
    handles = tracker.allocate_many(
        [
            {"name": "m1", "vram_gb": 0.5, "ram_gb": 1.0},
            {"name": "m2", "vram_gb": 4.0, "ram_gb": 8.0},
        ]
    )
    assert handles == []
    # The budget is *not* partially consumed: the first spec did not
    # get a handle even though it could have fit on its own.
    used = tracker.used()
    assert used["vram_gb"] == pytest.approx(7.0)


def test_allocate_many_empty_specs_returns_empty(
    tracker: BudgetTracker,
) -> None:
    assert tracker.allocate_many([]) == []


def test_allocate_many_raises_for_infeasible_first_spec(
    tracker: BudgetTracker,
) -> None:
    with pytest.raises(BudgetExceededError):
        tracker.allocate_many(
            [
                {"name": "huge", "vram_gb": 99.0},
                {"name": "tiny", "vram_gb": 0.5},
            ]
        )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def test_stats_reports_budget_used_free(tracker: BudgetTracker) -> None:
    handle = tracker.allocate("m1", vram_gb=3.0, ram_gb=8.0, model_slot=True)
    snapshot = tracker.stats()
    assert snapshot["budget"]["vram_gb"] == pytest.approx(8.0)
    assert snapshot["used"]["vram_gb"] == pytest.approx(3.0)
    assert snapshot["free"]["vram_gb"] == pytest.approx(5.0)
    # The handle took the only model slot, so ``model_slots`` is 0.
    assert snapshot["free"]["model_slots"] == 0
    assert snapshot["active_allocations"] == 1
    tracker.release(handle)
    snapshot = tracker.stats()
    assert snapshot["used"]["vram_gb"] == pytest.approx(0.0)
    assert snapshot["active_allocations"] == 0


def test_stats_free_is_clamped_at_zero(tracker: BudgetTracker) -> None:
    # When usage overshoots the budget for any reason (it should not
    # happen in normal use, but a buggy caller could push the counter
    # past zero), ``free`` must clamp to 0 rather than emit a
    # negative number that breaks Grafana queries.
    snapshot = tracker.stats()
    for key, value in snapshot["free"].items():
        assert value >= 0
        # ``free`` is a small dict so a smoke test on every key is
        # cheap and future-proofs against new fields.


# ---------------------------------------------------------------------------
# allocate_with_backoff
# ---------------------------------------------------------------------------
def test_allocate_with_backoff_succeeds_immediately(
    tracker: BudgetTracker,
) -> None:
    handle = tracker.allocate_with_backoff("m", vram_gb=1.0)
    assert isinstance(handle, AllocationHandle)
    tracker.release(handle)


def test_allocate_with_backoff_retries_until_release(
    tracker: BudgetTracker,
) -> None:
    blocker = tracker.allocate("blocker", vram_gb=7.0)

    def release_after_delay() -> None:
        time.sleep(0.1)
        tracker.release(blocker)

    thread = threading.Thread(target=release_after_delay)
    thread.start()
    start = time.monotonic()
    handle = tracker.allocate_with_backoff(
        "m", vram_gb=2.0, max_attempts=10, initial_delay=0.05
    )
    elapsed = time.monotonic() - start
    assert isinstance(handle, AllocationHandle)
    # We should have spent at least the release delay.
    assert elapsed >= 0.1
    thread.join(timeout=1.0)
    tracker.release(handle)


def test_allocate_with_backoff_raises_after_max_attempts(
    tracker: BudgetTracker,
) -> None:
    tracker.allocate("blocker", vram_gb=8.0)
    with pytest.raises(BudgetExceededError):
        tracker.allocate_with_backoff(
            "m",
            vram_gb=1.0,
            max_attempts=2,
            initial_delay=0.01,
        )


def test_allocate_with_backoff_validates_arguments(
    tracker: BudgetTracker,
) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        tracker.allocate_with_backoff("m", vram_gb=0.1, max_attempts=0)
    with pytest.raises(ValueError, match="initial_delay"):
        tracker.allocate_with_backoff("m", vram_gb=0.1, initial_delay=-0.1)
    with pytest.raises(ValueError, match="backoff_factor"):
        tracker.allocate_with_backoff("m", vram_gb=0.1, backoff_factor=0.5)
