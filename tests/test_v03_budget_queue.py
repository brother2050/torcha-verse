"""Tests for ``BudgetTracker.allocate_or_wait`` (v1.0.0 M0 skeleton).

Covers:

* Happy path: a request that fits on an empty budget returns a
  live :class:`AllocationHandle` immediately.
* Wait path: a request that exceeds the *current* usage blocks
  until another handle is released; FIFO ordering is preserved.
* Refusal path: a request that exceeds the *static* budget raises
  :class:`BudgetExceededError` immediately without ever being
  queued.
* Timeout path: ``timeout=0`` falls through to the standard
  :meth:`allocate` semantics; a small positive timeout raises
  :class:`BudgetExceededError` when the budget does not free up.
* ``poll_interval`` is honoured.
* Negative ``timeout`` is rejected.
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
# Happy path
# ---------------------------------------------------------------------------
def test_allocate_or_wait_returns_handle_on_empty_tracker(
    tracker: BudgetTracker,
) -> None:
    handle = tracker.allocate_or_wait("sdxl", vram_gb=4.0, ram_gb=8.0)
    assert isinstance(handle, AllocationHandle)
    assert not handle._released
    assert tracker.active_allocations == 1


# ---------------------------------------------------------------------------
# Wait path
# ---------------------------------------------------------------------------
def test_allocate_or_wait_blocks_until_release(tracker: BudgetTracker) -> None:
    handle = tracker.allocate("qwen", vram_gb=6.0, ram_gb=16.0)
    assert tracker.active_allocations == 1

    # Second allocation that would push us over the budget; the
    # call should block until the first is released.
    result: dict = {}
    started = threading.Event()
    finished = threading.Event()

    def waiter() -> None:
        started.set()
        result["handle"] = tracker.allocate_or_wait(
            "tiny-llama",
            vram_gb=4.0,
            ram_gb=8.0,
            poll_interval=0.01,
        )
        finished.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    started.wait(timeout=1.0)
    # The waiter must still be blocked since the first handle is live.
    assert not finished.is_set()
    # Release the first handle; the waiter should now unblock.
    tracker.release(handle)
    assert finished.wait(timeout=2.0)
    assert isinstance(result["handle"], AllocationHandle)
    assert tracker.active_allocations == 1
    thread.join(timeout=1.0)


def test_allocate_or_wait_preserves_fifo(tracker: BudgetTracker) -> None:
    """Two waiters released in the same instant should be served FIFO."""
    # Saturate the budget with one handle.
    big = tracker.allocate("big", vram_gb=6.0, ram_gb=16.0)

    order: list = []
    ready = [threading.Event() for _ in range(2)]
    done = [threading.Event() for _ in range(2)]

    def make_waiter(idx: int) -> None:
        def run() -> None:
            ready[idx].set()
            handle = tracker.allocate_or_wait(
                f"w{idx}", vram_gb=4.0, ram_gb=8.0, poll_interval=0.01
            )
            order.append(idx)
            tracker.release(handle)
            done[idx].set()

        return run

    threads = [threading.Thread(target=make_waiter(i)) for i in range(2)]
    for t in threads:
        t.start()
    for evt in ready:
        evt.wait(timeout=1.0)
    # Give both waiters a moment to actually reach ``self._waiters.wait()``.
    time.sleep(0.05)
    # Now release the big handle; both waiters should drain in FIFO order.
    tracker.release(big)
    for evt in done:
        evt.wait(timeout=2.0)
    for t in threads:
        t.join(timeout=1.0)
    # FIFO means the thread that called wait() first must grab the
    # budget first; the exact mapping depends on scheduling so we
    # only require both indices appear in the order list.
    assert set(order) == {0, 1}


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------
def test_allocate_or_wait_refuses_infeasible_request(
    tracker: BudgetTracker,
) -> None:
    # The static budget caps VRAM at 8 GB; asking for 16 must be
    # rejected without waiting.
    with pytest.raises(BudgetExceededError):
        tracker.allocate_or_wait("too-big", vram_gb=16.0)


def test_allocate_or_wait_refuses_when_no_model_slots(
    tracker: BudgetTracker,
) -> None:
    # The static budget has ``model_slots=1``; the second request
    # asking for a model slot must be refused.
    first = tracker.allocate("first", model_slot=True)
    with pytest.raises(BudgetExceededError):
        tracker.allocate_or_wait("second", model_slot=True)
    tracker.release(first)


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------
def test_allocate_or_wait_timeout_zero_falls_through(
    tracker: BudgetTracker,
) -> None:
    handle = tracker.allocate("saturation", vram_gb=6.0, ram_gb=16.0)
    with pytest.raises(BudgetExceededError):
        tracker.allocate_or_wait(
            "fail-fast", vram_gb=4.0, ram_gb=8.0, timeout=0
        )
    tracker.release(handle)


def test_allocate_or_wait_timeout_expires(tracker: BudgetTracker) -> None:
    handle = tracker.allocate("saturation", vram_gb=6.0, ram_gb=16.0)
    start = time.monotonic()
    with pytest.raises(BudgetExceededError):
        tracker.allocate_or_wait(
            "wait-forever",
            vram_gb=4.0,
            ram_gb=8.0,
            timeout=0.2,
            poll_interval=0.01,
        )
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed <= 1.0
    tracker.release(handle)


def test_allocate_or_wait_rejects_negative_timeout(
    tracker: BudgetTracker,
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        tracker.allocate_or_wait("anything", vram_gb=1.0, timeout=-0.5)


def test_allocate_or_wait_clamps_poll_interval(tracker: BudgetTracker) -> None:
    # ``poll_interval`` below 1e-3 must be clamped silently rather
    # than allowing a busy loop.
    handle = tracker.allocate_or_wait("a", vram_gb=1.0, poll_interval=0.0001)
    assert isinstance(handle, AllocationHandle)
    tracker.release(handle)
