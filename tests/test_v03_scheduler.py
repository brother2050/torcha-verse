"""Tests for the v0.4.2 RuntimeScheduler skeleton.

Covers:

* :class:`InlineScheduler` happy path, exception propagation, and
  the "shut down" guard.
* :class:`ThreadPoolScheduler` happy path, parallel execution,
  ``shutdown`` semantics, and the
  ``max_workers``-must-be-positive validation.
* :func:`default_scheduler` returns a usable instance.
"""

from __future__ import annotations

import threading
import time

import pytest

from infrastructure.scheduler import (
    InlineScheduler,
    RuntimeScheduler,
    ThreadPoolScheduler,
    default_scheduler,
)


# ---------------------------------------------------------------------------
# InlineScheduler
# ---------------------------------------------------------------------------
def test_inline_runs_synchronously_and_returns_value() -> None:
    sched = InlineScheduler()
    future = sched.submit(pow, 2, 10)
    assert future.result(timeout=1.0) == 1024
    assert future.done()


def test_inline_propagates_exceptions() -> None:
    sched = InlineScheduler()
    future = sched.submit(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        future.result(timeout=1.0)


def test_inline_rejects_submit_after_shutdown() -> None:
    sched = InlineScheduler()
    sched.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        sched.submit(lambda: 1)


def test_inline_tracks_inflight_count() -> None:
    sched = InlineScheduler()
    observed: list = []

    def task() -> int:
        observed.append(sched.max_inflight)
        return len(observed)

    sched.submit(task).result(timeout=1.0)
    sched.submit(task).result(timeout=1.0)
    # Both tasks run sequentially, so ``max_inflight`` should never
    # exceed 1 in the inline implementation.
    assert sched.max_inflight == 1
    assert observed == [1, 1]


# ---------------------------------------------------------------------------
# ThreadPoolScheduler
# ---------------------------------------------------------------------------
def test_thread_pool_runs_in_parallel() -> None:
    sched = ThreadPoolScheduler(max_workers=4)
    start = time.monotonic()

    def slow() -> str:
        time.sleep(0.05)
        return "ok"

    futures = [sched.submit(slow) for _ in range(4)]
    elapsed = time.monotonic() - start
    # 4 tasks x 0.05s in parallel should complete in <0.1s rather
    # than 0.2s.
    assert elapsed < 0.15
    for f in futures:
        assert f.result(timeout=1.0) == "ok"
    sched.shutdown(wait=True)


def test_thread_pool_validates_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        ThreadPoolScheduler(max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        ThreadPoolScheduler(max_workers=-1)


def test_thread_pool_propagates_exceptions() -> None:
    sched = ThreadPoolScheduler(max_workers=2)
    future = sched.submit(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        future.result(timeout=1.0)
    sched.shutdown(wait=True)


def test_thread_pool_tracks_submitted_and_completed() -> None:
    sched = ThreadPoolScheduler(max_workers=2)
    f1 = sched.submit(lambda: None)
    f2 = sched.submit(lambda: None)
    f1.result(timeout=1.0)
    f2.result(timeout=1.0)
    # Both tasks completed.
    assert sched.submitted == 2
    assert sched.completed == 2
    sched.shutdown(wait=True)


def test_thread_pool_shutdown_can_be_re_entered() -> None:
    sched = ThreadPoolScheduler(max_workers=1)
    sched.shutdown()
    # Second shutdown is a no-op (executor already cleaned up) but
    # must not raise.
    sched.shutdown()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
def test_default_scheduler_is_runtime_scheduler_instance() -> None:
    sched = default_scheduler()
    assert isinstance(sched, RuntimeScheduler)
    assert sched.name in {"thread_pool", "inline"}


def test_runtime_scheduler_is_abstract() -> None:
    with pytest.raises(TypeError):
        RuntimeScheduler()  # type: ignore[abstract]
