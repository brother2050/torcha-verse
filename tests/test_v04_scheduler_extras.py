"""Tests for the v0.4.3 ProcessPoolScheduler addition.

ProcessPoolScheduler must:

* validate ``max_workers`` like ThreadPoolScheduler does;
* run a picklable task in a separate process and return its result;
* propagate exceptions from the worker process;
* raise a clear ``RuntimeError`` when the task is not picklable;
* report submitted / completed counts and shut down cleanly.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest

from infrastructure.scheduler import (
    ProcessPoolScheduler,
    RuntimeScheduler,
    ThreadPoolScheduler,
)


def _picklable_add(x: int, y: int) -> int:
    return x + y


def _picklable_kw(**kwargs: int) -> int:
    return sum(kwargs.values())


def _picklable_boom() -> None:
    raise RuntimeError("worker-boom")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_process_pool_validates_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        ProcessPoolScheduler(max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        ProcessPoolScheduler(max_workers=-1)


def test_process_pool_name_and_type() -> None:
    sched = ProcessPoolScheduler(max_workers=1)
    assert sched.name == "process_pool"
    assert isinstance(sched, RuntimeScheduler)
    # ``max_workers=1`` so we only spin up one worker process.
    assert sched.max_workers == 1
    sched.shutdown()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_process_pool_runs_picklable_task() -> None:
    sched = ProcessPoolScheduler(max_workers=2)
    future = sched.submit(_picklable_add, 2, 3)
    assert isinstance(future, Future)
    assert future.result(timeout=10.0) == 5
    assert future.done()
    sched.shutdown()


def test_process_pool_runs_picklable_task_with_kwargs() -> None:
    sched = ProcessPoolScheduler(max_workers=1)
    future = sched.submit(_picklable_kw, a=1, b=2, c=3)
    assert future.result(timeout=10.0) == 6
    sched.shutdown()


def test_process_pool_propagates_exceptions() -> None:
    sched = ProcessPoolScheduler(max_workers=1)
    future = sched.submit(_picklable_boom)
    with pytest.raises(RuntimeError, match="worker-boom"):
        future.result(timeout=10.0)
    sched.shutdown()


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------
def test_process_pool_rejects_unpicklable_with_clear_error() -> None:
    sched = ProcessPoolScheduler(max_workers=1)

    class _Unpicklable:
        """An object whose ``__reduce__`` raises so pickle fails."""

        def __reduce__(self):  # pragma: no cover - error path
            raise pickle.PicklingError("intentionally unpicklable")

    unpicklable = _Unpicklable()

    def _unpicklable_task(_=unpicklable) -> int:  # type: ignore[no-redef]
        return 1

    with pytest.raises(RuntimeError, match="refused to pickle"):
        sched.submit(_unpicklable_task)
    sched.shutdown()


# ---------------------------------------------------------------------------
# Stats + shutdown
# ---------------------------------------------------------------------------
def test_process_pool_tracks_submitted_and_completed() -> None:
    sched = ProcessPoolScheduler(max_workers=2)
    f1 = sched.submit(_picklable_add, 1, 1)
    f2 = sched.submit(_picklable_add, 2, 2)
    f1.result(timeout=10.0)
    f2.result(timeout=10.0)
    assert sched.submitted == 2
    assert sched.completed == 2
    sched.shutdown()


def test_process_pool_shutdown_is_idempotent() -> None:
    sched = ProcessPoolScheduler(max_workers=1)
    sched.shutdown()
    # Second shutdown must not raise.
    sched.shutdown()


# ---------------------------------------------------------------------------
# Cross-scheduler compatibility
# ---------------------------------------------------------------------------
def test_process_pool_is_a_runtime_scheduler() -> None:
    # This is the property that lets ``default_scheduler()`` switch
    # between Inline / ThreadPool / ProcessPool at startup.
    sched = ProcessPoolScheduler(max_workers=1)
    assert isinstance(sched, RuntimeScheduler)
    # ``ThreadPoolScheduler`` is the other concrete implementation.
    other = ThreadPoolScheduler(max_workers=1)
    assert isinstance(other, RuntimeScheduler)
    sched.shutdown()
    other.shutdown()
