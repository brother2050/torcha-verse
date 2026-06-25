"""Runtime scheduler abstraction (v1.0.0 M1 skeleton, shipped in v0.4.2).

This module defines :class:`RuntimeScheduler`, a small protocol that
hides the choice of execution substrate behind a single
``submit(callable, *args, **kwargs) -> Future`` entry point, plus two
reference implementations:

* :class:`InlineScheduler` — runs every task on the calling thread.
  Useful for tests and for the "fully serial" mode used by the
  v0.4.x P0 examples.
* :class:`ThreadPoolScheduler` — backed by a
  :class:`concurrent.futures.ThreadPoolExecutor`.  The v0.4.x default
  for the ``serving.app`` server.

The v1.0.0 M1 deliverable will add :class:`ProcessPoolScheduler` and
:class:`AsyncIOScheduler`; the v0.4.2 skeleton intentionally keeps
the surface area small so the protocol can be re-used as a
dependency-injection seam.
"""

from __future__ import annotations

import abc
import concurrent.futures
import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional

from .logger import get_logger

__all__ = [
    "RuntimeScheduler",
    "InlineScheduler",
    "ThreadPoolScheduler",
    "default_scheduler",
]


class RuntimeScheduler(abc.ABC):
    """Abstract base class for runtime schedulers.

    Implementations are expected to be safe to call from multiple
    threads concurrently.
    """

    @abc.abstractmethod
    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future:
        """Schedule ``fn(*args, **kwargs)`` for execution.

        Args:
            fn: The callable to run.  Must be picklable when the
                implementation runs in a separate process.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            A :class:`concurrent.futures.Future` that resolves to
            the return value of ``fn`` (or raises its exception).
        """

    @abc.abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Release the underlying resources.

        Implementations may be re-submitted after :meth:`shutdown`
        only if explicitly documented.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable scheduler name (e.g. ``"inline"``)."""


class InlineScheduler(RuntimeScheduler):
    """A scheduler that runs every task synchronously on the caller.

    Intended for tests and for the v0.4.x P0 example path where the
    pipeline is intentionally serial.
    """

    def __init__(self) -> None:
        self._logger = get_logger("infrastructure.scheduler.inline")
        self._shutdown: bool = False
        self._lock: threading.RLock = threading.RLock()
        # ``_inflight`` lets tests assert that all submitted work
        # has been observed, even when the scheduler is inline.
        self._inflight: int = 0
        self._max_inflight: int = 0

    @property
    def name(self) -> str:
        return "inline"

    @property
    def max_inflight(self) -> int:
        """Highest number of overlapping tasks observed (always 1 here)."""
        with self._lock:
            return self._max_inflight

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("InlineScheduler is shut down.")
            self._inflight += 1
            self._max_inflight = max(self._max_inflight, self._inflight)
        future: Future = Future()
        try:
            if not future.set_running_or_notify_cancel():
                return future
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate everything
                future.set_exception(exc)
            else:
                future.set_result(result)
        finally:
            with self._lock:
                self._inflight -= 1
        return future

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
        self._logger.debug("InlineScheduler shut down (wait=%s).", wait)


class ThreadPoolScheduler(RuntimeScheduler):
    """A scheduler backed by a :class:`ThreadPoolExecutor`.

    The thread pool is created lazily on the first :meth:`submit`
    call so that constructing the scheduler does not immediately
    spin up worker threads (which is useful in unit tests).
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        if max_workers is not None and max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {max_workers}.")
        self._max_workers: Optional[int] = max_workers
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.scheduler.thread_pool")
        self._submitted: int = 0
        self._completed: int = 0

    @property
    def name(self) -> str:
        return "thread_pool"

    @property
    def max_workers(self) -> Optional[int]:
        """Configured maximum number of worker threads."""
        return self._max_workers

    @property
    def submitted(self) -> int:
        """Number of tasks that have been submitted so far."""
        with self._lock:
            return self._submitted

    @property
    def completed(self) -> int:
        """Number of submitted tasks that have completed."""
        with self._lock:
            return self._completed

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="torcha-sched",
                )
            return self._executor

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future:
        with self._lock:
            self._submitted += 1

        executor = self._ensure_executor()
        future = executor.submit(fn, *args, **kwargs)
        # ``add_done_callback`` runs on the worker thread; keep
        # the counter updates minimal so we do not contend on the
        # lock for hot paths.
        future.add_done_callback(self._on_done)
        return future

    def _on_done(self, _future: Future) -> None:
        with self._lock:
            self._completed += 1

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait)
        self._logger.debug(
            "ThreadPoolScheduler shut down (wait=%s, submitted=%d, completed=%d).",
            wait,
            self.submitted,
            self.completed,
        )


#: Default process-wide :class:`ThreadPoolScheduler` used by
#: ``serving.app`` and any pipeline that does not pass an explicit
#: ``scheduler`` argument.  Tests should construct their own
#: :class:`InlineScheduler` instead of mutating this global.
_default_scheduler: RuntimeScheduler = ThreadPoolScheduler(max_workers=4)


def default_scheduler() -> RuntimeScheduler:
    """Return the process-wide default :class:`RuntimeScheduler`."""
    return _default_scheduler
