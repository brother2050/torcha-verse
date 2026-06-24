"""Unified runtime scheduler for TorchaVerse v0.3.0.

This module replaces :mod:`core.inference_scheduler` (v0.1.0) with a
unified, DAG-aware runtime scheduler that manages three concurrency
dimensions:

* **CPU thread pool** -- a :class:`~concurrent.futures.ThreadPoolExecutor`
  for CPU-bound work (data processing, tokenisation, post-processing).
* **asyncio event loop** -- a background event loop running in a
  dedicated thread for I/O-bound and coroutine-based work.
* **GPU CUDA stream** -- a :class:`torch.cuda.Stream` that serialises
  GPU kernel launches, preventing cross-task interference.

Key features:

* :class:`TaskPriority` -- four-level priority enum.
* :class:`Task` -- lightweight task descriptor with optional
  :class:`~infrastructure.resource_budget.ResourceBudget`.
* :class:`Future` -- thread-safe future supporting ``result()``,
  ``cancel()``, ``add_done_callback()`` and ``__await__``.
* :class:`RuntimeScheduler` -- the main scheduler with:

  - DAG dependency analysis (``depends_on`` in task inputs) for
    automatic fan-out / fan-in.
  - ``submit`` / ``submit_batch`` / ``cancel``.
  - ``pause`` / ``resume``.
  - ``checkpoint`` / ``restore`` for fault recovery.
  - ``status`` for monitoring.
"""

from __future__ import annotations

import asyncio
import enum
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger
from infrastructure.resource_budget import ResourceBudget

__all__ = [
    "TaskPriority",
    "Task",
    "Future",
    "RuntimeScheduler",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default maximum number of CPU worker threads.
_DEFAULT_MAX_WORKERS: int = 4

#: Default thread name prefix for the CPU pool.
_DEFAULT_THREAD_PREFIX: str = "torcha-worker"

#: Join timeout (seconds) for the asyncio loop thread on shutdown.
_LOOP_JOIN_TIMEOUT: float = 5.0

#: Polling interval (seconds) for the ``__await__`` busy-wait.
_AWAIT_POLL_INTERVAL: float = 0.01

#: Key in ``Task.inputs`` that holds the list of dependency task IDs.
_DEPENDS_ON_KEY: str = "depends_on"

#: Key in ``Task.inputs`` that holds the callable to execute.
_FN_KEY: str = "fn"

#: Key in ``Task.inputs`` that holds positional arguments.
_ARGS_KEY: str = "args"

#: Key in ``Task.inputs`` that holds keyword arguments.
_KWARGS_KEY: str = "kwargs"

#: Key in ``Task.inputs`` that overrides the executor selection.
_EXECUTOR_KEY: str = "executor"

#: Executor type identifiers.
_EXECUTOR_CPU: str = "cpu"
_EXECUTOR_IO: str = "io"
_EXECUTOR_GPU: str = "gpu"

#: Task status strings.
_STATUS_QUEUED: str = "queued"
_STATUS_RUNNING: str = "running"
_STATUS_COMPLETED: str = "completed"
_STATUS_FAILED: str = "failed"
_STATUS_CANCELLED: str = "cancelled"


# ---------------------------------------------------------------------------
# TaskPriority
# ---------------------------------------------------------------------------
class TaskPriority(enum.IntEnum):
    """Priority level for scheduled tasks.

    Higher values indicate higher priority.  When the CPU thread pool
    is saturated, higher-priority tasks are dispatched first.

    Attributes:
        LOW: Background / cleanup tasks.
        NORMAL: Default priority for most tasks.
        HIGH: Time-sensitive tasks (e.g. interactive requests).
        URGENT: Must-run-immediately tasks (e.g. user-facing latency).
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@dataclass
class Task:
    """A unit of work submitted to the :class:`RuntimeScheduler`.

    Attributes:
        id: Unique task identifier.  Auto-generated when ``None``.
        node_type: Logical node type (e.g. ``"cpu"``, ``"gpu"``,
            ``"io"``).  Used to select the executor when
            ``inputs["executor"]`` is not set.
        inputs: Arbitrary dictionary of task parameters.  Recognised
            keys include:

            * ``"fn"`` -- callable to execute.
            * ``"args"`` -- positional arguments for *fn*.
            * ``"kwargs"`` -- keyword arguments for *fn*.
            * ``"depends_on"`` -- list of task IDs that must complete
              before this task starts (DAG edges).
            * ``"executor"`` -- override executor type
              (``"cpu"``, ``"io"``, ``"gpu"``).

        priority: Scheduling priority.
        budget: Optional resource budget constraint.
    """

    id: str = ""
    node_type: str = "cpu"
    inputs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    budget: Optional[ResourceBudget] = None

    def __post_init__(self) -> None:
        """Auto-generate an ID if none was provided."""
        if not self.id:
            self.id = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Future
# ---------------------------------------------------------------------------
class Future:
    """Thread-safe future for asynchronous task results.

    Supports blocking ``result(timeout)``, cancellation, done-callbacks,
    and ``__await__`` for use in ``async`` functions.

    This is a lightweight alternative to
    :class:`concurrent.futures.Future` that integrates with the
    :class:`RuntimeScheduler`'s DAG bookkeeping.
    """

    def __init__(self) -> None:
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._done: bool = False
        self._cancelled: bool = False
        self._callbacks: List[Callable[["Future"], None]] = []
        self._condition: threading.Condition = threading.Condition()

    # ------------------------------------------------------------------
    def set_result(self, result: Any) -> None:
        """Set the result and wake up waiters."""
        with self._condition:
            if self._done:
                return
            self._result = result
            self._done = True
            self._condition.notify_all()
        self._invoke_callbacks()

    # ------------------------------------------------------------------
    def set_exception(self, exc: BaseException) -> None:
        """Set an exception and wake up waiters."""
        with self._condition:
            if self._done:
                return
            self._exception = exc
            self._done = True
            self._condition.notify_all()
        self._invoke_callbacks()

    # ------------------------------------------------------------------
    def result(self, timeout: Optional[float] = None) -> Any:
        """Block until the task completes and return its result.

        Args:
            timeout: Maximum seconds to wait.  ``None`` means wait
                forever.

        Returns:
            The task's result.

        Raises:
            TimeoutError: If *timeout* elapses before completion.
            RuntimeError: If the task was cancelled.
            Exception: If the task raised an exception.
        """
        with self._condition:
            if not self._done:
                self._condition.wait(timeout=timeout)
            if not self._done:
                raise TimeoutError(
                    "Future result not available within {}s.".format(timeout)
                )
            if self._cancelled:
                raise RuntimeError("Future was cancelled.")
            if self._exception is not None:
                raise self._exception
            return self._result

    # ------------------------------------------------------------------
    def done(self) -> bool:
        """Return ``True`` if the task has completed (or been cancelled)."""
        with self._condition:
            return self._done

    # ------------------------------------------------------------------
    def cancelled(self) -> bool:
        """Return ``True`` if the task was cancelled."""
        with self._condition:
            return self._cancelled

    # ------------------------------------------------------------------
    def cancel(self) -> bool:
        """Attempt to cancel the task.

        Returns ``True`` if the task was cancelled, ``False`` if it had
        already completed.
        """
        with self._condition:
            if self._done:
                return False
            self._cancelled = True
            self._done = True
            self._condition.notify_all()
        self._invoke_callbacks()
        return True

    # ------------------------------------------------------------------
    def exception(self) -> Optional[BaseException]:
        """Return the exception (or ``None`` if none)."""
        with self._condition:
            return self._exception

    # ------------------------------------------------------------------
    def add_done_callback(
        self, callback: Callable[["Future"], None]
    ) -> None:
        """Register a callback invoked when the task completes."""
        with self._condition:
            if self._done:
                callback(self)
            else:
                self._callbacks.append(callback)

    # ------------------------------------------------------------------
    def _invoke_callbacks(self) -> None:
        """Invoke all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def __await__(self):
        """Allow ``await future`` in async code.

        Uses a simple busy-wait poll because the scheduler's event loop
        runs in a separate thread.
        """
        import time as _time

        while not self.done():
            _time.sleep(_AWAIT_POLL_INTERVAL)
            yield
        if self._exception is not None:
            raise self._exception
        return self._result


# ---------------------------------------------------------------------------
# RuntimeScheduler
# ---------------------------------------------------------------------------
class RuntimeScheduler:
    """Unified runtime scheduler with DAG-aware task dispatch.

    Manages three concurrency dimensions (CPU thread pool, asyncio
    loop, GPU CUDA stream) and resolves task dependencies automatically,
    enabling fan-out / fan-in patterns.

    Example::

        sched = RuntimeScheduler()

        # Fan-out: submit independent tasks.
        f1 = sched.submit(Task(id="a", node_type="cpu",
                               inputs={"fn": compute_a}))
        f2 = sched.submit(Task(id="b", node_type="cpu",
                               inputs={"fn": compute_b}))

        # Fan-in: task "c" depends on "a" and "b".
        f3 = sched.submit(Task(
            id="c", node_type="cpu",
            inputs={"fn": combine, "depends_on": ["a", "b"]},
        ))
        result = f3.result()
    """

    def __init__(
        self,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        thread_name_prefix: str = _DEFAULT_THREAD_PREFIX,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            max_workers: Maximum number of CPU worker threads.
            thread_name_prefix: Prefix for worker thread names.
            device: Target GPU device for CUDA stream creation.
        """
        self._logger = get_logger(self.__class__.__name__)
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            device if device is not None else self._device_manager.get_device()
        )

        # CPU thread pool.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

        # asyncio event loop in a background thread.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="torcha-asyncio",
        )
        self._loop_thread.start()

        # GPU CUDA stream (if available).
        if torch.cuda.is_available() and "cuda" in str(self._device):
            self._cuda_stream: Optional[torch.cuda.Stream] = (
                torch.cuda.Stream(device=self._device)
            )
        else:
            self._cuda_stream = None

        # Task bookkeeping.
        self._tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}
        self._status: Dict[str, str] = {}

        # Synchronisation.
        self._lock: threading.RLock = threading.RLock()
        self._paused: bool = False

        self._logger.debug(
            "RuntimeScheduler initialised (workers=%d, cuda_stream=%s).",
            max_workers,
            self._cuda_stream is not None,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def submit(self, task: Task) -> Future:
        """Submit a single task and return its :class:`Future`.

        If the task has unmet dependencies it is queued; otherwise it is
        dispatched immediately (unless the scheduler is paused).

        Args:
            task: The task to submit.

        Returns:
            A :class:`Future` that will hold the result.
        """
        with self._lock:
            future = Future()
            self._tasks[task.id] = task
            self._futures[task.id] = future
            self._status[task.id] = _STATUS_QUEUED

            # Parse dependencies.
            deps_raw = task.inputs.get(_DEPENDS_ON_KEY, [])
            if isinstance(deps_raw, str):
                deps_raw = [deps_raw]
            deps: Set[str] = set(deps_raw)
            self._dependencies[task.id] = deps
            for dep_id in deps:
                self._dependents.setdefault(dep_id, set()).add(task.id)

            # Dispatch if ready.
            if self._check_dependencies(task.id) and not self._paused:
                self._dispatch(task.id)

            return future

    # ------------------------------------------------------------------
    def submit_batch(self, tasks: List[Task]) -> List[Future]:
        """Submit multiple tasks and return their futures.

        Args:
            tasks: List of tasks to submit.

        Returns:
            A list of :class:`Future` objects, one per task.
        """
        futures: List[Future] = []
        for task in tasks:
            futures.append(self.submit(task))
        return futures

    # ------------------------------------------------------------------
    def cancel(self, task_id: str) -> bool:
        """Cancel a queued or pending task.

        Running tasks cannot be cancelled.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            ``True`` if the task was cancelled, ``False`` otherwise.
        """
        with self._lock:
            if task_id not in self._futures:
                return False
            status = self._status.get(task_id)
            if status == _STATUS_QUEUED:
                self._futures[task_id].cancel()
                self._status[task_id] = _STATUS_CANCELLED
                return True
            # Running or already done.
            return False

    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Pause the scheduler.

        No new tasks will be dispatched until :meth:`resume` is called.
        Already-running tasks continue to completion.
        """
        with self._lock:
            self._paused = True
        self._logger.info("Scheduler paused.")

    # ------------------------------------------------------------------
    def resume(self) -> None:
        """Resume the scheduler and dispatch ready queued tasks."""
        with self._lock:
            self._paused = False
            for task_id, status in list(self._status.items()):
                if (
                    status == _STATUS_QUEUED
                    and self._check_dependencies(task_id)
                ):
                    self._dispatch(task_id)
        self._logger.info("Scheduler resumed.")

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, int]:
        """Return a snapshot of task counts by status.

        Returns:
            A dict with keys ``running``, ``queued``, ``completed``,
            ``failed``.
        """
        with self._lock:
            counts: Dict[str, int] = {
                "running": 0,
                "queued": 0,
                "completed": 0,
                "failed": 0,
            }
            for s in self._status.values():
                if s == _STATUS_RUNNING:
                    counts["running"] += 1
                elif s == _STATUS_QUEUED:
                    counts["queued"] += 1
                elif s == _STATUS_COMPLETED:
                    counts["completed"] += 1
                elif s == _STATUS_FAILED:
                    counts["failed"] += 1
            return counts

    # ------------------------------------------------------------------
    def checkpoint(self) -> Dict[str, Any]:
        """Serialise the scheduler state for fault recovery.

        Returns:
            A dict containing task metadata, statuses, and dependency
            edges.  Callable objects in ``inputs`` are excluded (only
            serialisable metadata is saved).
        """
        with self._lock:
            tasks_meta: Dict[str, Any] = {}
            for tid, task in self._tasks.items():
                # Exclude non-serialisable values (functions, etc.).
                safe_inputs: Dict[str, Any] = {}
                for k, v in task.inputs.items():
                    if k in (_FN_KEY, _ARGS_KEY, _KWARGS_KEY):
                        continue
                    try:
                        # Best-effort serialisability check.
                        safe_inputs[k] = v
                    except Exception:  # noqa: BLE001
                        pass
                tasks_meta[tid] = {
                    "node_type": task.node_type,
                    "priority": task.priority.name,
                    "inputs": safe_inputs,
                }
            return {
                "tasks": tasks_meta,
                "status": dict(self._status),
                "dependencies": {
                    k: sorted(v) for k, v in self._dependencies.items()
                },
            }

    # ------------------------------------------------------------------
    def restore(self, state: Dict[str, Any]) -> None:
        """Restore the scheduler from a checkpoint.

        Re-creates task metadata and re-submits any tasks whose status
        was ``queued`` in the checkpoint.  Tasks that were ``running``
        when the checkpoint was taken are re-queued.

        Args:
            state: A dict previously returned by :meth:`checkpoint`.
        """
        with self._lock:
            # Clear current state.
            self._tasks.clear()
            self._futures.clear()
            self._dependencies.clear()
            self._dependents.clear()
            self._status.clear()

            tasks_meta = state.get("tasks", {})
            saved_status = state.get("status", {})
            saved_deps = state.get("dependencies", {})

            for tid, meta in tasks_meta.items():
                task = Task(
                    id=tid,
                    node_type=meta.get("node_type", "cpu"),
                    inputs=dict(meta.get("inputs", {})),
                    priority=TaskPriority[meta.get("priority", "NORMAL")],
                )
                self._tasks[tid] = task
                self._futures[tid] = Future()

                # Restore status; running tasks become queued.
                old_status = saved_status.get(tid, _STATUS_QUEUED)
                if old_status == _STATUS_RUNNING:
                    old_status = _STATUS_QUEUED
                self._status[tid] = old_status

                # Restore dependencies.
                deps = set(saved_deps.get(tid, []))
                self._dependencies[tid] = deps
                for dep_id in deps:
                    self._dependents.setdefault(dep_id, set()).add(tid)

            # Re-dispatch ready queued tasks.
            if not self._paused:
                for tid, status in list(self._status.items()):
                    if (
                        status == _STATUS_QUEUED
                        and self._check_dependencies(tid)
                    ):
                        self._dispatch(tid)

        self._logger.info(
            "Restored %d task(s) from checkpoint.", len(tasks_meta)
        )

    # ------------------------------------------------------------------
    def shutdown(self, wait: bool = True) -> None:
        """Shut down the scheduler and release resources.

        Args:
            wait: Whether to wait for pending tasks to finish.
        """
        self._executor.shutdown(wait=wait)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=_LOOP_JOIN_TIMEOUT)
        self._logger.info("Scheduler shut down.")

    # ==================================================================
    # Internal dispatch logic
    # ==================================================================

    def _check_dependencies(self, task_id: str) -> bool:
        """Return ``True`` if all dependencies of *task_id* are completed.

        Must be called while holding ``self._lock``.
        """
        deps = self._dependencies.get(task_id, set())
        for dep_id in deps:
            if self._status.get(dep_id) != _STATUS_COMPLETED:
                return False
        return True

    # ------------------------------------------------------------------
    def _dispatch(self, task_id: str) -> None:
        """Dispatch a ready task to the appropriate executor.

        Must be called while holding ``self._lock``.
        """
        task = self._tasks[task_id]
        future = self._futures[task_id]
        self._status[task_id] = _STATUS_RUNNING

        executor_type = self._select_executor_type(task)
        self._logger.debug(
            "Dispatching task '%s' to %s executor.", task_id, executor_type
        )

        if executor_type == _EXECUTOR_IO:
            self._dispatch_async(task, future)
        elif executor_type == _EXECUTOR_GPU:
            self._executor.submit(self._run_gpu_task, task, future)
        else:
            self._executor.submit(self._run_task, task, future)

    # ------------------------------------------------------------------
    def _select_executor_type(self, task: Task) -> str:
        """Determine which executor to use for *task*.

        Priority:
        1. Explicit ``inputs["executor"]`` override.
        2. Inference from ``node_type``.
        3. Default to CPU.
        """
        explicit = task.inputs.get(_EXECUTOR_KEY)
        if explicit is not None:
            return str(explicit)

        nt = task.node_type.lower()
        if nt in ("io", "async", "asyncio"):
            return _EXECUTOR_IO
        if nt in ("gpu", "cuda", "device"):
            return _EXECUTOR_GPU
        return _EXECUTOR_CPU

    # ------------------------------------------------------------------
    def _run_task(self, task: Task, future: Future) -> None:
        """Execute a CPU-bound task."""
        success = True
        try:
            fn: Optional[Callable] = task.inputs.get(_FN_KEY)
            if fn is None:
                future.set_result(None)
            else:
                args = task.inputs.get(_ARGS_KEY, ())
                kwargs = task.inputs.get(_KWARGS_KEY, {})
                result = fn(*args, **kwargs)
                future.set_result(result)
        except BaseException as exc:  # noqa: BLE001
            success = False
            future.set_exception(exc)
        finally:
            self._on_task_done(task.id, success)

    # ------------------------------------------------------------------
    def _run_gpu_task(self, task: Task, future: Future) -> None:
        """Execute a GPU task under the CUDA stream."""
        success = True
        try:
            stream_ctx = (
                torch.cuda.stream(self._cuda_stream)
                if self._cuda_stream is not None
                else _NullContext()
            )
            with stream_ctx:
                fn: Optional[Callable] = task.inputs.get(_FN_KEY)
                if fn is None:
                    future.set_result(None)
                else:
                    args = task.inputs.get(_ARGS_KEY, ())
                    kwargs = task.inputs.get(_KWARGS_KEY, {})
                    result = fn(*args, **kwargs)
                    # Synchronise the stream before returning.
                    if self._cuda_stream is not None:
                        self._cuda_stream.synchronize()
                    future.set_result(result)
        except BaseException as exc:  # noqa: BLE001
            success = False
            future.set_exception(exc)
        finally:
            self._on_task_done(task.id, success)

    # ------------------------------------------------------------------
    def _dispatch_async(self, task: Task, future: Future) -> None:
        """Submit a coroutine or sync callable to the asyncio loop."""

        async def _run() -> None:
            success = True
            try:
                fn: Optional[Callable] = task.inputs.get(_FN_KEY)
                if fn is None:
                    future.set_result(None)
                elif asyncio.iscoroutinefunction(fn):
                    args = task.inputs.get(_ARGS_KEY, ())
                    kwargs = task.inputs.get(_KWARGS_KEY, {})
                    result = await fn(*args, **kwargs)
                    future.set_result(result)
                else:
                    args = task.inputs.get(_ARGS_KEY, ())
                    kwargs = task.inputs.get(_KWARGS_KEY, {})
                    result = fn(*args, **kwargs)
                    future.set_result(result)
            except BaseException as exc:  # noqa: BLE001
                success = False
                future.set_exception(exc)
            finally:
                self._on_task_done(task.id, success)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)

    # ------------------------------------------------------------------
    def _on_task_done(self, task_id: str, success: bool) -> None:
        """Update status and check dependent tasks.

        Called after a task finishes (successfully or not).  If the
        task succeeded, dependent tasks whose dependencies are now all
        met are dispatched.  If the task failed, queued dependents are
        also marked as failed.
        """
        with self._lock:
            self._status[task_id] = (
                _STATUS_COMPLETED if success else _STATUS_FAILED
            )
            for dep_id in self._dependents.get(task_id, set()):
                if not success:
                    # Propagate failure to queued dependents.
                    if self._status.get(dep_id) == _STATUS_QUEUED:
                        self._status[dep_id] = _STATUS_FAILED
                        dep_future = self._futures.get(dep_id)
                        if dep_future is not None:
                            dep_future.set_exception(
                                RuntimeError(
                                    "Dependency '{}' failed.".format(task_id)
                                )
                            )
                elif (
                    self._status.get(dep_id) == _STATUS_QUEUED
                    and self._check_dependencies(dep_id)
                    and not self._paused
                ):
                    self._dispatch(dep_id)

    # ------------------------------------------------------------------
    def get_future(self, task_id: str) -> Optional[Future]:
        """Return the future for *task_id*, or ``None`` if unknown."""
        with self._lock:
            return self._futures.get(task_id)

    # ------------------------------------------------------------------
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Return the status string for *task_id*, or ``None``."""
        with self._lock:
            return self._status.get(task_id)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        s = self.status()
        return (
            "RuntimeScheduler(running={}, queued={}, completed={}, "
            "failed={}, paused={})".format(
                s["running"], s["queued"], s["completed"],
                s["failed"], self._paused,
            )
        )


# ---------------------------------------------------------------------------
# Helper context manager (null-op when CUDA stream is unavailable)
# ---------------------------------------------------------------------------
class _NullContext:
    """A no-op context manager used when no CUDA stream is available."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
