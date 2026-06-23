"""Inference scheduling engine for TorchaVerse.

This module provides :class:`InferenceScheduler`, the central engine that
manages all inference requests across every modality (text, image, audio,
video).  Key features:

* **Priority queue** -- Requests are ordered by priority, allowing
  high-priority requests to jump the queue.
* **Automatic batching** -- Requests of the same modality are merged
  into batches to maximise GPU utilisation.
* **Continuous batching** -- New requests can be inserted into an
  in-flight batch, and completed requests are removed, without waiting
  for the entire batch to finish.
* **Streaming output** -- Supports token-by-token (text), frame-by-frame
  (audio), and segment-by-segment (video) streaming via async generators.
* **Async execution** -- The :meth:`submit` method returns a
  :class:`Future` that resolves when the request completes.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

from infrastructure.logger import get_logger

__all__ = [
    "InferenceRequest",
    "InferenceResult",
    "RequestStatus",
    "Future",
    "InferenceScheduler",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RequestStatus(Enum):
    """Lifecycle status of an inference request."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(order=True)
class InferenceRequest:
    """A single inference request.

    Attributes:
        request_id: Unique identifier (auto-generated when not provided).
        modality: The modality (``"text"``, ``"image"``, ``"audio"``,
            ``"video"``).
        prompt: The input prompt or conditioning data.
        params: Additional generation parameters (temperature, top_k,
            num_inference_steps, etc.).
        priority: Priority level (higher = more urgent).  Default 0.
        stream: Whether to stream the output.
        timestamp: Submission time (set automatically).
        status: Current request status.
    """

    modality: str = field(compare=False)
    prompt: Any = field(compare=False)
    request_id: str = field(default_factory=lambda: _generate_id(), compare=False)
    params: Dict[str, Any] = field(default_factory=dict, compare=False)
    priority: int = field(default=0)
    stream: bool = field(default=False, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    status: RequestStatus = field(
        default=RequestStatus.PENDING, compare=False
    )

    def __post_init__(self) -> None:
        self.modality = self.modality.lower().strip()


@dataclass
class InferenceResult:
    """The result of an inference request.

    Attributes:
        request_id: The originating request id.
        output: The generated output (tensor, text, etc.).
        status: Final status.
        error: Error message when ``status == FAILED``.
        metadata: Additional metadata (timing, tokens generated, etc.).
    """

    request_id: str
    output: Any = None
    status: RequestStatus = RequestStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Future
# ---------------------------------------------------------------------------
class Future:
    """A simple future for asynchronous inference results.

    Supports both synchronous (``result()``) and asynchronous
    (``await``) waiting, as well as callback registration.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id: str = request_id
        self._done: bool = False
        self._cancelled: bool = False
        self._result: Optional[InferenceResult] = None
        self._callbacks: List[Callable[[InferenceResult], None]] = []
        self._event: threading.Event = threading.Event()
        self._async_future: asyncio.Future = asyncio.get_event_loop().create_future() if False else None  # type: ignore

    # ------------------------------------------------------------------
    def set_result(self, result: InferenceResult) -> None:
        """Set the result and mark the future as done."""
        if self._done:
            raise RuntimeError("Future already resolved.")
        self._result = result
        self._done = True
        self._event.set()
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass

    def set_exception(self, error: str) -> None:
        """Set an error and mark the future as failed."""
        if self._done:
            raise RuntimeError("Future already resolved.")
        self._result = InferenceResult(
            request_id=self.request_id,
            status=RequestStatus.FAILED,
            error=error,
        )
        self._done = True
        self._event.set()
        for callback in self._callbacks:
            try:
                callback(self._result)  # type: ignore[arg-type]
            except Exception:
                pass

    def cancel(self) -> bool:
        """Attempt to cancel the future.

        Returns:
            ``True`` if cancelled, ``False`` if already done.
        """
        if self._done:
            return False
        self._cancelled = True
        self._done = True
        self._result = InferenceResult(
            request_id=self.request_id,
            status=RequestStatus.CANCELLED,
        )
        self._event.set()
        return True

    # ------------------------------------------------------------------
    def done(self) -> bool:
        """Return ``True`` if the future has resolved."""
        return self._done

    def cancelled(self) -> bool:
        """Return ``True`` if the future was cancelled."""
        return self._cancelled

    def result(self, timeout: Optional[float] = None) -> InferenceResult:
        """Block until the result is available.

        Args:
            timeout: Maximum seconds to wait.  ``None`` waits forever.

        Returns:
            The :class:`InferenceResult`.

        Raises:
            TimeoutError: If ``timeout`` is exceeded.
        """
        if not self._event.wait(timeout):
            raise TimeoutError(f"Future {self.request_id} timed out after {timeout}s.")
        assert self._result is not None
        return self._result

    def add_done_callback(
        self, callback: Callable[[InferenceResult], None]
    ) -> None:
        """Register a callback to be invoked when the future resolves."""
        if self._done:
            assert self._result is not None
            callback(self._result)
        else:
            self._callbacks.append(callback)

    def __await__(self):
        """Support ``await future`` in async contexts."""
        if self._done:
            assert self._result is not None
            return self._result
        # Fall back to blocking wait in a thread.
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, self.result).__await__()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_id() -> str:
    """Generate a unique request id."""
    return f"req_{next(_id_counter)}"


_id_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# InferenceScheduler
# ---------------------------------------------------------------------------
class InferenceScheduler:
    """Unified inference scheduling engine.

    Manages a priority queue of inference requests, automatically batches
    requests of the same modality, and supports continuous batching and
    streaming output.

    Args:
        max_batch_size: Maximum number of requests per batch.
        max_waiting_time: Seconds to wait before flushing a partial batch.
        enable_continuous_batching: Whether to use continuous batching.
        enable_streaming: Whether to support streaming output.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_waiting_time: float = 0.1,
        enable_continuous_batching: bool = True,
        enable_streaming: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}.")
        if max_waiting_time <= 0:
            raise ValueError(f"max_waiting_time must be > 0, got {max_waiting_time}.")

        self.max_batch_size: int = max_batch_size
        self.max_waiting_time: float = max_waiting_time
        self.enable_continuous_batching: bool = enable_continuous_batching
        self.enable_streaming: bool = enable_streaming

        self._logger = get_logger(self.__class__.__name__)

        # Priority queue (min-heap by negative priority).
        self._queue: List[Tuple[int, float, InferenceRequest]] = []
        self._queue_lock: threading.Lock = threading.Lock()
        self._counter: itertools.count = itertools.count()

        # Active futures.
        self._futures: Dict[str, Future] = {}

        # Running batch state.
        self._running: List[InferenceRequest] = []

        # Registered model handlers: modality -> callable.
        self._handlers: Dict[str, Callable[..., Any]] = {}

        # Background worker thread.
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._running_flag: bool = False

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------
    def register_handler(
        self,
        modality: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register an inference handler for a modality.

        Args:
            modality: The modality name (e.g. ``"text"``).
            handler: A callable with signature
                ``handler(batch: List[InferenceRequest]) -> List[Any]``.
        """
        self._handlers[modality.lower().strip()] = handler
        self._logger.debug("Registered handler for modality '%s'.", modality)

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------
    def submit(self, request: InferenceRequest) -> Future:
        """Submit an inference request.

        Args:
            request: The :class:`InferenceRequest` to submit.

        Returns:
            A :class:`Future` that resolves with the result.
        """
        future = Future(request.request_id)
        self._futures[request.request_id] = future

        request.status = RequestStatus.QUEUED
        with self._queue_lock:
            # Use negative priority so higher priority comes first.
            heapq.heappush(
                self._queue,
                (-request.priority, next(self._counter), request),
            )

        self._logger.debug(
            "Queued request %s (modality=%s, priority=%d).",
            request.request_id,
            request.modality,
            request.priority,
        )
        return future

    def submit_simple(
        self,
        modality: str,
        prompt: Any,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        stream: bool = False,
    ) -> Future:
        """Convenience method to submit a request without building it manually.

        Args:
            modality: The modality.
            prompt: The input prompt.
            params: Generation parameters.
            priority: Priority level.
            stream: Whether to stream.

        Returns:
            A :class:`Future`.
        """
        request = InferenceRequest(
            modality=modality,
            prompt=prompt,
            params=params or {},
            priority=priority,
            stream=stream,
        )
        return self.submit(request)

    # ------------------------------------------------------------------
    # Batch formation
    # ------------------------------------------------------------------
    def batch_requests(self) -> List[InferenceRequest]:
        """Form a batch from queued requests.

        Requests of the same modality are grouped together.  When
        continuous batching is enabled, currently running requests
        that are not yet finished are included in the new batch.

        Returns:
            A list of :class:`InferenceRequest` forming the batch.
            Returns an empty list when no requests are available.
        """
        with self._queue_lock:
            if not self._queue:
                return []

            # Peek at the highest-priority request to determine the modality.
            _, _, first_request = self._queue[0]
            target_modality = first_request.modality

            batch: List[InferenceRequest] = []
            remaining: List[Tuple[int, float, InferenceRequest]] = []

            # Collect requests of the same modality up to max_batch_size.
            while self._queue and len(batch) < self.max_batch_size:
                _, _, request = heapq.heappop(self._queue)
                if request.modality == target_modality:
                    request.status = RequestStatus.RUNNING
                    batch.append(request)
                else:
                    remaining.append((-request.priority, next(self._counter), request))

            # Push back non-matching requests.
            for item in remaining:
                heapq.heappush(self._queue, item)

            # Include running requests for continuous batching.
            if self.enable_continuous_batching:
                running_same = [
                    r for r in self._running if r.modality == target_modality
                ]
                batch = running_same + batch

            return batch

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_batch(self, batch: List[InferenceRequest]) -> None:
        """Execute a batch of requests.

        Args:
            batch: The batch to execute.
        """
        if not batch:
            return

        modality = batch[0].modality
        handler = self._handlers.get(modality)

        if handler is None:
            error_msg = f"No handler registered for modality '{modality}'."
            self._logger.error(error_msg)
            for req in batch:
                future = self._futures.get(req.request_id)
                if future is not None:
                    future.set_exception(error_msg)
            return

        # Update running state.
        self._running = batch

        try:
            results = handler(batch)
            if not isinstance(results, (list, tuple)):
                results = [results] * len(batch)

            for req, result in zip(batch, results):
                req.status = RequestStatus.COMPLETED
                future = self._futures.get(req.request_id)
                if future is not None and not future.done():
                    future.set_result(
                        InferenceResult(
                            request_id=req.request_id,
                            output=result,
                            status=RequestStatus.COMPLETED,
                            metadata={"modality": req.modality},
                        )
                    )
        except Exception as exc:
            self._logger.exception("Batch execution failed: %s", exc)
            for req in batch:
                req.status = RequestStatus.FAILED
                future = self._futures.get(req.request_id)
                if future is not None and not future.done():
                    future.set_exception(str(exc))
        finally:
            # Remove completed requests from running state.
            self._running = [r for r in self._running if r.status == RequestStatus.RUNNING]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def stream_request(
        self,
        request: InferenceRequest,
    ) -> Generator[Any, None, None]:
        """Stream the output of a request token-by-token / frame-by-frame.

        This is a synchronous generator.  For async streaming use
        :meth:`stream_request_async`.

        Args:
            request: The request to stream.

        Yields:
            Output chunks (tokens, frames, or segments).
        """
        if not self.enable_streaming:
            raise RuntimeError("Streaming is disabled.")

        handler = self._handlers.get(request.modality)
        if handler is None:
            raise ValueError(f"No handler for modality '{request.modality}'.")

        request.status = RequestStatus.STREAMING

        # Check if the handler supports streaming (is a generator).
        result = handler([request])
        if hasattr(result, "__iter__") and not isinstance(result, (list, tuple)):
            for chunk in result:
                yield chunk
        else:
            # Non-streaming handler: yield the full result.
            yield result

        request.status = RequestStatus.COMPLETED

    async def stream_request_async(
        self,
        request: InferenceRequest,
    ) -> AsyncGenerator[Any, None]:
        """Async version of :meth:`stream_request`.

        Args:
            request: The request to stream.

        Yields:
            Output chunks.
        """
        loop = asyncio.get_event_loop()
        gen = self.stream_request(request)
        try:
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, gen)
                    yield chunk
                except StopIteration:
                    break
        except StopIteration:
            pass

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background worker thread."""
        if self._running_flag:
            return
        self._running_flag = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="inference-scheduler"
        )
        self._worker_thread.start()
        self._logger.info("Inference scheduler worker started.")

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._stop_event.set()
        self._running_flag = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        self._logger.info("Inference scheduler worker stopped.")

    def _worker_loop(self) -> None:
        """Main worker loop: form batches and execute them."""
        while not self._stop_event.is_set():
            batch = self.batch_requests()
            if batch:
                self.execute_batch(batch)
            else:
                # Wait for new requests.
                time.sleep(self.max_waiting_time)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------
    async def submit_async(self, request: InferenceRequest) -> InferenceResult:
        """Submit a request and await its result asynchronously.

        Args:
            request: The request to submit.

        Returns:
            The :class:`InferenceResult`.
        """
        future = self.submit(request)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, future.result)

    async def run_async(self) -> None:
        """Run the scheduler in an async event loop."""
        loop = asyncio.get_event_loop()
        while True:
            batch = self.batch_requests()
            if batch:
                await loop.run_in_executor(None, self.execute_batch, batch)
            else:
                await asyncio.sleep(self.max_waiting_time)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def queue_size(self) -> int:
        """Return the number of queued requests."""
        with self._queue_lock:
            return len(self._queue)

    def running_count(self) -> int:
        """Return the number of currently running requests."""
        return len(self._running)

    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """Return the status of a request by id."""
        future = self._futures.get(request_id)
        if future is None:
            return None
        if future.done():
            result = future.result()
            return result.status
        return RequestStatus.RUNNING

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending or running request.

        Args:
            request_id: The id of the request to cancel.

        Returns:
            ``True`` if cancelled, ``False`` if not found or already done.
        """
        future = self._futures.get(request_id)
        if future is None or future.done():
            return False
        return future.cancel()

    def clear(self) -> None:
        """Clear all queued requests and cancel pending futures."""
        with self._queue_lock:
            self._queue.clear()
        for future in self._futures.values():
            if not future.done():
                future.cancel()
        self._futures.clear()
        self._running.clear()
