"""GPU memory pool and estimation layer for TorchaVerse v0.3.0.

This module replaces the coarse-grained :class:`~core.memory_manager.MemoryManager`
from v0.1.0 with a precise, thread-safe memory management stack:

* :class:`MemoryPool` -- a bookkeeping pool that tracks GPU memory
  allocations against a configurable capacity, supporting ``allocate``,
  ``release``, ``available``, ``used`` and ``peak`` queries.
* :class:`MemoryBlock` -- lightweight dataclass representing a single
  allocation handle.
* :class:`MemoryEstimator` -- precise memory estimation using real
  forward-pass profiling (when CUDA is available) and analytical
  formulas for KV-cache and activation sizes.
* :class:`MemoryEstimate` -- structured estimate result.
* :class:`OffloadStrategy` -- enum controlling how evicted models are
  stored (``NONE``, ``CPU``, ``DISK``).
* :class:`ModelOffloader` -- LRU-based model eviction manager that
  moves models between GPU, CPU pinned memory, and disk.

All classes are thread-safe (using :class:`threading.RLock`) and
device-aware.
"""

from __future__ import annotations

import enum
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "OffloadStrategy",
    "MemoryBlock",
    "MemoryPool",
    "MemoryEstimate",
    "MemoryEstimator",
    "ModelOffloader",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Bytes per gigabyte.
_BYTES_PER_GB: int = 1024 ** 3

#: Bytes per megabyte.
_BYTES_PER_MB: int = 1024 ** 2

#: Default fraction of total device memory the pool may use.
_DEFAULT_CAPACITY_FRACTION: float = 0.90

#: Default dtype used for size estimation (float32 = 4 bytes).
_DEFAULT_DTYPE: torch.dtype = torch.float32

#: Numerical-stability epsilon.
_EPSILON: float = 1e-8

#: Default offload strategy for the :class:`ModelOffloader`.
_DEFAULT_OFFLOAD_STRATEGY: "OffloadStrategy" = None  # set after enum def

#: Prefix for disk-offload temporary files.
_DISK_FILE_PREFIX: str = "torcha_offload_"

#: Default disk-offload directory (system temp).
_DEFAULT_DISK_DIR: str = tempfile.gettempdir()


# ---------------------------------------------------------------------------
# OffloadStrategy
# ---------------------------------------------------------------------------
class OffloadStrategy(enum.Enum):
    """Strategy for storing evicted models.

    Attributes:
        NONE: No offloading; evicted models are dereferenced and left
            for Python's garbage collector.  They cannot be reloaded.
        CPU: Evicted models are moved to CPU (pinned) memory and can be
            reloaded back to GPU.
        DISK: Evicted models are serialized to disk via
            :func:`torch.save` and removed from RAM.  Reloading reads
            them back with :func:`torch.load`.
    """

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


# Update the forward-reference default.
_DEFAULT_OFFLOAD_STRATEGY = OffloadStrategy.CPU


# ---------------------------------------------------------------------------
# MemoryBlock
# ---------------------------------------------------------------------------
@dataclass
class MemoryBlock:
    """A single allocation handle returned by :class:`MemoryPool`.

    Attributes:
        ptr: Unique integer handle identifying this allocation.
        size: Allocated size in bytes.
        tag: Optional human-readable tag for debugging.
        device: Device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    """

    ptr: int
    size: int
    tag: str = ""
    device: str = "cpu"

    def __repr__(self) -> str:
        return (
            "MemoryBlock(ptr={}, size={}B, tag='{}', device='{}')".format(
                self.ptr, self.size, self.tag, self.device
            )
        )


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------
class MemoryPool:
    """Thread-safe GPU memory bookkeeping pool.

    The pool tracks logical allocations against a configurable capacity
    (defaulting to a fraction of the device's total memory).  It does
    not allocate raw tensors; instead it provides an accounting layer
    that callers can use to plan and enforce memory budgets.

    Example:
        >>> pool = MemoryPool()
        >>> block = pool.allocate(1024, tag="kv_cache")
        >>> pool.used()
        1024
        >>> pool.release(block)
        >>> pool.used()
        0
    """

    def __init__(
        self,
        capacity_bytes: Optional[int] = None,
        device: Optional[torch.device] = None,
        capacity_fraction: float = _DEFAULT_CAPACITY_FRACTION,
    ) -> None:
        """Initialise the pool.

        Args:
            capacity_bytes: Maximum bytes the pool may track.  When
                ``None`` the capacity is derived from the device's total
                memory multiplied by *capacity_fraction*.
            device: Target device.  Defaults to the device reported by
                :class:`~infrastructure.device_manager.DeviceManager`.
            capacity_fraction: Fraction of total device memory to use
                when *capacity_bytes* is ``None``.
        """
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            device if device is not None else self._device_manager.get_device()
        )
        self._logger = get_logger(self.__class__.__name__)

        if capacity_bytes is not None:
            self._capacity: int = int(capacity_bytes)
        else:
            self._capacity = self._infer_capacity(capacity_fraction)

        self._used: int = 0
        self._peak: int = 0
        self._blocks: Dict[int, MemoryBlock] = {}
        self._ptr_counter: int = 0
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    def _infer_capacity(self, fraction: float) -> int:
        """Derive the pool capacity from the device's total memory."""
        if torch.cuda.is_available() and "cuda" in str(self._device):
            try:
                idx = self._device.index if self._device.index is not None else 0
                total = torch.cuda.get_device_properties(idx).total_memory
                return int(total * fraction)
            except (RuntimeError, IndexError):
                pass
        # CPU fallback: use a conservative default.
        return int(8 * _BYTES_PER_GB * fraction)

    # ------------------------------------------------------------------
    def allocate(self, size_bytes: int, tag: str = "") -> MemoryBlock:
        """Allocate *size_bytes* and return a :class:`MemoryBlock`.

        Args:
            size_bytes: Number of bytes to allocate (``> 0``).
            tag: Optional debug tag.

        Returns:
            A :class:`MemoryBlock` handle.

        Raises:
            ValueError: If *size_bytes* is non-positive.
            MemoryError: If the pool does not have enough free capacity.
        """
        if size_bytes <= 0:
            raise ValueError(
                "size_bytes must be > 0, got {}.".format(size_bytes)
            )

        with self._lock:
            if self._used + size_bytes > self._capacity:
                raise MemoryError(
                    "MemoryPool out of capacity: requested {}B, "
                    "used {}B, capacity {}B, available {}B.".format(
                        size_bytes, self._used, self._capacity,
                        self._capacity - self._used,
                    )
                )
            self._ptr_counter += 1
            block = MemoryBlock(
                ptr=self._ptr_counter,
                size=size_bytes,
                tag=tag,
                device=str(self._device),
            )
            self._blocks[block.ptr] = block
            self._used += size_bytes
            if self._used > self._peak:
                self._peak = self._used
            self._logger.debug(
                "Allocated %dB (ptr=%d, tag='%s'); used=%dB.",
                size_bytes, block.ptr, tag, self._used,
            )
            return block

    # ------------------------------------------------------------------
    def release(self, block: MemoryBlock) -> None:
        """Release a previously allocated block.

        Args:
            block: The :class:`MemoryBlock` to release.

        Raises:
            KeyError: If *block* was already released or never
                allocated by this pool.
        """
        with self._lock:
            if block.ptr not in self._blocks:
                raise KeyError(
                    "Block ptr={} not found (already released?).".format(
                        block.ptr
                    )
                )
            del self._blocks[block.ptr]
            self._used -= block.size
            if self._used < 0:
                self._used = 0
            self._logger.debug(
                "Released %dB (ptr=%d); used=%dB.",
                block.size, block.ptr, self._used,
            )

    # ------------------------------------------------------------------
    def available(self) -> int:
        """Return the number of free bytes in the pool."""
        with self._lock:
            return max(0, self._capacity - self._used)

    # ------------------------------------------------------------------
    def used(self) -> int:
        """Return the number of currently allocated bytes."""
        with self._lock:
            return self._used

    # ------------------------------------------------------------------
    def peak(self) -> int:
        """Return the peak allocation in bytes."""
        with self._lock:
            return self._peak

    # ------------------------------------------------------------------
    def capacity(self) -> int:
        """Return the total capacity in bytes."""
        with self._lock:
            return self._capacity

    # ------------------------------------------------------------------
    def reset_peak(self) -> None:
        """Reset the peak counter to the current usage."""
        with self._lock:
            self._peak = self._used

    # ------------------------------------------------------------------
    def blocks(self) -> List[MemoryBlock]:
        """Return a snapshot of all active blocks."""
        with self._lock:
            return list(self._blocks.values())

    def __repr__(self) -> str:
        return (
            "MemoryPool(device='{}', used={}B, capacity={}B, peak={}B)".format(
                self._device, self._used, self._capacity, self._peak
            )
        )


# ---------------------------------------------------------------------------
# MemoryEstimate
# ---------------------------------------------------------------------------
@dataclass
class MemoryEstimate:
    """Structured memory estimate for a model + activations + KV cache.

    Attributes:
        weights_gb: Estimated weight memory in gigabytes.
        activations_gb: Estimated activation memory in gigabytes.
        kv_cache_gb: Estimated KV-cache memory in gigabytes.
        total_gb: Total estimated memory in gigabytes.
        feasible: Whether the total fits within the available device
            memory.
    """

    weights_gb: float
    activations_gb: float
    kv_cache_gb: float
    total_gb: float
    feasible: bool

    def __repr__(self) -> str:
        return (
            "MemoryEstimate(weights={:.3f}GB, activations={:.3f}GB, "
            "kv_cache={:.3f}GB, total={:.3f}GB, feasible={})".format(
                self.weights_gb, self.activations_gb,
                self.kv_cache_gb, self.total_gb, self.feasible,
            )
        )


# ---------------------------------------------------------------------------
# MemoryEstimator
# ---------------------------------------------------------------------------
class MemoryEstimator:
    """Precise GPU memory estimator.

    Provides analytical and profiling-based estimates for model weights,
    activations, and KV cache.  When CUDA is available, the estimator
    uses real forward-pass profiling via
    :func:`torch.cuda.max_memory_allocated`; otherwise it falls back to
    parameter-count heuristics.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = _DEFAULT_DTYPE,
    ) -> None:
        """Initialise the estimator.

        Args:
            device: Target device for estimation.
            dtype: Assumed element dtype for size calculations.
        """
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            device if device is not None else self._device_manager.get_device()
        )
        self._dtype: torch.dtype = dtype
        self._bytes_per_element: int = self._dtype_bytes(dtype)
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    @staticmethod
    def _dtype_bytes(dtype: torch.dtype) -> int:
        """Return the number of bytes per element for *dtype*."""
        _table: Dict[torch.dtype, int] = {
            torch.float32: 4,
            torch.float64: 8,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.uint8: 1,
            torch.bool: 1,
        }
        return _table.get(dtype, 4)

    # ------------------------------------------------------------------
    def estimate_model(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> int:
        """Estimate the peak activation memory of *model* via a forward pass.

        When CUDA is available and *input_shape* is provided, a real
        forward pass is executed under :func:`torch.no_grad` and the
        peak GPU memory is measured with
        :func:`torch.cuda.max_memory_allocated`.  Otherwise the weight
        memory is returned as a lower-bound estimate.

        Args:
            model: The model to profile.
            input_shape: Shape of a dummy input tensor.  If ``None``,
                only weight memory is estimated.

        Returns:
            Estimated peak memory in bytes.
        """
        # Weight memory (always available).
        weight_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )

        if input_shape is None:
            self._logger.debug(
                "No input_shape provided; returning weight estimate (%dB).",
                weight_bytes,
            )
            return weight_bytes

        if not torch.cuda.is_available() or "cuda" not in str(self._device):
            # CPU fallback: weight + rough activation estimate.
            activation_bytes = self._estimate_activations_cpu(
                model, input_shape
            )
            return weight_bytes + activation_bytes

        # CUDA profiling path.
        device = self._device
        try:
            model = model.to(device)
            model.eval()
            # Reset peak memory.
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            with torch.no_grad():
                dummy = torch.zeros(
                    input_shape, device=device, dtype=self._dtype
                )
                _ = model(dummy)

            torch.cuda.synchronize(device)
            peak_bytes = torch.cuda.max_memory_allocated(device)
            # max_memory_allocated includes weights; subtract to get
            # activation-only, then add back weights for the total.
            activation_bytes = max(0, peak_bytes - weight_bytes)
            total = weight_bytes + activation_bytes
            self._logger.debug(
                "Profiled model: weights=%dB, activations=%dB, total=%dB.",
                weight_bytes, activation_bytes, total,
            )
            return total
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Forward-pass profiling failed (%s); falling back to "
                "weight estimate.", exc,
            )
            return weight_bytes

    # ------------------------------------------------------------------
    def estimate_kv_cache(
        self,
        num_tokens: int,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> int:
        """Estimate the KV-cache memory for a transformer model.

        The formula is::

            bytes = 2 * num_layers * num_tokens * num_kv_heads * head_dim * dtype_bytes

        where ``head_dim = hidden_size // num_heads`` and the factor of
        2 accounts for both keys and values.

        Args:
            num_tokens: Sequence length.
            num_layers: Number of transformer layers.
            hidden_size: Model hidden dimension.
            num_heads: Number of attention heads.
            num_kv_heads: Number of key/value heads (for GQA/MQA).

        Returns:
            Estimated KV-cache memory in bytes.
        """
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        head_dim: int = hidden_size // num_heads
        kv_elements = (
            2  # K and V
            * num_layers
            * num_tokens
            * num_kv_heads
            * head_dim
        )
        return kv_elements * self._bytes_per_element

    # ------------------------------------------------------------------
    def estimate_activations(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> int:
        """Estimate activation memory via a fake forward pass.

        When CUDA is available, this runs a forward pass and measures
        the peak activation memory (excluding weights).  On CPU it uses
        a heuristic based on the input and output tensor sizes.

        Args:
            model: The model to profile.
            input_shape: Shape of the dummy input.

        Returns:
            Estimated activation memory in bytes.
        """
        if torch.cuda.is_available() and "cuda" in str(self._device):
            device = self._device
            weight_bytes = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            )
            try:
                model = model.to(device)
                model.eval()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                with torch.no_grad():
                    dummy = torch.zeros(
                        input_shape, device=device, dtype=self._dtype
                    )
                    _ = model(dummy)
                torch.cuda.synchronize(device)
                peak_bytes = torch.cuda.max_memory_allocated(device)
                return max(0, peak_bytes - weight_bytes)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Activation profiling failed (%s); using CPU heuristic.",
                    exc,
                )
        return self._estimate_activations_cpu(model, input_shape)

    # ------------------------------------------------------------------
    def _estimate_activations_cpu(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> int:
        """Heuristic activation estimate for CPU-only environments.

        Uses a simple multiplier on the input tensor size plus the
        number of parameters as a proxy for intermediate buffer sizes.
        """
        input_elements = 1
        for dim in input_shape:
            input_elements *= max(1, dim)
        input_bytes = input_elements * self._bytes_per_element
        # Heuristic: activations are roughly 3-5x the input size for
        # typical transformer models.
        _ACTIVATION_MULTIPLIER: int = 4
        return input_bytes * _ACTIVATION_MULTIPLIER

    # ------------------------------------------------------------------
    def estimate_total(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        kv_cache_size: int = 0,
    ) -> MemoryEstimate:
        """Compute a full :class:`MemoryEstimate`.

        Combines weight, activation, and KV-cache estimates and
        determines feasibility against available device memory.

        Args:
            model: The model to estimate.
            input_shape: Shape of the dummy input for activation
                profiling.
            kv_cache_size: Pre-computed KV-cache size in bytes (e.g.
                from :meth:`estimate_kv_cache`).  ``0`` means no KV
                cache.

        Returns:
            A :class:`MemoryEstimate` with all fields populated.
        """
        weight_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        activation_bytes = self.estimate_activations(model, input_shape)
        kv_bytes = int(kv_cache_size)

        total_bytes = weight_bytes + activation_bytes + kv_bytes

        weights_gb = weight_bytes / _BYTES_PER_GB
        activations_gb = activation_bytes / _BYTES_PER_GB
        kv_cache_gb = kv_bytes / _BYTES_PER_GB
        total_gb = total_bytes / _BYTES_PER_GB

        # Determine feasibility.
        available_bytes = self._available_device_memory()
        feasible = total_bytes <= available_bytes

        return MemoryEstimate(
            weights_gb=weights_gb,
            activations_gb=activations_gb,
            kv_cache_gb=kv_cache_gb,
            total_gb=total_gb,
            feasible=feasible,
        )

    # ------------------------------------------------------------------
    def _available_device_memory(self) -> int:
        """Return the available (free) memory on the target device."""
        if torch.cuda.is_available() and "cuda" in str(self._device):
            try:
                idx = self._device.index if self._device.index is not None else 0
                props = torch.cuda.get_device_properties(idx)
                allocated = torch.cuda.memory_allocated(idx)
                return max(0, props.total_memory - allocated)
            except (RuntimeError, IndexError):
                pass
        # CPU fallback.
        return int(8 * _BYTES_PER_GB)


# ---------------------------------------------------------------------------
# Internal dataclasses for ModelOffloader
# ---------------------------------------------------------------------------
@dataclass
class _LoadedEntry:
    """Internal entry for a model currently resident on GPU."""

    model: nn.Module
    size_bytes: int
    loaded_at: float


@dataclass
class _OffloadedEntry:
    """Internal entry for a model that has been evicted."""

    model: Optional[nn.Module]
    size_bytes: int
    strategy: OffloadStrategy
    disk_path: Optional[str] = None


# ---------------------------------------------------------------------------
# ModelOffloader
# ---------------------------------------------------------------------------
class ModelOffloader:
    """LRU-based model offloading manager.

    Manages a set of models resident on GPU, evicting the least
    recently used models to CPU or disk when more GPU memory is needed.

    Example:
        >>> offloader = ModelOffloader(strategy=OffloadStrategy.CPU)
        >>> offloader.load("unet", unet_model)
        >>> model = offloader.get("unet")
        >>> offloader.evict_if_needed(2.0)  # free 2 GB
    """

    def __init__(
        self,
        strategy: OffloadStrategy = _DEFAULT_OFFLOAD_STRATEGY,
        device: Optional[torch.device] = None,
        disk_dir: str = _DEFAULT_DISK_DIR,
        pool: Optional[MemoryPool] = None,
    ) -> None:
        """Initialise the offloader.

        Args:
            strategy: How to store evicted models.
            device: Target GPU device.
            disk_dir: Directory for disk-offloaded model files.
            pool: Optional :class:`MemoryPool` for tracking allocations.
                If ``None``, a new pool is created.
        """
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            device if device is not None else self._device_manager.get_device()
        )
        self._strategy: OffloadStrategy = strategy
        self._disk_dir: str = disk_dir
        self._pool: MemoryPool = pool if pool is not None else MemoryPool(
            device=self._device
        )
        self._estimator: MemoryEstimator = MemoryEstimator(device=self._device)
        self._logger = get_logger(self.__class__.__name__)

        # LRU-ordered dict of loaded models (front = most recent).
        self._loaded: "OrderedDict[str, _LoadedEntry]" = OrderedDict()
        # Offloaded models awaiting reload.
        self._offloaded: Dict[str, _OffloadedEntry] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    def _model_size_bytes(self, model: nn.Module) -> int:
        """Compute the parameter memory of *model* in bytes."""
        return sum(p.nelement() * p.element_size() for p in model.parameters())

    # ------------------------------------------------------------------
    def load(self, model_id: str, model: nn.Module) -> None:
        """Load *model* onto the GPU under *model_id*.

        If the model was previously offloaded, it is reloaded from its
        offload location.  Otherwise the model is moved to the target
        device and registered.

        Args:
            model_id: Unique identifier for the model.
            model: The model to load (used when loading for the first
                time; ignored when reloading from offload).

        Raises:
            MemoryError: If there is not enough GPU memory even after
                eviction.
        """
        with self._lock:
            # Already loaded?  Just touch LRU.
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id, last=False)
                self._logger.debug("Model '%s' already loaded; touched LRU.", model_id)
                return

            # Reload from offload?
            if model_id in self._offloaded:
                entry = self._offloaded.pop(model_id)
                self._reload_from_offload(model_id, entry)
                return

            # Fresh load.
            size_bytes = self._model_size_bytes(model)
            self._evict_if_needed_bytes(size_bytes)

            model = model.to(self._device)
            self._loaded[model_id] = _LoadedEntry(
                model=model, size_bytes=size_bytes, loaded_at=time.time(),
            )
            self._loaded.move_to_end(model_id, last=False)
            self._logger.info(
                "Loaded model '%s' (%dB) onto %s.",
                model_id, size_bytes, self._device,
            )

    # ------------------------------------------------------------------
    def _reload_from_offload(
        self, model_id: str, entry: _OffloadedEntry
    ) -> None:
        """Reload a model from its offload location."""
        size_bytes = entry.size_bytes
        self._evict_if_needed_bytes(size_bytes)

        if entry.strategy == OffloadStrategy.CPU:
            assert entry.model is not None
            model = entry.model.to(self._device)
        elif entry.strategy == OffloadStrategy.DISK:
            assert entry.disk_path is not None
            model = torch.load(entry.disk_path, map_location=self._device)
            try:
                os.remove(entry.disk_path)
            except OSError:
                self._logger.warning(
                    "Failed to remove disk file %s.", entry.disk_path
                )
        else:
            # NONE strategy: model was dereferenced; cannot reload.
            raise RuntimeError(
                "Cannot reload model '{}': it was evicted with "
                "OffloadStrategy.NONE.".format(model_id)
            )

        self._loaded[model_id] = _LoadedEntry(
            model=model, size_bytes=size_bytes, loaded_at=time.time(),
        )
        self._loaded.move_to_end(model_id, last=False)
        self._logger.info(
            "Reloaded model '%s' from %s.", model_id, entry.strategy.value
        )

    # ------------------------------------------------------------------
    def unload(self, model_id: str) -> None:
        """Unload *model_id* from GPU to its offload location.

        Args:
            model_id: The model to unload.

        Raises:
            KeyError: If *model_id* is not currently loaded.
        """
        with self._lock:
            if model_id not in self._loaded:
                raise KeyError(
                    "Model '{}' is not loaded.".format(model_id)
                )
            entry = self._loaded.pop(model_id)
            self._offload_model(model_id, entry)

    # ------------------------------------------------------------------
    def _offload_model(
        self, model_id: str, entry: _LoadedEntry
    ) -> None:
        """Move a loaded model to its offload location."""
        model = entry.model
        size_bytes = entry.size_bytes

        if self._strategy == OffloadStrategy.CPU:
            model = model.to("cpu")
            # ``del model`` here only dropped the *local* binding; the
            # previous implementation never assigned the CPU copy back
            # to ``self._offloaded`` and as a result the original GPU
            # model was kept alive by the offloaded entry's attribute
            # (or by the caller holding a reference), which meant GPU
            # memory was *never* released.  We now store the CPU copy
            # in the entry and explicitly clear the CUDA cache.
            self._offloaded[model_id] = _OffloadedEntry(
                model=model, size_bytes=size_bytes, strategy=OffloadStrategy.CPU,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self._strategy == OffloadStrategy.DISK:
            disk_path = os.path.join(
                self._disk_dir,
                "{}{}.pt".format(_DISK_FILE_PREFIX, model_id),
            )
            torch.save(model, disk_path)
            # Move the (now-saved) model to CPU and drop the GPU tensor
            # explicitly.  Without ``empty_cache()`` the CUDA caching
            # allocator keeps the freed blocks around for reuse which
            # is the right thing for steady state, but a top-level
            # offload operation should expose the released memory so
            # that the next allocation does not have to wait for
            # fragmentation.
            model = model.to("cpu")
            self._offloaded[model_id] = _OffloadedEntry(
                model=None, size_bytes=size_bytes,
                strategy=OffloadStrategy.DISK, disk_path=disk_path,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # NONE: just dereference.
            self._offloaded[model_id] = _OffloadedEntry(
                model=None, size_bytes=size_bytes, strategy=OffloadStrategy.NONE,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._logger.info(
            "Offloaded model '%s' (%dB) via %s.",
            model_id, size_bytes, self._strategy.value,
        )

    # ------------------------------------------------------------------
    def get(self, model_id: str) -> nn.Module:
        """Retrieve a loaded model, updating its LRU position.

        Args:
            model_id: The model to retrieve.

        Returns:
            The model on the GPU device.

        Raises:
            KeyError: If *model_id* is not currently loaded (it may
                have been offloaded; call :meth:`load` to reload it).
        """
        with self._lock:
            if model_id not in self._loaded:
                raise KeyError(
                    "Model '{}' is not loaded. Use load() to reload it "
                    "from offload.".format(model_id)
                )
            self._loaded.move_to_end(model_id, last=False)
            return self._loaded[model_id].model

    # ------------------------------------------------------------------
    def evict_if_needed(self, needed_gb: float) -> int:
        """Evict LRU models until at least *needed_gb* is freed.

        Args:
            needed_gb: The amount of GPU memory to free, in gigabytes.

        Returns:
            The number of models evicted.
        """
        needed_bytes = int(needed_gb * _BYTES_PER_GB)
        return self._evict_if_needed_bytes(needed_bytes)

    # ------------------------------------------------------------------
    def _evict_if_needed_bytes(self, needed_bytes: int) -> int:
        """Evict LRU models until *needed_bytes* are free.

        This method does NOT acquire the lock; callers must hold
        ``self._lock``.

        Returns:
            The number of models evicted.
        """
        evicted = 0
        freed = 0
        # Evict from the back of the OrderedDict (least recently used).
        while freed < needed_bytes and self._loaded:
            # Get the LRU model (last item).
            model_id, entry = self._loaded.popitem(last=True)
            self._offload_model(model_id, entry)
            freed += entry.size_bytes
            evicted += 1

        if evicted > 0:
            self._logger.info(
                "Evicted %d model(s), freed %.3f GB.",
                evicted, freed / _BYTES_PER_GB,
            )
        return evicted

    # ------------------------------------------------------------------
    def loaded_models(self) -> List[str]:
        """Return a list of currently loaded model IDs (LRU order)."""
        with self._lock:
            return list(self._loaded.keys())

    # ------------------------------------------------------------------
    def offloaded_models(self) -> List[str]:
        """Return a list of currently offloaded model IDs."""
        with self._lock:
            return list(self._offloaded.keys())

    def __repr__(self) -> str:
        return (
            "ModelOffloader(strategy={}, device='{}', "
            "loaded={}, offloaded={})".format(
                self._strategy.value, self._device,
                len(self._loaded), len(self._offloaded),
            )
        )
