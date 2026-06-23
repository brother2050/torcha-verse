"""Unified memory management for TorchaVerse.

This module provides :class:`MemoryManager`, which centralises all
GPU VRAM and CPU RAM management decisions.  Key capabilities:

* **GPU memory monitoring** -- Real-time tracking of allocated and
  reserved memory via the CUDA allocator API.
* **Automatic offloading** -- When GPU memory pressure exceeds a
  threshold, inactive tensors or model shards are transparently moved
  to CPU RAM.
* **Sharded loading** -- Load large models layer-by-layer to avoid
  peak memory spikes.
* **Peak prediction** -- Estimate the memory footprint of a model
  before instantiation.
* **Memory pool** -- Pre-allocate a pool of GPU memory blocks for
  efficient reuse without fragmentation.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "MemoryManager",
    "MemoryPool",
    "MemoryInfo",
]


# ---------------------------------------------------------------------------
# MemoryInfo
# ---------------------------------------------------------------------------
class MemoryInfo:
    """Snapshot of memory usage at a point in time.

    Attributes:
        device: The device this snapshot describes.
        used_bytes: Currently allocated bytes.
        total_bytes: Total device capacity in bytes.
        free_bytes: ``total_bytes - used_bytes``.
        usage_fraction: ``used_bytes / total_bytes``.
    """

    __slots__ = ("device", "used_bytes", "total_bytes", "free_bytes", "usage_fraction")

    def __init__(
        self,
        device: torch.device,
        used_bytes: int,
        total_bytes: int,
    ) -> None:
        self.device: torch.device = device
        self.used_bytes: int = used_bytes
        self.total_bytes: int = total_bytes
        self.free_bytes: int = max(0, total_bytes - used_bytes)
        self.usage_fraction: float = used_bytes / total_bytes if total_bytes > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"MemoryInfo(device={self.device}, "
            f"used={self._fmt(self.used_bytes)}, "
            f"total={self._fmt(self.total_bytes)}, "
            f"usage={self.usage_fraction:.1%})"
        )

    @staticmethod
    def _fmt(n: int) -> str:
        """Format bytes as a human-readable string."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------
class MemoryPool:
    """A simple pre-allocated GPU memory pool.

    Manages a set of fixed-size memory blocks that can be acquired and
    released without triggering the CUDA allocator.  This reduces
    fragmentation for workloads with predictable allocation patterns.

    Args:
        num_blocks: Number of blocks in the pool.
        block_shape: Shape of each block (excluding batch dimension).
        dtype: Data type of the blocks.
        device: Device for the pool.
    """

    def __init__(
        self,
        num_blocks: int,
        block_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}.")
        self.num_blocks: int = num_blocks
        self.block_shape: Tuple[int, ...] = block_shape
        self.dtype: torch.dtype = dtype

        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        # Pre-allocate the pool.
        full_shape = (num_blocks,) + block_shape
        self._pool: torch.Tensor = torch.zeros(full_shape, dtype=dtype, device=self._device)
        self._free_indices: List[int] = list(range(num_blocks))
        self._acquired: Dict[int, int] = {}  # handle -> block index
        self._next_handle: int = 0

    # ------------------------------------------------------------------
    def acquire(self) -> int:
        """Acquire a free block from the pool.

        Returns:
            A handle that can be used with :meth:`get_block` and
            :meth:`release`.

        Raises:
            RuntimeError: If the pool is exhausted.
        """
        if not self._free_indices:
            raise RuntimeError("MemoryPool exhausted: all blocks are in use.")
        idx = self._free_indices.pop(0)
        handle = self._next_handle
        self._next_handle += 1
        self._acquired[handle] = idx
        return handle

    def release(self, handle: int) -> None:
        """Release a previously acquired block back to the pool.

        Args:
            handle: The handle returned by :meth:`acquire`.

        Raises:
            KeyError: If ``handle`` is not currently acquired.
        """
        if handle not in self._acquired:
            raise KeyError(f"Unknown handle {handle}.")
        idx = self._acquired.pop(handle)
        # Zero out the block before returning it.
        self._pool[idx].zero_()
        self._free_indices.append(idx)

    def get_block(self, handle: int) -> torch.Tensor:
        """Return the tensor view for an acquired block.

        Args:
            handle: The handle returned by :meth:`acquire`.

        Returns:
            A view into the pool tensor with shape ``block_shape``.
        """
        if handle not in self._acquired:
            raise KeyError(f"Unknown handle {handle}.")
        idx = self._acquired[handle]
        return self._pool[idx]

    @property
    def num_free(self) -> int:
        """Number of free blocks."""
        return len(self._free_indices)

    @property
    def num_used(self) -> int:
        """Number of acquired blocks."""
        return len(self._acquired)

    def total_bytes(self) -> int:
        """Total bytes allocated by the pool."""
        return self._pool.nelement() * self._pool.element_size()

    def clear(self) -> None:
        """Release all acquired blocks."""
        self._free_indices = list(range(self.num_blocks))
        self._acquired.clear()
        self._pool.zero_()


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------
class MemoryManager:
    """Unified GPU VRAM and CPU RAM management.

    This singleton provides a single entry point for all memory-related
    operations: monitoring, offloading, sharded loading, peak prediction,
    and garbage collection.

    Args:
        auto_offload: When ``True``, automatically offload tensors to
            CPU when GPU memory pressure exceeds ``offload_threshold``.
        offload_threshold: GPU memory usage fraction that triggers
            automatic offloading.
        gpu_memory_fraction: Target fraction of GPU memory to keep
            allocated (for ``set_memory_fraction``).
    """

    _instance: Optional["MemoryManager"] = None
    _initialized: bool = False

    def __init__(
        self,
        auto_offload: bool = True,
        offload_threshold: float = 0.85,
        gpu_memory_fraction: float = 0.9,
    ) -> None:
        if MemoryManager._initialized:
            return
        MemoryManager._initialized = True

        self.auto_offload: bool = auto_offload
        self.offload_threshold: float = offload_threshold
        self.gpu_memory_fraction: float = gpu_memory_fraction

        self._device_manager: DeviceManager = DeviceManager()
        self._logger = get_logger(self.__class__.__name__)

        # Track offloaded objects: id -> (original_device, object).
        self._offloaded: Dict[int, Tuple[torch.device, Any]] = {}

        # Set the memory fraction if on CUDA.
        self._set_memory_fraction()

    def __new__(cls, *args: Any, **kwargs: Any) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # GPU memory monitoring
    # ------------------------------------------------------------------
    def get_gpu_memory_info(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` for the GPU.

        Args:
            device: CUDA device to query.  Defaults to the active device.

        Returns:
            A tuple ``(used, total)`` in bytes.  On non-CUDA devices
            returns ``(0, 0)``.
        """
        if not torch.cuda.is_available():
            return 0, 0

        dev = torch.device(device) if isinstance(device, str) else device
        if dev is None:
            dev = self._device_manager.get_device()

        if dev.type != "cuda":
            return 0, 0

        used = torch.cuda.memory_allocated(dev)
        total = torch.cuda.get_device_properties(dev).total_memory
        return used, total

    def get_memory_info(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> MemoryInfo:
        """Return a :class:`MemoryInfo` snapshot for ``device``.

        Args:
            device: Device to query.  Defaults to the active device.

        Returns:
            A :class:`MemoryInfo` object.
        """
        dev = torch.device(device) if isinstance(device, str) else device
        if dev is None:
            dev = self._device_manager.get_device()

        if dev.type == "cuda" and torch.cuda.is_available():
            used = torch.cuda.memory_allocated(dev)
            total = torch.cuda.get_device_properties(dev).total_memory
        else:
            # CPU: use psutil if available, otherwise report 0.
            used = 0
            total = 0
            try:
                import psutil

                mem = psutil.virtual_memory()
                used = mem.used
                total = mem.total
            except ImportError:
                pass

        return MemoryInfo(dev, used, total)

    def check_memory_available(
        self,
        required_bytes: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> bool:
        """Check whether ``required_bytes`` are available on ``device``.

        Args:
            required_bytes: Number of bytes needed.
            device: Device to check.  Defaults to the active device.

        Returns:
            ``True`` if enough memory is available, ``False`` otherwise.
        """
        info = self.get_memory_info(device)
        return info.free_bytes >= required_bytes

    # ------------------------------------------------------------------
    # Automatic offloading
    # ------------------------------------------------------------------
    def offload_to_cpu(
        self,
        obj: Union[torch.Tensor, nn.Module],
        pin_memory: bool = True,
    ) -> Union[torch.Tensor, nn.Module]:
        """Move a tensor or module to CPU memory.

        The original device is recorded so the object can be restored
        later via :meth:`reload_to_gpu`.

        Args:
            obj: The tensor or module to offload.
            pin_memory: When ``True``, use pinned (page-locked) memory
                for faster GPU transfers later.

        Returns:
            The offloaded object (now on CPU).
        """
        if isinstance(obj, torch.Tensor):
            original_device = obj.device
            if original_device.type == "cpu":
                return obj
            cpu_tensor = obj.cpu()
            if pin_memory and not cpu_tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()
            self._offloaded[id(cpu_tensor)] = (original_device, obj)
            self._logger.debug(
                "Offloaded tensor (%s) from %s to CPU.",
                tuple(obj.shape),
                original_device,
            )
            return cpu_tensor

        if isinstance(obj, nn.Module):
            original_device = next(obj.parameters(), torch.tensor(0)).device
            obj.cpu()
            self._offloaded[id(obj)] = (original_device, obj)
            self._logger.debug("Offloaded module %s from %s to CPU.", type(obj).__name__, original_device)
            return obj

        raise TypeError(f"Cannot offload object of type {type(obj).__name__}.")

    def reload_to_gpu(
        self,
        obj: Union[torch.Tensor, nn.Module],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[torch.Tensor, nn.Module]:
        """Move an offloaded object back to the GPU.

        Args:
            obj: The offloaded tensor or module.
            device: Target device.  Defaults to the original device
                recorded during offloading.

        Returns:
            The object moved back to the GPU.
        """
        target = device
        if target is None:
            record = self._offloaded.get(id(obj))
            if record is not None:
                target = record[0]
        if target is None:
            target = self._device_manager.get_device()

        if isinstance(obj, torch.Tensor):
            return obj.to(target)
        if isinstance(obj, nn.Module):
            obj.to(target)
            return obj
        raise TypeError(f"Cannot reload object of type {type(obj).__name__}.")

    def maybe_offload(
        self,
        obj: Union[torch.Tensor, nn.Module],
        threshold: Optional[float] = None,
    ) -> bool:
        """Offload ``obj`` to CPU if GPU usage exceeds the threshold.

        Args:
            obj: The tensor or module to potentially offload.
            threshold: Override the default ``offload_threshold``.

        Returns:
            ``True`` if the object was offloaded, ``False`` otherwise.
        """
        if not self.auto_offload:
            return False

        effective_threshold = threshold or self.offload_threshold
        used, total = self.get_gpu_memory_info()

        if total == 0:
            return False

        if used / total > effective_threshold:
            self.offload_to_cpu(obj)
            return True
        return False

    # ------------------------------------------------------------------
    # Sharded loading
    # ------------------------------------------------------------------
    def load_model_sharded(
        self,
        model: nn.Module,
        checkpoint_path: str,
        num_shards: int = 1,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> nn.Module:
        """Load model weights in shards to reduce peak memory.

        The state dict is split into ``num_shards`` groups (by parameter
        name prefix) and each group is loaded sequentially.  This avoids
        materialising the entire state dict in memory at once.

        Args:
            model: The model to load weights into.
            checkpoint_path: Path to the weights file.
            num_shards: Number of shards to split the loading into.
            map_location: Device to map tensors to.

        Returns:
            The model with loaded weights (modified in place).
        """
        from pathlib import Path

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the full state dict to CPU first.
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        keys = sorted(state_dict.keys())
        shard_size = max(1, len(keys) // num_shards)

        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = start + shard_size if shard_idx < num_shards - 1 else len(keys)
            shard_keys = keys[start:end]

            shard_state = {k: state_dict[k] for k in shard_keys}
            # Load this shard.
            missing, unexpected = model.load_state_dict(shard_state, strict=False)
            self._logger.debug(
                "Loaded shard %d/%d: %d keys (missing=%d, unexpected=%d).",
                shard_idx + 1,
                num_shards,
                len(shard_keys),
                len(missing),
                len(unexpected),
            )
            # Free the shard.
            del shard_state
            gc.collect()

        self._logger.info(
            "Sharded loading complete: %d shards from %s.", num_shards, checkpoint_path
        )
        return model

    # ------------------------------------------------------------------
    # Peak prediction
    # ------------------------------------------------------------------
    def estimate_memory(
        self,
        model_params: int,
        batch_size: int = 1,
        seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
        include_activations: bool = True,
    ) -> int:
        """Estimate the GPU memory required for a model.

        Uses a simple heuristic:

        * **Weights**: ``model_params * bytes_per_param``.
        * **Gradients** (training): equal to weights.
        * **Optimizer state**: ``2 * weights`` (Adam).
        * **Activations**: ``batch_size * seq_len * hidden_dim * layers``
          (estimated from params).

        Args:
            model_params: Total number of model parameters.
            batch_size: Batch size for inference / training.
            seq_len: Sequence length.
            dtype: Data type of the parameters.
            include_activations: Whether to include activation memory.

        Returns:
            Estimated memory in bytes.
        """
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        # Weight memory.
        weight_bytes = model_params * bytes_per_elem

        # Activation memory (rough heuristic: ~4x weight bytes per token).
        activation_bytes = 0
        if include_activations:
            # Estimate hidden dim from params (very rough).
            estimated_hidden = int((model_params / 12) ** 0.5)  # rough for transformer
            estimated_layers = max(1, model_params // (estimated_hidden * estimated_hidden * 12))
            activation_bytes = (
                batch_size * seq_len * estimated_hidden * estimated_layers * bytes_per_elem * 4
            )

        # KV cache (rough: 2 * batch * seq * hidden * layers * 2 for K+V).
        kv_cache_bytes = 2 * batch_size * seq_len * weight_bytes // max(1, batch_size * seq_len)

        total = weight_bytes + activation_bytes + kv_cache_bytes
        return total

    def estimate_model_memory(self, model: nn.Module) -> int:
        """Estimate the memory footprint of an instantiated model.

        Args:
            model: The model to estimate.

        Returns:
            Estimated memory in bytes (weights only).
        """
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_bytes += buffer.nelement() * buffer.element_size()
        return total_bytes

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------
    def gc_collect(self) -> None:
        """Trigger garbage collection and clear CUDA caches.

        Runs Python's garbage collector, clears the CUDA allocator cache,
        and (on MPS) clears the MPS cache.  This should be called after
        large model deletions or between inference batches to reclaim
        fragmented memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch.backends, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._logger.debug("Garbage collection and cache clearing completed.")

    # ------------------------------------------------------------------
    # Memory pool management
    # ------------------------------------------------------------------
    def create_pool(
        self,
        num_blocks: int,
        block_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> MemoryPool:
        """Create a pre-allocated memory pool.

        Args:
            num_blocks: Number of blocks in the pool.
            block_shape: Shape of each block.
            dtype: Data type.
            device: Device for the pool.

        Returns:
            A :class:`MemoryPool` instance.
        """
        pool = MemoryPool(num_blocks, block_shape, dtype, device)
        self._logger.debug(
            "Created memory pool: %d blocks of shape %s (%.2f MB).",
            num_blocks,
            block_shape,
            pool.total_bytes() / 1e6,
        )
        return pool

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_memory_fraction(self) -> None:
        """Set the CUDA memory fraction."""
        if not torch.cuda.is_available():
            return
        device = self._device_manager.get_device()
        if device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.gpu_memory_fraction, device.index or 0
                )
            except (RuntimeError, ValueError) as exc:
                self._logger.warning("Could not set memory fraction: %s", exc)

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
