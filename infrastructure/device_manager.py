"""Device and distributed-computing abstractions for TorchaVerse.

This module centralises all hardware-related decisions: device selection
(CUDA / MPS / CPU), tensor and model migration, mixed-precision dtype
policies, and the DistributedDataParallel (DDP) lifecycle.  Interface
definitions for tensor and pipeline parallelism are also provided so that
higher layers can program against a stable contract even before the
concrete implementations are available.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn

__all__ = [
    "DeviceManager",
    "DTypePolicy",
    "TensorParallel",
    "PipelineParallel",
]

#: 模块级日志器，用于记录设备操作中的警告信息。
_logger: logging.Logger = logging.getLogger("infrastructure.device_manager")


# ---------------------------------------------------------------------------
# DType policy
# ---------------------------------------------------------------------------
class DTypePolicy:
    """Manage mixed-precision dtype selection (BF16 / FP16 / FP32).

    The policy translates human-readable dtype strings (as found in the YAML
    configuration files, e.g. ``"bf16"``) into actual ``torch.dtype`` objects
    and validates that the active device supports the requested precision.
    """

    #: Mapping from config string to ``torch.dtype``.
    _DTYPE_MAP: Dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, default_dtype: Union[str, torch.dtype] = "bf16") -> None:
        """Initialise the policy with a default dtype.

        Args:
            default_dtype: Default precision used when no explicit dtype is
                requested.  Accepts either a config string or a
                ``torch.dtype``.
        """
        self._default_dtype: torch.dtype = self.get_dtype(default_dtype)

    # ------------------------------------------------------------------
    @classmethod
    def get_dtype(cls, dtype: Union[str, torch.dtype]) -> torch.dtype:
        """Convert a dtype string or ``torch.dtype`` into a ``torch.dtype``.

        Args:
            dtype: One of ``"fp32"``, ``"fp16"``, ``"bf16"`` (case-insensitive)
                or an existing ``torch.dtype``.

        Returns:
            The corresponding ``torch.dtype``.

        Raises:
            ValueError: If ``dtype`` is not recognised.
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        key = str(dtype).strip().lower()
        if key not in cls._DTYPE_MAP:
            supported = ", ".join(sorted(set(cls._DTYPE_MAP.keys())))
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported: {supported}."
            )
        return cls._DTYPE_MAP[key]

    @classmethod
    def get_dtype_string(cls, dtype: torch.dtype) -> str:
        """Return the canonical config string for a ``torch.dtype``."""
        reverse = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }
        if dtype not in reverse:
            raise ValueError(f"No canonical string for dtype {dtype}.")
        return reverse[dtype]

    @property
    def default_dtype(self) -> torch.dtype:
        """The default dtype assigned by this policy."""
        return self._default_dtype

    def set_default_dtype(self, dtype: Union[str, torch.dtype]) -> None:
        """Update the default dtype."""
        self._default_dtype = self.get_dtype(dtype)

    def is_supported(
        self, dtype: Union[str, torch.dtype], device: Optional[torch.device] = None
    ) -> bool:
        """Check whether ``device`` supports the requested precision.

        On CPU, ``fp16`` is generally not natively supported for compute.
        """
        resolved = self.get_dtype(dtype)
        if device is None:
            device = DeviceManager().get_device()
        if device.type == "cpu" and resolved == torch.float16:
            return False
        return True

    def resolve(
        self, dtype: Optional[Union[str, torch.dtype]] = None, device: Optional[torch.device] = None
    ) -> torch.dtype:
        """Resolve a requested dtype, falling back to the policy default.

        Args:
            dtype: Explicit dtype or ``None`` to use the default.
            device: Device used to validate support.

        Returns:
            A validated ``torch.dtype``.
        """
        target = self._default_dtype if dtype is None else self.get_dtype(dtype)
        if device is None:
            device = DeviceManager().get_device()
        if not self.is_supported(target, device):
            # Gracefully fall back to fp32 when the device cannot handle it.
            return torch.float32
        return target


# ---------------------------------------------------------------------------
# Parallelism protocols (interfaces)
# ---------------------------------------------------------------------------
@runtime_checkable
class TensorParallel(Protocol):
    """Interface for tensor-parallel model sharding.

    Concrete implementations will split individual layers/tensors across
    multiple devices so that a single forward pass is computed in parallel.
    """

    def shard(self, model: nn.Module, num_devices: int) -> nn.Module:
        """Shard ``model`` across ``num_devices`` devices.

        Args:
            model: The model to shard.
            num_devices: Number of devices to shard across.

        Returns:
            The sharded model wrapper.

        Note:
            Not yet implemented. Concrete support is planned.
        """
        ...

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a sharded tensor back to a single device."""
        ...


@runtime_checkable
class PipelineParallel(Protocol):
    """Interface for pipeline-parallel model staging.

    Concrete implementations will partition the model into stages, each
    placed on a different device, and micro-batch the activations through
    the stages.
    """

    def split_into_stages(
        self, model: nn.Module, num_stages: int, devices: List[torch.device]
    ) -> nn.Module:
        """Split ``model`` into ``num_stages`` pipeline stages.

        Args:
            model: The model to partition.
            num_stages: Number of pipeline stages.
            devices: Devices to assign each stage to.

        Returns:
            A pipeline-wrapped model.

        Note:
            Not yet implemented. Concrete support is planned.
        """
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run a forward pass through the pipeline stages."""
        ...


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------
class DeviceManager:
    """Unified device allocation and distributed-training abstraction.

    The manager auto-detects the best available hardware (CUDA, MPS, or CPU),
    exposes helpers to migrate tensors and models, wraps the DDP lifecycle,
    and reports detailed device information.

    Implemented as a singleton so that every component agrees on the active
    device and distributed settings.
    """

    _instance: Optional["DeviceManager"] = None
    _initialized: bool = False
    # Class-level lock guarding singleton creation / initialisation so
    # that concurrent ``DeviceManager()`` calls cannot race past the
    # ``_initialized`` flag (TOCTOU).
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "DeviceManager":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:  # double-check
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        # Fast path: already initialised -- avoid the lock entirely.
        if self._initialized:
            return
        with self._singleton_lock:
            # Double-check under the lock to prevent two threads from
            # initialising concurrently.
            if self._initialized:
                return
            self._initialized = True

            self._device: torch.device = self._resolve_device(device)
            self._dtype_policy: DTypePolicy = DTypePolicy()
            self._ddp_initialized: bool = False
            self._local_rank: int = 0
            self._world_size: int = 1

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_device() -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve a user-provided device specification.

        ``"auto"`` triggers auto-detection.
        """
        if device is None or device == "auto":
            return self._detect_device()
        if isinstance(device, str):
            return torch.device(device)
        return device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_device(self) -> torch.device:
        """Return the best available (or explicitly configured) device."""
        return self._device

    def set_device(self, device: Union[str, torch.device]) -> None:
        """Override the active device.

        Args:
            device: A device string (``"cuda"``, ``"cuda:1"``, ``"cpu"`` ...)
                or ``torch.device``.
        """
        with self._singleton_lock:
            self._device = self._resolve_device(device)
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            _logger.info("Device set to %s", self._device)

    @property
    def dtype_policy(self) -> DTypePolicy:
        """The active :class:`DTypePolicy`."""
        return self._dtype_policy

    @property
    def is_distributed(self) -> bool:
        """``True`` when DDP has been initialised."""
        return self._ddp_initialized

    @property
    def local_rank(self) -> int:
        """Local rank of the current process within the DDP group."""
        return self._local_rank

    @property
    def world_size(self) -> int:
        """Total number of processes in the DDP group."""
        return self._world_size

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------
    def to_device(
        self,
        obj: Union[torch.Tensor, nn.Module, Dict[str, Any], List[Any]],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> Union[torch.Tensor, nn.Module, Dict[str, Any], List[Any]]:
        """Migrate a tensor, model, or collection to ``device``.

        Args:
            obj: A tensor, ``nn.Module``, dict, or list thereof.
            device: Target device. Defaults to the manager's active device.
            dtype: Optional dtype to cast tensors to.

        Returns:
            The migrated object (tensors/models are moved in place where
            possible).
        """
        target_device = self._resolve_device(device)
        target_dtype = self._dtype_policy.resolve(dtype, target_device) if dtype else None

        if isinstance(obj, torch.Tensor):
            tensor = obj.to(target_device)
            if target_dtype is not None:
                tensor = tensor.to(target_dtype)
            return tensor

        if isinstance(obj, nn.Module):
            obj.to(target_device)
            if target_dtype is not None:
                obj = obj.to(target_dtype)
            return obj

        if isinstance(obj, dict):
            return {k: self.to_device(v, target_device, dtype) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            converted = [self.to_device(v, target_device, dtype) for v in obj]
            return type(obj)(converted)  # type: ignore[call-arg]

        raise TypeError(
            f"to_device() does not support objects of type "
            f"{type(obj).__name__}."
        )

    # ------------------------------------------------------------------
    # Distributed Data Parallel
    # ------------------------------------------------------------------
    def setup_ddp(
        self,
        backend: Optional[str] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        """Initialise the DistributedDataParallel process group.

        When the standard environment variables (``RANK``, ``WORLD_SIZE``,
        ``LOCAL_RANK``) are set (e.g. by ``torchrun``) they are used
        automatically.  Otherwise the explicit ``rank``/``world_size``
        arguments take precedence.

        Args:
            backend: ``"nccl"`` (default for CUDA) or ``"gloo"``.
            rank: Global rank of this process.
            world_size: Total number of processes.
        """
        if self._ddp_initialized:
            return

        rank = int(rank if rank is not None else os.environ.get("RANK", 0))
        world_size = int(
            world_size if world_size is not None else os.environ.get("WORLD_SIZE", 1)
        )
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # 参数校验：rank / world_size / local_rank 必须处于合法范围。
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        if local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {local_rank}")

        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        if world_size > 1:
            torch.distributed.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )

        self._local_rank = local_rank
        self._world_size = world_size

        # Pin the device for this process.
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            self._device = device

        self._ddp_initialized = True

    def cleanup_ddp(self) -> None:
        """Destroy the DDP process group and reset distributed state."""
        if self._ddp_initialized and torch.distributed.is_available():
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        self._ddp_initialized = False
        self._local_rank = 0
        self._world_size = 1

    def wrap_ddp(
        self,
        model: nn.Module,
        device_ids: Optional[List[int]] = None,
        find_unused_parameters: bool = False,
    ) -> nn.Module:
        """Wrap a model with ``DistributedDataParallel``.

        Args:
            model: The model to wrap. It is moved to the local device first.
            device_ids: CUDA device ids for DDP. Defaults to ``[local_rank]``.
            find_unused_parameters: Forwarded to DDP.

        Returns:
            The DDP-wrapped model.
        """
        if not self._ddp_initialized:
            raise RuntimeError(
                "DDP has not been initialised. Call setup_ddp() first."
            )
        model = self.to_device(model)  # type: ignore[assignment]
        if self._device.type == "cuda":
            if device_ids is None:
                device_ids = [self._local_rank]
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids,
                find_unused_parameters=find_unused_parameters,
            )
        return nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=find_unused_parameters
        )

    # ------------------------------------------------------------------
    # Parallelism (interface placeholders)
    # ------------------------------------------------------------------
    def tensor_parallel(self, model: nn.Module, num_devices: int) -> nn.Module:
        """Shard ``model`` using tensor parallelism.

        Note:
            Not yet implemented. This is a placeholder that raises
            ``NotImplementedError`` until a concrete backend is available.
        """
        raise NotImplementedError(
            "Tensor parallelism is not yet implemented. See the "
            "TensorParallel protocol for the planned interface."
        )

    def pipeline_parallel(
        self, model: nn.Module, num_stages: int, devices: Optional[List[torch.device]] = None
    ) -> nn.Module:
        """Partition ``model`` into pipeline stages.

        Note:
            Not yet implemented. This is a placeholder that raises
            ``NotImplementedError`` until a concrete backend is available.
        """
        raise NotImplementedError(
            "Pipeline parallelism is not yet implemented. See the "
            "PipelineParallel protocol for the planned interface."
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_device_info(self) -> Dict[str, Any]:
        """Return a dictionary describing the active device(s).

        The returned dictionary contains the device type, index, CUDA device
        count, name, total memory (when applicable), and distributed state.
        """
        device = self._device
        info: Dict[str, Any] = {
            "device": str(device),
            "type": device.type,
            "index": device.index,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "distributed": self._ddp_initialized,
            "local_rank": self._local_rank,
            "world_size": self._world_size,
        }

        if device.type == "cuda":
            info["name"] = torch.cuda.get_device_name(device)
            info["capability"] = torch.cuda.get_device_capability(device)
            info["total_memory_bytes"] = torch.cuda.get_device_properties(
                device
            ).total_memory
            info["total_memory_gb"] = round(
                info["total_memory_bytes"] / (1024 ** 3), 2
            )
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            info["memory_allocated_bytes"] = allocated
            info["memory_reserved_bytes"] = reserved
        else:
            info["name"] = device.type.upper()

        return info

    def empty_cache(self) -> None:
        """Release unoccupied cached memory held by the allocator."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - 最佳努力
                    _logger.warning("清空 MPS 缓存失败，已跳过: %s", exc)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._singleton_lock:
            if cls._instance is not None:
                try:
                    cls._instance.cleanup_ddp()
                except Exception:
                    pass
            cls._instance = None
            cls._initialized = False
