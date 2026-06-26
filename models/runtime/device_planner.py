"""设备规划器 (v0.10.0)。

类似 ``accelerate`` 的 ``infer_auto_device_map`` 但:

* **零外部依赖** (不用 ``accelerate``)。
* 只做 CPU / CUDA / MPS / 多 GPU 的**粗粒度**分配,不做 pipeline
  并行的复杂平衡。
* 接口与 diffusers 兼容 (``device_map="cpu"|"cuda"|"cuda:0"|`` 
  ``{"layer.0": "cuda:0", "layer.1": "cpu"}``)。

设计原则:

1. **优先级** (从高到低):
   ``CUDA > MPS > CPU`` (当 ``device=None`` 时)。
2. **多 GPU**:如果检测到多张 CUDA 卡,默认按层数 N 平均切片
   (N=2 时 layer 0..L/2 → cuda:0, layer L/2..L → cuda:1)。
3. **dtype 推断**:
   - 用户显式给 → 用
   - 缺省 + CUDA  → ``torch.float16`` (与 diffusers default 对齐)
   - 缺省 + CPU  → ``torch.float32`` (CPU fp16 慢且部分 kernel 缺)
4. **失败优雅**:任何设备探测失败 → 退回 ``cpu`` + ``float32``,
   不抛异常 (与 diffusers 的 ``enable_model_cpu_offload`` 行为对齐)。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

__all__ = [
    "DevicePlan",
    "plan_device",
    "pick_default_device",
    "get_device_map",
    "is_cuda_available",
    "is_mps_available",
]


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------
def is_cuda_available() -> bool:
    """``True`` when ``torch.cuda.is_available()`` returns True."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def is_mps_available() -> bool:
    """``True`` when MPS (Apple Silicon) is built and available."""
    try:
        return bool(getattr(torch.backends, "mps", None)) and bool(
            torch.backends.mps.is_available()
        )
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# DevicePlan
# ---------------------------------------------------------------------------
@dataclass
class DevicePlan:
    """The result of a :func:`plan_device` call.

    Attributes:
        device: The single device the entire model will live on (when
            no per-layer ``device_map`` is in effect).  ``None`` means
            the caller should use :attr:`device_map` instead.
        dtype: The recommended ``torch.dtype`` for weights.
        device_map: Optional per-layer device mapping.  When set, the
            caller is expected to apply it via
            :meth:`models.base.ModelMixin._apply_device_map`.
        notes: Human-readable notes about how the plan was built
            (useful for logging).
    """

    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    device_map: Optional[Dict[str, str]] = None
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.device_map is not None:
            return (
                f"DevicePlan(device_map={self.device_map!r}, "
                f"dtype={self.dtype}, notes={self.notes!r})"
            )
        return (
            f"DevicePlan(device={self.device!r}, dtype={self.dtype}, "
            f"notes={self.notes!r})"
        )


# ---------------------------------------------------------------------------
# pick_default_device
# ---------------------------------------------------------------------------
def pick_default_device() -> torch.device:
    """Return the project's preferred default device.

    Priority: ``CUDA > MPS > CPU``.  No exceptions are raised on
    failure -- the function always returns at least a CPU device.
    """
    if is_cuda_available():
        try:
            return torch.device("cuda:0")
        except Exception:  # noqa: BLE001
            pass
    if is_mps_available():
        try:
            return torch.device("mps")
        except Exception:  # noqa: BLE001
            pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# dtype inference
# ---------------------------------------------------------------------------
def _infer_dtype(
    requested: Optional[torch.dtype],
    device: torch.device,
) -> torch.dtype:
    """Resolve a final ``torch.dtype`` from user request + target device.

    Rules:

    1. If the user supplied a dtype, use it.
    2. Else if the target is a CUDA device, default to ``float16``.
    3. Else default to ``float32`` (CPU fp16 is slow and some
       kernels are missing for many ops).
    """
    if requested is not None:
        return requested
    if device.type == "cuda":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Multi-GPU slicing
# ---------------------------------------------------------------------------
def _shard_module_across_gpus(
    n_gpus: int,
    module: Optional[torch.nn.Module] = None,
) -> Dict[str, str]:
    """Evenly slice a module's top-level children across N CUDA devices.

    The sharding is intentionally simple: ``n_gpus=2`` maps the first
    half of named children to ``cuda:0`` and the second half to
    ``cuda:1``.  This is the same heuristic diffusers uses when
    ``device_map="balanced"`` is requested for a model with explicit
    layer structure.

    Args:
        n_gpus: Number of CUDA devices to use.  Must be >= 1.
        module: An optional module whose top-level child names drive
            the sharding.  When ``None`` we emit a generic
            ``layers.{0..N-1}`` slicing, which is what most
            Transformer / DiT / UNet layouts follow.

    Returns:
        A ``{child_name: "cuda:i"}`` mapping suitable for
        :class:`models.base.ModelMixin._apply_device_map`.
    """
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")
    # Collect child names.  Prefer the real module's children; fall
    # back to a generic "layers" layout.
    if module is not None:
        names = [name for name, _ in module.named_children()]
    else:
        names = [f"layers.{i}" for i in range(n_gpus * 2)]  # 2 per GPU
    if not names:
        return {}
    # Even split.
    n = len(names)
    per_gpu = max(1, (n + n_gpus - 1) // n_gpus)
    out: Dict[str, str] = {}
    for i, name in enumerate(names):
        gpu_idx = min(i // per_gpu, n_gpus - 1)
        out[name] = f"cuda:{gpu_idx}"
    return out


# ---------------------------------------------------------------------------
# get_device_map
# ---------------------------------------------------------------------------
def get_device_map(
    requested: Union[None, str, Dict[str, str], torch.device],
    *,
    model: Optional[torch.nn.Module] = None,
) -> DevicePlan:
    """Translate a (possibly-string) ``device_map`` request into a
    fully-resolved :class:`DevicePlan`.

    Accepted forms for ``requested``:

    * ``None`` -- use the project default (CUDA > MPS > CPU).
    * ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` -- a single
      device.
    * ``"balanced"`` / ``"auto"`` -- multi-GPU even split (only when
      CUDA is available and n_gpus > 1).
    * ``{"layer.0": "cuda:0", "layer.1": "cpu"}`` -- diffusers-style
      per-layer sharding (passed through verbatim).

    Args:
        requested: The caller's device / device_map spec.
        model: An optional module used to drive multi-GPU slicing.

    Returns:
        A :class:`DevicePlan` with at least one of ``device`` /
        ``device_map`` populated.
    """
    notes: List[str] = []

    # None → project default
    if requested is None:
        device = pick_default_device()
        notes.append(f"default device={device!r}")
        return DevicePlan(
            device=device,
            dtype=_infer_dtype(None, device),
            notes=notes,
        )

    # Already a torch.device
    if isinstance(requested, torch.device):
        notes.append(f"explicit device={requested!r}")
        return DevicePlan(
            device=requested,
            dtype=_infer_dtype(None, requested),
            notes=notes,
        )

    # String
    if isinstance(requested, str):
        # Multi-GPU shortcuts
        if requested in ("balanced", "auto"):
            if is_cuda_available():
                try:
                    n_gpus = torch.cuda.device_count()
                except Exception:  # noqa: BLE001
                    n_gpus = 1
                if n_gpus > 1:
                    plan = _shard_module_across_gpus(n_gpus, model)
                    notes.append(
                        f"multi-GPU 'balanced' across {n_gpus} devices"
                    )
                    return DevicePlan(
                        device=None,
                        dtype=torch.float16,
                        device_map=plan,
                        notes=notes,
                    )
            # Fall through to single-device path.
            device = pick_default_device()
            notes.append(
                f"'balanced' requested but only 1 GPU; using {device!r}"
            )
            return DevicePlan(
                device=device,
                dtype=_infer_dtype(None, device),
                notes=notes,
            )

        # Plain string device
        try:
            device = torch.device(requested)
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"failed to parse {requested!r} ({exc}); defaulting to cpu"
            )
            return DevicePlan(
                device=torch.device("cpu"),
                dtype=torch.float32,
                notes=notes,
            )
        notes.append(f"string device={device!r}")
        return DevicePlan(
            device=device,
            dtype=_infer_dtype(None, device),
            notes=notes,
        )

    # Mapping
    if isinstance(requested, Mapping):
        # Pass through verbatim; the device type decides the dtype.
        first_dev_str = next(iter(requested.values()), "cpu")
        try:
            inferred_device = torch.device(first_dev_str)
        except Exception:  # noqa: BLE001
            inferred_device = torch.device("cpu")
        notes.append(
            f"per-layer device_map with {len(requested)} entries"
        )
        return DevicePlan(
            device=None,
            dtype=_infer_dtype(None, inferred_device),
            device_map=dict(requested),
            notes=notes,
        )

    # Unknown type → safe fallback
    notes.append(f"unsupported device spec {type(requested).__name__}; cpu")
    return DevicePlan(
        device=torch.device("cpu"),
        dtype=torch.float32,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# plan_device
# ---------------------------------------------------------------------------
def plan_device(
    requested: Union[None, str, torch.device] = None,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    model: Optional[torch.nn.Module] = None,
) -> DevicePlan:
    """Resolve (device, dtype) for a model load.

    This is the thin convenience wrapper around :func:`get_device_map`
    used by :func:`models.runtime.local_loader.load_model_and_tokenizer`.

    Args:
        requested: ``None`` / a torch.device / a string spec.
        torch_dtype: User-requested dtype.  ``None`` triggers the
            device-aware default heuristic.
        model: An optional module used for multi-GPU slicing.

    Returns:
        A :class:`DevicePlan`.
    """
    plan = get_device_map(requested, model=model)
    if torch_dtype is not None:
        plan.dtype = torch_dtype
    return plan
