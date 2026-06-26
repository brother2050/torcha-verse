"""v0.8.5 — ``@pytest.mark.gpu`` example + CPU-side offload peak helper.

The v0.8.5 / v1.0.0 acceptance target is::

    "enable_model_cpu_offload 在 cuda 设备上峰值 -40% 以上"

This module provides:

* :class:`PeakMemoryProbe` -- a context manager that records the
  current peak GPU memory.  It is a no-op on CPU so the same
  helper code runs in CI without a GPU.
* :func:`offload_peak_ratio` -- run ``model.forward`` with both
  per-submodule and sequential offload, and return the
  ``peak_per_submodule / peak_sequential`` ratio.  When both
  are run on the same hardware, the sequential mode is
  expected to be lower (the v0.8.5 acceptance target
  is a -40% gap in peak).
* :func:`is_cuda_available` -- a thin wrapper so the gpu-marked
  tests can ``pytest.skip`` cleanly on CPU-only CI.
* :func:`load_dummy_model` -- build a 0.4 GB-ish chain of
  ``nn.Linear`` blocks for peak measurements; the weights
  are random but the model is large enough to push peak
  memory above the noise floor.
"""
from __future__ import annotations

import contextlib
import gc
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn

from core.offload import (
    enable_model_cpu_offload,
    enable_sequential_cpu_offload,
)

__all__ = [
    "is_cuda_available",
    "PeakMemoryProbe",
    "load_dummy_model",
    "offload_peak_ratio",
]


def is_cuda_available() -> bool:
    """Return True when a CUDA device is reachable."""
    return bool(torch.cuda.is_available())


class PeakMemoryProbe:
    """A no-op-on-CPU context manager that captures peak GPU memory.

    Usage::

        with PeakMemoryProbe() as probe:
            _ = model(x)
        peak_bytes = probe.peak_bytes

    On CPU the probe returns ``0`` for ``peak_bytes``.
    """

    def __init__(self) -> None:
        self.peak_bytes: int = 0

    def __enter__(self) -> "PeakMemoryProbe":
        gc.collect()
        if is_cuda_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: object) -> None:
        if is_cuda_available():
            self.peak_bytes = int(torch.cuda.max_memory_allocated())


def load_dummy_model(
    *,
    num_blocks: int = 4,
    hidden: int = 256,
    ffn: int = 1024,
    seed: int = 0,
) -> nn.Module:
    """Build a chain of ``num_blocks`` ``nn.Linear`` blocks with
    a hidden dim of ``hidden`` and a feed-forward dim of ``ffn``.
    On CPU / fp32 the chain is roughly ``num_blocks * (hidden*hidden + hidden*ffn + ffn*hidden)`` * 4 bytes.

    For the default settings (4 * (256*256 + 256*1024 + 1024*256) = ~2.6M params = 10 MB) the model
    is large enough for the sequential / per-submodule offload
    difference to register on real hardware while still fitting
    comfortably on a 4 GB GPU.
    """
    torch.manual_seed(seed)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(hidden, ffn, bias=False)
            self.fc2 = nn.Linear(ffn, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    return nn.Sequential(*[_Block() for _ in range(num_blocks)])


@contextlib.contextmanager
def _seeded_run(model: nn.Module, batch: int, hidden: int) -> Iterator[torch.Tensor]:
    x = torch.randn(batch, hidden)
    yield x


def offload_peak_ratio(
    *,
    num_blocks: int = 4,
    hidden: int = 256,
    ffn: int = 1024,
    batch: int = 32,
) -> Tuple[int, int]:
    """Return ``(peak_per_submodule, peak_sequential)`` in bytes.

    Both measurements are taken on the same model (deep-copied
    so they share initial state) and the same input.  When the
    device is CPU the returned pair is ``(0, 0)``.
    """
    if not is_cuda_available():
        return (0, 0)
    peaks: List[int] = []
    for sequential in (False, True):
        model = load_dummy_model(
            num_blocks=num_blocks, hidden=hidden, ffn=ffn,
        ).cuda()
        if sequential:
            enable_sequential_cpu_offload(model, compute_device="cuda", offload_device="cpu")
        else:
            enable_model_cpu_offload(model, compute_device="cuda", offload_device="cpu")
        with PeakMemoryProbe() as probe:
            x = torch.randn(batch, hidden, device="cuda")
            _ = model(x).sum().backward()
        peaks.append(probe.peak_bytes)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return (peaks[0], peaks[1])
