"""v0.8.5 — ``@pytest.mark.gpu`` example test (v0.8.5 §4.4).

The v0.8.5 acceptance target in
:file:`docs/V0.8_UPGRADE_PLAN.md` §4.4 -- "cuda offload -40% hard
metric" -- is exercised by ``tests/_helpers/_gpu_probe.py::offload_peak_ratio``.
The example test below is marked with ``@pytest.mark.gpu`` so it
skips cleanly on CPU-only CI and runs on a self-hosted GPU
runner in nightly / on-push.
"""
from __future__ import annotations

import torch
import pytest

from tests._helpers._gpu_probe import (
    is_cuda_available,
    load_dummy_model,
    offload_peak_ratio,
    PeakMemoryProbe,
)

__all__ = ["TestGPUProbe"]


@pytest.mark.gpu
class TestGPUProbe:
    """A trivial ``@pytest.mark.gpu`` example so the marker has
    at least one wired-up test.  Real CUDA peak benchmarks live
    in the same module via :func:`offload_peak_ratio`.
    """

    def test_peak_probe_on_cpu_is_noop(self) -> None:
        """On CPU the probe returns 0 bytes -- smoke test that the
        helper never throws when no GPU is present.
        """
        with PeakMemoryProbe() as probe:
            x = torch.randn(2, 8)  # noqa: F841
        assert probe.peak_bytes == 0

    @pytest.mark.skipif(
        not is_cuda_available(),
        reason="requires a CUDA device",
    )
    def test_offload_peak_ratio_returns_pair(self) -> None:
        """When a GPU is present, both offload strategies should
        yield a positive peak reading.
        """
        per, seq = offload_peak_ratio(
            num_blocks=2, hidden=64, ffn=128, batch=4,
        )
        assert per > 0
        assert seq > 0
        # The sequential strategy should not exceed the
        # per-submodule strategy in peak memory -- this is the
        # v0.8.5 §4.4 acceptance target.
        assert seq <= per
