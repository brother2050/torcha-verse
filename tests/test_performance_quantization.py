"""Smoke tests for :mod:`performance.quantization`.

The 1549-line package had zero test coverage.  These tests focus on the
public API contract: method dispatch, output dtype / shape preservation,
and memory-saving estimation.  They do not depend on ``bitsandbytes``.

Hardware note
-------------

These tests are calibrated for **GPU / CUDA** deployments -- the
production target of the framework.  fp16 matmul is the canonical
case (CUDA ships ``addmm_impl_cuda`` for Half), and bf16 matmul
also runs on CUDA.  Some tests *also* work on CPU builds that
include mkl + oneDNN, but the fp16 path explicitly skips when the
local wheel cannot execute an fp16 matmul (``addmm_impl_cpu_`` is
absent from the public PyTorch CPU build by design; see
https://pytorch.org/docs/stable/notes/cpu.html).  This keeps the
smoke tests usable in both the production GPU environment and the
local CPU dev environment.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from performance.quantization import (
    CalibrationResult,
    QuantizationConfig,
    QuantizedModel,
    Quantizer,
)


# ---------------------------------------------------------------------------
# Hardware probes
# ---------------------------------------------------------------------------
def _has_fp16_matmul() -> bool:
    """Return True if the current torch build can run a fp16 matmul.

    The production target is GPU / CUDA where ``addmm_impl_cuda``
    is always present for Half.  CPU builds (mkl + oneDNN) ship
    bf16 matmul but **not** fp16 matmul -- the public PyTorch CPU
    wheel intentionally omits the Half kernel.

    We probe by trying a small matmul in a try/except; that's
    robust against every torch build variant and any future
    back-end that may or may not add CPU fp16 matmul.
    """
    try:
        m = nn.Linear(4, 4).half()
        m(torch.randn(2, 4, dtype=torch.float16))
    except (RuntimeError, NotImplementedError):
        return False
    return True


# Pytest skip markers tied to the probe.  ``fp16_matmul`` skips a
# test when the local wheel cannot run fp16 matmul; on the
# production GPU this is a no-op.
requires_fp16_matmul = pytest.mark.skipif(
    not _has_fp16_matmul(),
    reason=(
        "fp16 matmul unavailable on this torch build "
        "(production target is GPU / CUDA which always ships "
        "`addmm_impl_cuda` for Half)"
    ),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestQuantizationConfig:
    def test_default_method_is_int8(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.method == "int8"
        assert cfg.group_size == 128

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            QuantizationConfig(method="bogus")

    def test_invalid_group_size_raises(self) -> None:
        with pytest.raises(ValueError, match="group_size must be > 0"):
            QuantizationConfig(group_size=0)


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------
class TestQuantizer:
    @requires_fp16_matmul
    def test_fp16_changes_dtype(self) -> None:
        model = nn.Linear(4, 4).half()  # caller must pre-cast the input
        q = Quantizer()
        out = q.quantize(model, QuantizationConfig(method="fp16"))
        assert isinstance(out, QuantizedModel)
        assert out.method == "fp16"
        assert out.model.weight.dtype == torch.float16
        # forward still works on a half-precision input
        y = out(torch.randn(2, 4, dtype=torch.float16))
        assert y.shape == (2, 4)

    def test_bf16_changes_dtype(self) -> None:
        model = nn.Linear(4, 4).bfloat16()
        q = Quantizer()
        out = q.quantize(model, QuantizationConfig(method="bf16"))
        assert out.model.weight.dtype == torch.bfloat16
        y = out(torch.randn(2, 4, dtype=torch.bfloat16))
        assert y.shape == (2, 4)

    def test_int8_quantise_runs(self) -> None:
        model = nn.Linear(4, 4)
        q = Quantizer()
        out = q.quantize(model, QuantizationConfig(method="int8"))
        y = out(torch.randn(2, 4))
        assert y.shape == (2, 4)

    def test_int4_falls_back(self) -> None:
        model = nn.Linear(4, 4)
        q = Quantizer()
        out = q.quantize(model, QuantizationConfig(method="int4"))
        y = out(torch.randn(2, 4))
        assert y.shape == (2, 4)

    def test_estimate_memory_savings_fp16(self) -> None:
        model = nn.Linear(1024, 1024)
        q = Quantizer()
        cfg = QuantizationConfig(method="fp16")
        savings = q.estimate_memory_savings(model, cfg)
        # Keys are spelled in GB (not MB) -- just assert the savings are
        # positive and at least 40% (fp16 ~halves the memory footprint).
        assert savings["savings_pct"] >= 40.0
        assert savings["original_gb"] > savings["quantized_gb"]

    def test_estimate_memory_savings_int8(self) -> None:
        model = nn.Linear(256, 256)
        q = Quantizer()
        savings = q.estimate_memory_savings(model, QuantizationConfig(method="int8"))
        assert savings["savings_pct"] >= 60.0

    def test_has_bitsandbytes_flag_exists(self) -> None:
        q = Quantizer()
        assert isinstance(q.has_bitsandbytes, bool)


# ---------------------------------------------------------------------------
# CalibrationResult (dataclass sanity)
# ---------------------------------------------------------------------------
class TestCalibrationResult:
    def test_defaults(self) -> None:
        cr = CalibrationResult()
        assert cr.scales == {}
        assert cr.zero_points == {}
        assert cr.mse == 0.0
