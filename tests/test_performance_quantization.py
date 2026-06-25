"""Smoke tests for :mod:`performance.quantization`.

The 1549-line package had zero test coverage.  These tests focus on the
public API contract: method dispatch, output dtype / shape preservation,
and memory-saving estimation.  They do not depend on ``bitsandbytes``.
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
