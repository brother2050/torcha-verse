"""Smoke tests for :mod:`performance.optimizer`.

Validates the public API of :class:`PerformanceOptimizer` against a
trivial ``nn.Linear`` model: ``optimize_model`` returns a working
``OptimizedModel`` wrapper and ``benchmark`` produces a non-negative
latency measurement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from performance.optimizer import (
    OptimizationConfig,
    OptimizedModel,
    PerformanceOptimizer,
)


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------
class TestOptimizationConfig:
    def test_defaults(self) -> None:
        cfg = OptimizationConfig()
        # ``OptimizationConfig`` enables SDPA by default and exposes
        # a compile mode selector.  We just assert the fields exist
        # and the enums are valid strings.
        assert isinstance(cfg.enable_sdpa, bool)
        assert isinstance(cfg.enable_compile, bool)
        assert isinstance(cfg.enable_cuda_graph, bool)
        assert cfg.compile_mode in ("default", "reduce-overhead", "max-autotune")


# ---------------------------------------------------------------------------
# PerformanceOptimizer
# ---------------------------------------------------------------------------
class TestPerformanceOptimizer:
    def test_optimize_model_returns_wrapper(self) -> None:
        model = nn.Linear(4, 4)
        opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
        out = opt.optimize_model(model)
        assert isinstance(out, OptimizedModel)
        y = out(torch.randn(2, 4))
        assert y.shape == (2, 4)

    def test_optimized_model_forwards_generate(self) -> None:
        """A model with a ``generate`` method should still expose it."""

        class _HasGenerate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def generate(self, x: torch.Tensor, **kw: object) -> torch.Tensor:
                return self.forward(x) + 1.0

        model = _HasGenerate()
        opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
        out = opt.optimize_model(model)
        # ``generate`` should be proxied.
        y = out.generate(torch.randn(2, 4))
        assert y.shape == (2, 4)

    def test_benchmark_returns_positive_latency(self) -> None:
        model = nn.Linear(4, 4)
        opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
        # ``benchmark`` takes ``input_shape`` (sequence of shapes) not
        # a sample tensor.
        result = opt.benchmark(model, input_shape=(2, 4), warmup=1, runs=3)
        assert result.latency_ms >= 0.0
        # ``BenchmarkResult`` exposes ``throughput`` (items/sec),
        # not ``throughput_per_sec``.
        assert result.throughput >= 0.0

    def test_active_optimizations_is_list(self) -> None:
        opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
        assert isinstance(opt._active_optimizations(), list)
