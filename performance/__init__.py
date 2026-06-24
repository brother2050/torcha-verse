"""Performance layer for the TorchaVerse framework (v0.3.0).

This package provides PyTorch-native performance optimisation, model
quantisation and a unified benchmark harness.

Submodules
----------
* :mod:`performance.optimizer` -- :class:`PerformanceOptimizer`
  applies SDPA, ``torch.compile`` and CUDA-graph capture to models and
  pipelines; :class:`OptimizedModel` wraps the result.
* :mod:`performance.quantization` -- :class:`Quantizer` lowers model
  precision (``int8`` / ``int4`` / ``nf4`` / ``fp16`` / ``bf16``) and
  estimates the resulting memory savings.
* :mod:`performance.benchmark` -- :class:`BenchmarkSuite` measures
  latency, throughput and peak memory for text, image, video and
  subtitle generation, and diffs two result sets via
  :class:`ComparisonReport`.

The package **depends on ``torch``**.  Optional backends
(``bitsandbytes``) are imported lazily with ``try/except`` guards.

Example:
    >>> from performance import PerformanceOptimizer, Quantizer, BenchmarkSuite
    >>> _ = PerformanceOptimizer()
    >>> _ = Quantizer()
    >>> _ = BenchmarkSuite()
"""

from __future__ import annotations

from .benchmark import BenchmarkSuite, ComparisonReport
from .optimizer import (
    BenchmarkResult,
    OptimizationConfig,
    OptimizedModel,
    PerformanceOptimizer,
)
from .quantization import (
    CalibrationResult,
    QuantizationConfig,
    QuantizedModel,
    Quantizer,
)

__all__ = [
    # optimizer
    "OptimizationConfig",
    "PerformanceOptimizer",
    "OptimizedModel",
    "BenchmarkResult",
    # quantization
    "QuantizationConfig",
    "Quantizer",
    "QuantizedModel",
    "CalibrationResult",
    # benchmark
    "BenchmarkSuite",
    "ComparisonReport",
]
