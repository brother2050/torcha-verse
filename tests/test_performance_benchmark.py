"""Smoke tests for :mod:`performance.benchmark`.

Validates the :class:`BenchmarkSuite` runs against a synthetic model
(provided via the ``model`` arg) and reports sane numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from performance.benchmark import BenchmarkSuite, BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def generate(self, prompt: str, max_new_tokens: int = 4, **kw: object) -> str:
        return (prompt + " ") * max_new_tokens


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------
class TestBenchmarkSuite:
    def test_text_benchmark_runs_with_synthetic_model(self) -> None:
        suite = BenchmarkSuite()
        result = suite.run_text_benchmark(
            "tiny",
            "hello",
            max_tokens=4,
            model=_TinyGenerator(),
        )
        assert isinstance(result, BenchmarkResult)
        assert result.latency_ms >= 0.0

    def test_text_benchmark_runs_without_model(self) -> None:
        """Without a model, the suite uses a synthetic micro-benchmark."""
        suite = BenchmarkSuite()
        result = suite.run_text_benchmark("tiny", "hi", max_tokens=4)
        assert result.latency_ms >= 0.0

    def test_image_benchmark_runs_without_model(self) -> None:
        suite = BenchmarkSuite()
        # ``run_image_benchmark`` takes ``steps`` not ``num_inference_steps``,
        # and ``width``/``height`` default to 512x512.
        result = suite.run_image_benchmark("tiny", "a cat", steps=2)
        assert result.latency_ms >= 0.0

    def test_subtitle_benchmark_runs_without_model(self) -> None:
        suite = BenchmarkSuite()
        # ``run_subtitle_benchmark`` takes ``audio_path`` as the first
        # positional argument (the model name is no longer required).
        result = suite.run_subtitle_benchmark("dummy.wav")
        assert result.latency_ms >= 0.0

    def test_warmup_default(self) -> None:
        suite = BenchmarkSuite()
        # ``BenchmarkSuite`` has a default warmup >= 1; we just assert
        # the field is set and the runs counter is positive.
        assert suite.warmup >= 0
        assert suite.runs >= 1

    def test_compare_returns_improvements(self) -> None:
        suite = BenchmarkSuite()
        # ``compare`` takes two dicts of ``name -> BenchmarkResult``
        # produced by ``run_all``, not two bare results.
        results_a = suite.run_all()
        results_b = suite.run_all()
        report = suite.compare(results_a, results_b)
        # ``improvements`` / ``regressions`` are already dicts -- no
        # ``.items()`` is required.
        assert isinstance(report.improvements, dict)
        assert isinstance(report.regressions, dict)
        assert isinstance(report.summary, str)
