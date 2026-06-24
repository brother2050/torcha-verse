"""Benchmark suite for the TorchaVerse framework.

This module provides :class:`BenchmarkSuite`, a unified harness for
measuring the latency, throughput and memory footprint of the four
generation modalities supported by the framework:

* :meth:`run_text_benchmark` -- text / LLM generation.
* :meth:`run_image_benchmark` -- image diffusion.
* :meth:`run_video_benchmark` -- video diffusion.
* :meth:`run_subtitle_benchmark` -- audio-to-subtitle transcription.

Each method returns a :class:`~performance.optimizer.BenchmarkResult`.
:meth:`run_all` executes every benchmark and returns a name-keyed
dictionary.  :meth:`compare` diffs two result sets and produces a
:class:`ComparisonReport` highlighting improvements and regressions.

The suite is designed to work with *any* model that exposes a
``generate`` / ``__call__`` interface.  When a real model is not
supplied the suite runs a synthetic micro-benchmark so the interface is
always exercisable.

The module **depends on ``torch``**.

Example:
    >>> from performance import BenchmarkSuite
    >>> suite = BenchmarkSuite()
    >>> result = suite.run_text_benchmark("dummy", "hello", max_tokens=8)
    >>> result.latency_ms >= 0.0
    True
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .optimizer import BenchmarkResult

__all__ = [
    "BenchmarkSuite",
    "ComparisonReport",
]

# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Default number of diffusion steps for image benchmarks.
_DEFAULT_IMAGE_STEPS: int = 20

#: Default number of diffusion steps for video benchmarks.
_DEFAULT_VIDEO_STEPS: int = 20

#: Default number of frames for video benchmarks.
_DEFAULT_VIDEO_FRAMES: int = 16

#: Default image width.
_DEFAULT_IMAGE_WIDTH: int = 512

#: Default image height.
_DEFAULT_IMAGE_HEIGHT: int = 512

#: Default number of warm-up iterations.
_DEFAULT_WARMUP: int = 1

#: Default number of measured iterations.
_DEFAULT_RUNS: int = 3

#: Bytes per GiB.
_BYTES_PER_GIB: float = 1024.0 ** 3

#: Relative improvement threshold (fraction) for "improvement".
_IMPROVEMENT_THRESHOLD: float = 0.05

#: Relative regression threshold (fraction) for "regression".
_REGRESSION_THRESHOLD: float = 0.05

#: Subtitle transcription methods recognised by the suite.
_SUBTITLE_METHODS: tuple[str, ...] = ("whisper", "wav2vec", "ctc")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ComparisonReport:
    """Diff between two benchmark result sets.

    Attributes:
        improvements: Metric -> relative improvement (positive fraction).
        regressions: Metric -> relative regression (positive fraction).
        summary: Human-readable summary string.
    """

    improvements: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """Unified benchmark harness for all TorchaVerse modalities.

    Args:
        device: Device on which to run the benchmarks.  Defaults to CUDA
            when available, otherwise CPU.
        warmup: Number of warm-up iterations per benchmark.
        runs: Number of measured iterations per benchmark.

    Example:
        >>> suite = BenchmarkSuite()
        >>> results = suite.run_all()
        >>> isinstance(results, dict)
        True
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        warmup: int = _DEFAULT_WARMUP,
        runs: int = _DEFAULT_RUNS,
    ) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}.")
        if runs <= 0:
            raise ValueError(f"runs must be > 0, got {runs}.")

        self._device: torch.device = (
            torch.device(device)
            if device
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        self._warmup: int = warmup
        self._runs: int = runs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """The device benchmarks run on."""
        return self._device

    @property
    def warmup(self) -> int:
        """Number of warm-up iterations."""
        return self._warmup

    @property
    def runs(self) -> int:
        """Number of measured iterations."""
        return self._runs

    # ------------------------------------------------------------------
    # Text benchmark
    # ------------------------------------------------------------------
    def run_text_benchmark(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        model: Optional[nn.Module] = None,
    ) -> BenchmarkResult:
        """Benchmark text generation.

        Args:
            model_name: Logical name of the model under test.
            prompt: Input prompt.
            max_tokens: Number of tokens to generate.
            model: Optional real model with a ``generate`` method.  When
                ``None`` a synthetic matmul workload is timed.

        Returns:
            A :class:`BenchmarkResult`.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}.")

        def _workload() -> Any:
            if model is not None and hasattr(model, "generate"):
                return model.generate(prompt, max_new_tokens=max_tokens)
            # Synthetic workload: simulate token generation.
            hidden = torch.randn(max_tokens, 1024, device=self._device)
            return hidden @ hidden.T

        return self._run_workload(
            workload=_workload,
            optimizations=[f"text:{model_name}", f"tokens={max_tokens}"],
            batch_size=1,
        )

    # ------------------------------------------------------------------
    # Image benchmark
    # ------------------------------------------------------------------
    def run_image_benchmark(
        self,
        model_name: str,
        prompt: str,
        width: int = _DEFAULT_IMAGE_WIDTH,
        height: int = _DEFAULT_IMAGE_HEIGHT,
        steps: int = _DEFAULT_IMAGE_STEPS,
        model: Optional[nn.Module] = None,
    ) -> BenchmarkResult:
        """Benchmark image generation.

        Args:
            model_name: Logical name of the model under test.
            prompt: Input prompt.
            width: Output image width.
            height: Output image height.
            steps: Number of diffusion steps.
            model: Optional real model.  When ``None`` a synthetic
                convolution workload is timed.

        Returns:
            A :class:`BenchmarkResult`.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}.")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}.")

        def _workload() -> Any:
            if model is not None:
                return model(prompt, width=width, height=height, steps=steps)
            # Synthetic workload: simulate diffusion denoising steps.
            latent = torch.randn(4, height // 8, width // 8, device=self._device)
            conv = nn.Conv2d(4, 4, 3, padding=1, device=self._device)
            for _ in range(steps):
                latent = conv(latent)
            return latent

        return self._run_workload(
            workload=_workload,
            optimizations=[f"image:{model_name}", f"{width}x{height}", f"steps={steps}"],
            batch_size=1,
        )

    # ------------------------------------------------------------------
    # Video benchmark
    # ------------------------------------------------------------------
    def run_video_benchmark(
        self,
        model_name: str,
        prompt: str,
        num_frames: int = _DEFAULT_VIDEO_FRAMES,
        width: int = _DEFAULT_IMAGE_WIDTH,
        height: int = _DEFAULT_IMAGE_HEIGHT,
        steps: int = _DEFAULT_VIDEO_STEPS,
        model: Optional[nn.Module] = None,
    ) -> BenchmarkResult:
        """Benchmark video generation.

        Args:
            model_name: Logical name of the model under test.
            prompt: Input prompt.
            num_frames: Number of frames to generate.
            width: Frame width.
            height: Frame height.
            steps: Number of diffusion steps.
            model: Optional real model.  When ``None`` a synthetic 3-D
                convolution workload is timed.

        Returns:
            A :class:`BenchmarkResult`.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}.")
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}.")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}.")

        def _workload() -> Any:
            if model is not None:
                return model(
                    prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    steps=steps,
                )
            # Synthetic workload: simulate temporal diffusion.
            latent = torch.randn(
                num_frames, 4, height // 8, width // 8, device=self._device
            )
            conv = nn.Conv3d(4, 4, 3, padding=1, device=self._device)
            for _ in range(steps):
                latent = conv(latent.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            return latent

        return self._run_workload(
            workload=_workload,
            optimizations=[
                f"video:{model_name}",
                f"frames={num_frames}",
                f"{width}x{height}",
                f"steps={steps}",
            ],
            batch_size=num_frames,
        )

    # ------------------------------------------------------------------
    # Subtitle benchmark
    # ------------------------------------------------------------------
    def run_subtitle_benchmark(
        self,
        audio_path: str,
        method: str = "whisper",
        model: Optional[nn.Module] = None,
    ) -> BenchmarkResult:
        """Benchmark audio-to-subtitle transcription.

        Args:
            audio_path: Path to the audio file.
            method: Transcription backend (``"whisper"``, ``"wav2vec"``,
                ``"ctc"``).
            model: Optional real model.  When ``None`` a synthetic
                waveform workload is timed.

        Returns:
            A :class:`BenchmarkResult`.
        """
        method_lower = method.lower()
        if method_lower not in _SUBTITLE_METHODS:
            raise ValueError(
                f"method must be one of {_SUBTITLE_METHODS}, got {method!r}."
            )

        def _workload() -> Any:
            if model is not None and hasattr(model, "transcribe"):
                return model.transcribe(audio_path)
            # Synthetic workload: simulate a conv over a waveform.
            waveform = torch.randn(1, 16000 * 5, device=self._device)
            conv = nn.Conv1d(1, 1, 7, padding=3, device=self._device)
            return conv(waveform)

        return self._run_workload(
            workload=_workload,
            optimizations=[f"subtitle:{method_lower}", f"audio={audio_path}"],
            batch_size=1,
        )

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run_all(self) -> dict[str, BenchmarkResult]:
        """Run every default benchmark and return a name-keyed dict.

        Returns:
            A dictionary mapping benchmark names to their results.
        """
        results: dict[str, BenchmarkResult] = {}
        results["text"] = self.run_text_benchmark(
            "default", "Hello, world.", max_tokens=16
        )
        results["image"] = self.run_image_benchmark(
            "default", "a cat", steps=_DEFAULT_IMAGE_STEPS
        )
        results["video"] = self.run_video_benchmark(
            "default", "a cat playing", num_frames=_DEFAULT_VIDEO_FRAMES
        )
        results["subtitle"] = self.run_subtitle_benchmark(
            "dummy.wav", method="whisper"
        )
        return results

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def compare(
        self,
        before: dict[str, BenchmarkResult],
        after: dict[str, BenchmarkResult],
    ) -> ComparisonReport:
        """Compare two result sets and produce a :class:`ComparisonReport`.

        A metric is an *improvement* when latency dropped by more than
        :data:`_IMPROVEMENT_THRESHOLD` (5 %) and a *regression* when it
        rose by more than :data:`_REGRESSION_THRESHOLD` (5 %).

        Args:
            before: Baseline results.
            after: Candidate results.

        Returns:
            A :class:`ComparisonReport`.
        """
        improvements: dict[str, float] = {}
        regressions: dict[str, float] = {}

        for name, after_res in after.items():
            before_res = before.get(name)
            if before_res is None:
                continue
            delta = self._relative_change(before_res, after_res)
            if delta <= -_IMPROVEMENT_THRESHOLD:
                improvements[name] = -delta
            elif delta >= _REGRESSION_THRESHOLD:
                regressions[name] = delta

        summary = self._build_summary(improvements, regressions)
        return ComparisonReport(
            improvements=improvements,
            regressions=regressions,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_workload(
        self,
        workload: Any,
        optimizations: list[str],
        batch_size: int,
    ) -> BenchmarkResult:
        """Time ``workload`` with warm-up and peak-memory tracking."""
        peak_mem = 0.0
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)

        # Warm-up.
        for _ in range(self._warmup):
            workload()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        # Timed runs.
        start = time.perf_counter()
        for _ in range(self._runs):
            workload()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elapsed = time.perf_counter() - start

        if self._device.type == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(self._device)
            peak_mem = peak_bytes / _BYTES_PER_GIB

        latency_ms = (elapsed / self._runs) * 1000.0
        throughput = (batch_size * self._runs) / elapsed if elapsed > 0 else 0.0

        return BenchmarkResult(
            latency_ms=latency_ms,
            throughput=throughput,
            peak_memory_gb=peak_mem,
            optimizations=optimizations,
        )

    @staticmethod
    def _relative_change(
        before: BenchmarkResult,
        after: BenchmarkResult,
    ) -> float:
        """Return the relative latency change (positive = slower)."""
        if before.latency_ms <= 0:
            return 0.0
        return (after.latency_ms - before.latency_ms) / before.latency_ms

    @staticmethod
    def _build_summary(
        improvements: dict[str, float],
        regressions: dict[str, float],
    ) -> str:
        """Build a human-readable comparison summary."""
        parts: list[str] = []
        if improvements:
            parts.append(
                "improvements: "
                + ", ".join(f"{k} +{v:.1%}" for k, v in improvements.items())
            )
        if regressions:
            parts.append(
                "regressions: "
                + ", ".join(f"{k} -{v:.1%}" for k, v in regressions.items())
            )
        if not parts:
            return "No significant changes detected."
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"BenchmarkSuite(device={self._device}, "
            f"warmup={self._warmup}, runs={self._runs})"
        )
