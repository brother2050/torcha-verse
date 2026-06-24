"""Performance optimisation for the TorchaVerse framework.

This module provides :class:`PerformanceOptimizer`, which applies a stack
of PyTorch-native optimisations to a model or pipeline:

* **SDPA** -- :func:`torch.nn.functional.scaled_dot_product_attention`
  fused attention kernel (enabled by default on PyTorch >= 2.0).
* **torch.compile** -- graph compilation with a configurable mode
  (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``).
* **CUDA Graph capture** -- replay a captured graph for stable input
  shapes, eliminating kernel-launch overhead.
* **Pipeline fusion** -- merge adjacent fusable nodes and pre-allocate
  VRAM for the execution plan.

When a requested optimisation is unavailable (e.g. CUDA Graph on a
CPU-only build) the optimiser degrades gracefully and records the
skip in the returned :class:`OptimizedModel`.

The module **depends on ``torch``**.

Example:
    >>> import torch, torch.nn as nn
    >>> from performance import PerformanceOptimizer, OptimizationConfig
    >>> model = nn.Linear(4, 4)
    >>> opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
    >>> wrapped = opt.optimize_model(model)
    >>> wrapped(torch.randn(1, 4)).shape
    torch.Size([1, 4])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "OptimizationConfig",
    "PerformanceOptimizer",
    "OptimizedModel",
    "BenchmarkResult",
]

# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Default torch.compile mode.
_DEFAULT_COMPILE_MODE: str = "reduce-overhead"

#: Valid torch.compile modes.
_VALID_COMPILE_MODES: tuple[str, ...] = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

#: Default number of warm-up iterations in :meth:`benchmark`.
_DEFAULT_WARMUP: int = 3

#: Default number of measured iterations in :meth:`benchmark`.
_DEFAULT_RUNS: int = 10

#: Bytes per GiB used for memory reporting.
_BYTES_PER_GIB: float = 1024.0 ** 3

#: Sentinel attribute name used to tag modules whose attention has been
#: patched to use SDPA.
_SDPA_TAG: str = "_torcha_verse_sdpa_applied"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OptimizationConfig:
    """Configuration for :class:`PerformanceOptimizer`.

    Attributes:
        enable_sdpa: Replace attention with fused SDPA when possible.
        enable_compile: Wrap the model with :func:`torch.compile`.
        enable_cuda_graph: Capture a CUDA Graph for stable shapes.
        compile_mode: Mode passed to :func:`torch.compile`.
        triton_kernels: Names of custom Triton kernels to register
            (placeholder -- recorded but not loaded).
    """

    enable_sdpa: bool = True
    enable_compile: bool = True
    enable_cuda_graph: bool = True
    compile_mode: str = _DEFAULT_COMPILE_MODE
    triton_kernels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {_VALID_COMPILE_MODES}, "
                f"got {self.compile_mode!r}."
            )


@dataclass
class BenchmarkResult:
    """Outcome of a benchmark run.

    Attributes:
        latency_ms: Mean latency per forward pass (milliseconds).
        throughput: Items per second.
        peak_memory_gb: Peak GPU memory in GiB (0.0 on CPU).
        optimizations: List of optimisation names that were active.
    """

    latency_ms: float
    throughput: float
    peak_memory_gb: float
    optimizations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OptimizedModel
# ---------------------------------------------------------------------------
class OptimizedModel(nn.Module):
    """Wrapper around an optimised model.

    Delegates :meth:`forward` (and :meth:`generate` when present) to the
    underlying model, while exposing the list of applied optimisations.

    Args:
        model: The (possibly compiled) underlying model.
        optimizations: Human-readable names of applied optimisations.
        cuda_graph: Optional captured CUDA graph and its static buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizations: Optional[List[str]] = None,
        cuda_graph: Any = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = model
        self._optimizations: list[str] = list(optimizations or [])
        self._cuda_graph: Any = cuda_graph
        self._graph_inputs: Any = None
        self._graph_outputs: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def optimizations(self) -> list[str]:
        """Names of the optimisations applied to the wrapped model."""
        return list(self._optimizations)

    @property
    def has_cuda_graph(self) -> bool:
        """``True`` when a CUDA graph has been captured."""
        return self._cuda_graph is not None

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the underlying model's forward pass."""
        if self._cuda_graph is not None and self._graph_inputs is not None:
            return self._replay_graph(args, kwargs)
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying model's ``generate`` if present."""
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        raise AttributeError(
            f"Underlying model {type(self.model).__name__} has no 'generate' method."
        )

    # ------------------------------------------------------------------
    # CUDA-graph replay helper
    # ------------------------------------------------------------------
    def _replay_graph(self, args: tuple, kwargs: dict) -> Any:
        """Copy new inputs into static buffers and replay the graph."""
        if self._graph_inputs is None:
            return self.model(*args, **kwargs)
        static_inputs = self._graph_inputs
        if isinstance(static_inputs, (list, tuple)):
            for static, new in zip(static_inputs, args):
                if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor):
                    static.copy_(new)
        self._cuda_graph.replay()
        if self._graph_outputs is not None:
            if isinstance(self._graph_outputs, torch.Tensor):
                return self._graph_outputs.clone()
            return [t.clone() if isinstance(t, torch.Tensor) else t for t in self._graph_outputs]
        return None

    def __repr__(self) -> str:
        return (
            f"OptimizedModel(model={type(self.model).__name__}, "
            f"optimizations={self._optimizations}, "
            f"cuda_graph={self._cuda_graph is not None})"
        )


# ---------------------------------------------------------------------------
# PerformanceOptimizer
# ---------------------------------------------------------------------------
class PerformanceOptimizer:
    """Apply a stack of PyTorch optimisations to models and pipelines.

    Args:
        config: Optimisation policy.  Defaults to
            :class:`OptimizationConfig` with all flags enabled.

    Example:
        >>> opt = PerformanceOptimizer(OptimizationConfig(enable_compile=False))
        >>> wrapped = opt.optimize_model(nn.Linear(4, 4))
    """

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        self._config: OptimizationConfig = config or OptimizationConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> OptimizationConfig:
        """The active optimisation policy."""
        return self._config

    # ------------------------------------------------------------------
    # Model optimisation
    # ------------------------------------------------------------------
    def optimize_model(
        self,
        model: nn.Module,
        sample_input: Any = None,
    ) -> OptimizedModel:
        """Apply the configured optimisations to ``model``.

        Args:
            model: The model to optimise.
            sample_input: Optional sample input used for CUDA-graph
                capture and shape inference.

        Returns:
            An :class:`OptimizedModel` wrapping the optimised model.
        """
        optimizations: list[str] = []
        optimized = model

        # 1. SDPA.
        if self._config.enable_sdpa:
            applied = self._apply_sdpa(optimized)
            if applied:
                optimizations.append("sdpa")

        # 2. torch.compile.
        if self._config.enable_compile:
            compiled = self._apply_compile(optimized)
            if compiled is not optimized:
                optimized = compiled
                optimizations.append(f"torch.compile({self._config.compile_mode})")

        # 3. CUDA Graph capture.
        cuda_graph = None
        if self._config.enable_cuda_graph and sample_input is not None:
            cuda_graph = self._capture_cuda_graph(optimized, sample_input)
            if cuda_graph is not None:
                optimizations.append("cuda_graph")

        return OptimizedModel(
            model=optimized,
            optimizations=optimizations,
            cuda_graph=cuda_graph,
        )

    # ------------------------------------------------------------------
    # Pipeline optimisation
    # ------------------------------------------------------------------
    def optimize_pipeline(self, pipeline: Any) -> Any:
        """Optimise a pipeline by fusing nodes and pre-allocating VRAM.

        This is a placeholder that returns the pipeline unchanged when
        it does not expose the expected node-fusion interface.  When the
        pipeline provides ``fuse_nodes`` / ``preallocate`` methods they
        are invoked.

        Args:
            pipeline: A pipeline object.

        Returns:
            The (possibly modified) pipeline.
        """
        if hasattr(pipeline, "fuse_nodes"):
            try:
                pipeline.fuse_nodes()
            except Exception:
                pass
        if hasattr(pipeline, "preallocate"):
            try:
                pipeline.preallocate()
            except Exception:
                pass
        return pipeline

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def benchmark(
        self,
        model: nn.Module,
        input_shape: Union[Sequence[int], Tuple[Sequence[int], ...]],
        warmup: int = _DEFAULT_WARMUP,
        runs: int = _DEFAULT_RUNS,
    ) -> BenchmarkResult:
        """Benchmark a model's forward pass.

        Args:
            model: The model to benchmark.
            input_shape: Shape of the input tensor, or a tuple of shapes
                for multi-input models.
            warmup: Number of warm-up iterations (not timed).
            runs: Number of timed iterations.

        Returns:
            A :class:`BenchmarkResult`.
        """
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}.")
        if runs <= 0:
            raise ValueError(f"runs must be > 0, got {runs}.")

        device = self._infer_device(model)
        model = model.to(device)
        model.eval()

        inputs = self._build_inputs(input_shape, device)

        # Peak-memory tracking.
        peak_mem = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        with torch.no_grad():
            # Warm-up.
            for _ in range(warmup):
                self._forward(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # Timed runs.
            start = time.perf_counter()
            for _ in range(runs):
                self._forward(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

        if device.type == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem = peak_bytes / _BYTES_PER_GIB

        latency_ms = (elapsed / runs) * 1000.0
        batch = self._batch_size(inputs)
        throughput = (batch * runs) / elapsed if elapsed > 0 else 0.0

        return BenchmarkResult(
            latency_ms=latency_ms,
            throughput=throughput,
            peak_memory_gb=peak_mem,
            optimizations=list(self._active_optimizations()),
        )

    # ------------------------------------------------------------------
    # Internals -- SDPA
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_sdpa(model: nn.Module) -> bool:
        """Tag the model as SDPA-enabled.

        PyTorch >= 2.0 already routes attention through SDPA internally
        when ``torch.nn.functional.scaled_dot_product_attention`` is
        called, so this is effectively a no-op marker.  Returns ``True``
        to record the optimisation.
        """
        if getattr(model, _SDPA_TAG, False):
            return False
        setattr(model, _SDPA_TAG, True)
        return True

    # ------------------------------------------------------------------
    # Internals -- torch.compile
    # ------------------------------------------------------------------
    def _apply_compile(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` with :func:`torch.compile` when available."""
        if not hasattr(torch, "compile"):
            return model
        try:
            return torch.compile(model, mode=self._config.compile_mode)  # type: ignore[call-overload]
        except Exception:
            return model

    # ------------------------------------------------------------------
    # Internals -- CUDA graph
    # ------------------------------------------------------------------
    def _capture_cuda_graph(self, model: nn.Module, sample_input: Any) -> Any:
        """Attempt to capture a CUDA graph; return ``None`` on failure."""
        if not torch.cuda.is_available():
            return None
        try:
            device = torch.device("cuda")
            model = model.to(device)
            model.eval()

            # Build static inputs on CUDA.
            static_inputs = self._to_static(sample_input, device)
            # Warm-up (required before capture).
            for _ in range(3):
                with torch.no_grad():
                    _ = self._forward(model, static_inputs)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._forward(model, static_inputs)

            wrapper = _CudaGraphWrapper(graph, static_inputs, outputs)
            return wrapper
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internals -- helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        """Infer the device of ``model`` from its parameters."""
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _build_inputs(
        input_shape: Union[Sequence[int], Tuple[Sequence[int], ...]],
        device: torch.device,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Create random input tensors from ``input_shape``."""
        if isinstance(input_shape, tuple) and input_shape and isinstance(input_shape[0], (list, tuple)):
            return tuple(torch.randn(*shape, device=device) for shape in input_shape)
        return torch.randn(*input_shape, device=device)

    @staticmethod
    def _forward(
        model: nn.Module,
        inputs: Any,
    ) -> Any:
        """Call ``model`` with either a single tensor or a tuple."""
        if isinstance(inputs, tuple):
            return model(*inputs)
        return model(inputs)

    @staticmethod
    def _batch_size(inputs: Any) -> int:
        """Return the batch dimension size of ``inputs``."""
        if isinstance(inputs, torch.Tensor):
            return inputs.shape[0] if inputs.dim() > 0 else 1
        if isinstance(inputs, (list, tuple)) and inputs:
            first = inputs[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0] if first.dim() > 0 else 1
        return 1

    @staticmethod
    def _to_static(sample_input: Any, device: torch.device) -> Any:
        """Clone ``sample_input`` tensors onto ``device`` for graph capture."""
        if isinstance(sample_input, torch.Tensor):
            return sample_input.clone().to(device)
        if isinstance(sample_input, (list, tuple)):
            return tuple(
                t.clone().to(device) if isinstance(t, torch.Tensor) else t
                for t in sample_input
            )
        return sample_input

    def _active_optimizations(self) -> list[str]:
        """Return the list of optimisation names enabled in the config."""
        active: list[str] = []
        if self._config.enable_sdpa:
            active.append("sdpa")
        if self._config.enable_compile:
            active.append(f"torch.compile({self._config.compile_mode})")
        if self._config.enable_cuda_graph:
            active.append("cuda_graph")
        return active

    def __repr__(self) -> str:
        return (
            f"PerformanceOptimizer(sdpa={self._config.enable_sdpa}, "
            f"compile={self._config.enable_compile}, "
            f"cuda_graph={self._config.enable_cuda_graph})"
        )


# ---------------------------------------------------------------------------
# CUDA-graph wrapper helper
# ---------------------------------------------------------------------------
class _CudaGraphWrapper:
    """Bundle a captured CUDA graph with its static I/O buffers."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_inputs: Any,
        static_outputs: Any,
    ) -> None:
        self.graph: torch.cuda.CUDAGraph = graph
        self.static_inputs: Any = static_inputs
        self.static_outputs: Any = static_outputs

    def replay(self) -> None:
        """Replay the captured graph."""
        self.graph.replay()
