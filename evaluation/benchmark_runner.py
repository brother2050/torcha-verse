"""Standardised benchmark runner for TorchaVerse.

This module provides :class:`BenchmarkRunner`, a unified harness for
running standardised evaluation benchmarks across all framework
modalities.  It follows the design philosophy of OpenCompass and
lm-evaluation-harness:

* **Task-based** -- each benchmark is a self-contained task with a
  ``run`` method that returns a metrics dictionary.
* **Configurable** -- tasks, datasets, and evaluation parameters are
  controlled via a configuration dictionary.
* **Reproducible** -- results include timestamps, model names, and
  environment metadata.
* **Extensible** -- new tasks can be registered via the
  :meth:`register_task` class method.

Supported built-in tasks:

* ``text_generation`` -- text generation quality (BLEU, ROUGE, etc.).
* ``image_generation`` -- image generation quality (FID, IS, CLIP).
* ``rag_accuracy`` -- RAG retrieval and answer accuracy.
* ``agent_success_rate`` -- agent task completion rate.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from infrastructure.config_center import ConfigCenter
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

from .image_eval import ImageEvaluator
from .text_eval import TextEvaluator

__all__ = ["BenchmarkRunner", "BenchmarkTask", "BenchmarkResult"]

logger = get_logger("BenchmarkRunner")


# ===========================================================================
# Data structures
# ===========================================================================
class BenchmarkResult:
    """Container for a single benchmark task result.

    Attributes:
        task_name: Name of the benchmark task.
        model_name: Name of the model evaluated.
        metrics: Dictionary of metric name -> value.
        metadata: Additional metadata (duration, device, etc.).
        timestamp: Unix timestamp of when the result was recorded.
        result_id: Unique identifier for this result.
    """

    def __init__(
        self,
        task_name: str,
        model_name: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_name: str = task_name
        self.model_name: str = model_name
        self.metrics: Dict[str, Any] = metrics
        self.metadata: Dict[str, Any] = metadata or {}
        self.timestamp: float = time.time()
        self.result_id: str = f"bench-{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "result_id": self.result_id,
            "task_name": self.task_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Deserialise from a dictionary."""
        result = cls(
            task_name=data["task_name"],
            model_name=data["model_name"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )
        result.timestamp = data.get("timestamp", time.time())
        result.result_id = data.get("result_id", result.result_id)
        return result

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(task={self.task_name!r}, "
            f"model={self.model_name!r}, metrics={self.metrics})"
        )


# ===========================================================================
# Base task class
# ===========================================================================
class BenchmarkTask:
    """Abstract base class for benchmark tasks.

    Subclasses must implement :meth:`run`.  The ``name`` class attribute
    uniquely identifies the task for registration and configuration.
    """

    name: str = "base"
    description: str = "Base benchmark task."
    modality: str = "general"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the task.

        Args:
            config: Optional configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}

    def run(self, runner: "BenchmarkRunner") -> Dict[str, Any]:
        """Run the benchmark task.

        Args:
            runner: The :class:`BenchmarkRunner` that invoked this task,
                providing access to engines and evaluators.

        Returns:
            A dictionary of metric name -> value.

        Raises:
            NotImplementedError: If the subclass does not override this.
        """
        raise NotImplementedError(
            f"Task '{self.name}' must implement run()."
        )


# ===========================================================================
# Built-in tasks
# ===========================================================================
class TextGenerationTask(BenchmarkTask):
    """Benchmark task for text generation quality.

    Evaluates a text model on a set of prompts and references using
    BLEU, ROUGE, accuracy, diversity, and toxicity metrics.
    """

    name = "text_generation"
    description = "Evaluate text generation quality (BLEU, ROUGE, diversity)."
    modality = "text"

    def run(self, runner: "BenchmarkRunner") -> Dict[str, Any]:
        """Run the text generation benchmark.

        Returns:
            A dictionary of text quality metrics.
        """
        cfg = self.config
        prompts: Sequence[str] = cfg.get(
            "prompts",
            [
                "Explain machine learning in simple terms.",
                "Write a short poem about the sea.",
                "What are the benefits of exercise?",
                "Describe the process of photosynthesis.",
                "Summarise the plot of Romeo and Juliet.",
            ],
        )
        references: Optional[Sequence[str]] = cfg.get("references")
        max_tokens: int = cfg.get("max_tokens", 128)
        temperature: float = cfg.get("temperature", 0.7)

        engine = runner.get_text_engine()
        evaluator = TextEvaluator()

        predictions: List[str] = []
        for prompt in prompts:
            try:
                output = engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                predictions.append(output)
            except Exception as exc:
                logger.warning("Generation failed for prompt '%s': %s", prompt, exc)
                predictions.append("")

        results = evaluator.evaluate_all(predictions, references)
        results["num_prompts"] = len(prompts)
        results["avg_length"] = (
            sum(len(p) for p in predictions) / max(len(predictions), 1)
        )
        return results


class ImageGenerationTask(BenchmarkTask):
    """Benchmark task for image generation quality.

    Evaluates an image model on a set of prompts using FID, Inception
    Score, and CLIP Score.
    """

    name = "image_generation"
    description = "Evaluate image generation quality (FID, IS, CLIP)."
    modality = "image"

    def run(self, runner: "BenchmarkRunner") -> Dict[str, Any]:
        """Run the image generation benchmark.

        Returns:
            A dictionary of image quality metrics.
        """
        cfg = self.config
        prompts: Sequence[str] = cfg.get(
            "prompts",
            [
                "a beautiful sunset over mountains",
                "a futuristic city at night",
                "a cute cat sitting on a windowsill",
                "an abstract painting with blue and gold",
                "a serene forest with a flowing river",
            ],
        )
        width: int = cfg.get("width", 256)
        height: int = cfg.get("height", 256)
        steps: int = cfg.get("steps", 20)
        real_images: Optional[Sequence[Any]] = cfg.get("real_images")

        engine = runner.get_image_engine()
        evaluator = ImageEvaluator()

        generated_images: List[Any] = []
        for prompt in prompts:
            try:
                image = engine.txt2img(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps,
                )
                generated_images.append(image)
            except Exception as exc:
                logger.warning("Image generation failed for '%s': %s", prompt, exc)

        if not generated_images:
            return {"error": "No images generated", "num_prompts": len(prompts)}

        results = evaluator.evaluate_all(
            real_images=real_images,
            generated_images=generated_images,
            prompts=prompts,
        )
        results["num_images"] = len(generated_images)
        return results


class RAGAccuracyTask(BenchmarkTask):
    """Benchmark task for RAG retrieval and answer accuracy.

    Evaluates a RAG engine on a set of question-answer pairs, measuring
    retrieval precision, answer accuracy, and average confidence.
    """

    name = "rag_accuracy"
    description = "Evaluate RAG retrieval and answer accuracy."
    modality = "rag"

    def run(self, runner: "BenchmarkRunner") -> Dict[str, Any]:
        """Run the RAG accuracy benchmark.

        Returns:
            A dictionary of RAG metrics.
        """
        cfg = self.config
        qa_pairs: Sequence[Dict[str, str]] = cfg.get(
            "qa_pairs",
            [
                {"question": "What is machine learning?", "answer": "machine learning"},
                {"question": "What is deep learning?", "answer": "neural networks"},
                {"question": "What is NLP?", "answer": "natural language"},
            ],
        )
        top_k: int = cfg.get("top_k", 5)

        engine = runner.get_rag_engine()
        evaluator = TextEvaluator()

        predictions: List[str] = []
        references: List[str] = []
        confidences: List[float] = []
        retrieval_hits: List[bool] = []

        for qa in qa_pairs:
            question = qa["question"]
            reference = qa["answer"]

            try:
                answer, sources = engine.query(question, top_k=top_k)
                predictions.append(answer.text)
                references.append(reference)
                confidences.append(answer.confidence)
                retrieval_hits.append(len(sources.chunks) > 0)
            except Exception as exc:
                logger.warning("RAG query failed: %s", exc)
                predictions.append("")
                references.append(reference)
                confidences.append(0.0)
                retrieval_hits.append(False)

        accuracy = evaluator.evaluate_accuracy(predictions, references)
        bleu = evaluator.evaluate_bleu(predictions, references)

        results: Dict[str, Any] = {
            "exact_match": accuracy["exact_match"],
            "normalized_match": accuracy["normalized_match"],
            "bleu": bleu,
            "avg_confidence": sum(confidences) / max(len(confidences), 1),
            "retrieval_hit_rate": sum(retrieval_hits) / max(len(retrieval_hits), 1),
            "num_questions": len(qa_pairs),
        }
        return results


class AgentSuccessRateTask(BenchmarkTask):
    """Benchmark task for agent task completion rate.

    Evaluates an agent engine on a set of tasks, measuring the success
    rate, average steps, and average execution time.
    """

    name = "agent_success_rate"
    description = "Evaluate agent task completion rate."
    modality = "agent"

    def run(self, runner: "BenchmarkRunner") -> Dict[str, Any]:
        """Run the agent success rate benchmark.

        Returns:
            A dictionary of agent metrics.
        """
        cfg = self.config
        tasks: Sequence[Dict[str, Any]] = cfg.get(
            "tasks",
            [
                {
                    "task": "Calculate 15 * 23 using the calculator tool.",
                    "success_criteria": "345",
                },
                {
                    "task": "Search the web for the capital of France.",
                    "success_criteria": "Paris",
                },
                {
                    "task": "List the files in the current directory.",
                    "success_criteria": "",
                },
            ],
        )
        max_steps: int = cfg.get("max_steps", 10)

        engine = runner.get_agent_engine()

        successes: List[bool] = []
        step_counts: List[int] = []
        durations: List[float] = []

        for task_spec in tasks:
            task = task_spec["task"]
            criteria = task_spec.get("success_criteria", "")

            start = time.time()
            try:
                result = engine.run(task, max_steps=max_steps)
                duration = time.time() - start

                output = result.output.lower()
                success = (
                    criteria.lower() in output if criteria else True
                )
                # Also consider it successful if it didn't truncate.
                if not success and not criteria:
                    success = not result.metadata.get("truncated", False)

                successes.append(success)
                step_counts.append(len(result.steps))
                durations.append(duration)
            except Exception as exc:
                logger.warning("Agent task failed: %s", exc)
                successes.append(False)
                step_counts.append(0)
                durations.append(time.time() - start)

        results: Dict[str, Any] = {
            "success_rate": sum(successes) / max(len(successes), 1),
            "avg_steps": sum(step_counts) / max(len(step_counts), 1),
            "avg_duration_s": sum(durations) / max(len(durations), 1),
            "num_tasks": len(tasks),
        }
        return results


# ===========================================================================
# Task registry
# ===========================================================================
class _TaskRegistry:
    """Registry mapping task names to task classes."""

    _tasks: Dict[str, Type[BenchmarkTask]] = {}

    @classmethod
    def register(cls, task_class: Type[BenchmarkTask]) -> Type[BenchmarkTask]:
        """Register a task class.

        Can be used as a decorator::

            @_TaskRegistry.register
            class MyTask(BenchmarkTask):
                name = "my_task"
                ...
        """
        cls._tasks[task_class.name] = task_class
        return task_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BenchmarkTask]]:
        """Look up a task class by name."""
        return cls._tasks.get(name)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """Return all registered task names."""
        return sorted(cls._tasks.keys())


# Register built-in tasks.
_TaskRegistry.register(TextGenerationTask)
_TaskRegistry.register(ImageGenerationTask)
_TaskRegistry.register(RAGAccuracyTask)
_TaskRegistry.register(AgentSuccessRateTask)


# ===========================================================================
# BenchmarkRunner
# ===========================================================================
class BenchmarkRunner:
    """Run standardised benchmark tests across all framework modalities.

    The runner manages engine instantiation, task execution, result
    collection, and persistence.  It follows the OpenCompass /
    lm-evaluation-harness pattern of task-based evaluation with a
    configurable pipeline.

    Example::

        runner = BenchmarkRunner("llama-8b", tasks=["text_generation"])
        results = runner.run_all()
        print(runner.format_results(results))
        runner.save_results(results, "results.json")
    """

    def __init__(
        self,
        model_name: str = "default",
        tasks: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the benchmark runner.

        Args:
            model_name: Name of the model to evaluate.
            tasks: List of task names to run.  Defaults to all registered
                tasks.
            config: Optional configuration dictionary.  Keys:

                * ``text`` -- config for text tasks.
                * ``image`` -- config for image tasks.
                * ``rag`` -- config for RAG tasks.
                * ``agent`` -- config for agent tasks.
                * ``output_dir`` -- directory for saving results.
        """
        self.model_name: str = model_name
        self.tasks: List[str] = tasks or _TaskRegistry.list_tasks()
        self.config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigCenter = ConfigCenter()
        self._device_manager: DeviceManager = DeviceManager()
        self._logger = logger

        # Lazy engine instances.
        self._text_engine: Any = None
        self._image_engine: Any = None
        self._rag_engine: Any = None
        self._agent_engine: Any = None

        # Results storage.
        self._results: List[BenchmarkResult] = []

        self._logger.info(
            "BenchmarkRunner initialised for model '%s' with tasks: %s",
            model_name,
            self.tasks,
        )

    # ------------------------------------------------------------------
    # Engine accessors (lazy)
    # ------------------------------------------------------------------
    def get_text_engine(self) -> Any:
        """Return a :class:`TextEngine` (lazily created)."""
        if self._text_engine is None:
            # engines removed

            self._text_engine = TextEngine(self.model_name)
        return self._text_engine

    def get_image_engine(self) -> Any:
        """Return an :class:`ImageEngine` (lazily created)."""
        if self._image_engine is None:
            # engines removed

            self._image_engine = ImageEngine(self.model_name)
        return self._image_engine

    def get_rag_engine(self) -> Any:
        """Return a :class:`RAGEngine` (lazily created)."""
        if self._rag_engine is None:
            # engines removed

            self._rag_engine = RAGEngine()
        return self._rag_engine

    def get_agent_engine(self) -> Any:
        """Return an :class:`AgentEngine` (lazily created)."""
        if self._agent_engine is None:
            # engines removed

            self._agent_engine = AgentEngine()
        return self._agent_engine

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------
    @classmethod
    def register_task(cls, task_class: Type[BenchmarkTask]) -> Type[BenchmarkTask]:
        """Register a custom benchmark task.

        Args:
            task_class: A subclass of :class:`BenchmarkTask` with a
                unique ``name`` attribute.

        Returns:
            The registered task class.
        """
        return _TaskRegistry.register(task_class)

    @classmethod
    def list_available_tasks(cls) -> List[str]:
        """Return all registered task names.

        Returns:
            A sorted list of task names.
        """
        return _TaskRegistry.list_tasks()

    # ------------------------------------------------------------------
    # Run a single benchmark
    # ------------------------------------------------------------------
    def run_benchmark(self, task_name: str) -> BenchmarkResult:
        """Run a single benchmark task.

        Args:
            task_name: The name of the task to run.

        Returns:
            A :class:`BenchmarkResult` with the task's metrics.

        Raises:
            ValueError: If ``task_name`` is not a registered task.
        """
        task_class = _TaskRegistry.get(task_name)
        if task_class is None:
            raise ValueError(
                f"Unknown benchmark task: '{task_name}'. "
                f"Available: {_TaskRegistry.list_tasks()}"
            )

        # Get task-specific config.
        task_config = self.config.get(task_class.modality, {})
        task_config.update(self.config.get(task_name, {}))

        task = task_class(config=task_config)

        self._logger.info("Running benchmark task: %s", task_name)
        start_time = time.time()

        try:
            metrics = task.run(self)
            error = None
        except Exception as exc:
            self._logger.error("Task '%s' failed: %s", task_name, exc, exc_info=True)
            metrics = {"error": str(exc)}
            error = str(exc)

        duration = time.time() - start_time

        result = BenchmarkResult(
            task_name=task_name,
            model_name=self.model_name,
            metrics=metrics,
            metadata={
                "duration_s": duration,
                "device": str(self._device_manager.get_device()),
                "config": task_config,
                "error": error,
            },
        )

        self._results.append(result)
        self._logger.info(
            "Task '%s' completed in %.2fs", task_name, duration
        )
        return result

    # ------------------------------------------------------------------
    # Run all benchmarks
    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        """Run all configured benchmark tasks.

        Returns:
            A dictionary with ``model``, ``tasks``, ``results``, and
            ``summary`` keys.
        """
        self._logger.info(
            "Running all benchmarks for model '%s'...", self.model_name
        )

        overall_start = time.time()
        results: List[BenchmarkResult] = []

        for task_name in self.tasks:
            try:
                result = self.run_benchmark(task_name)
                results.append(result)
            except Exception as exc:
                self._logger.error("Failed to run task '%s': %s", task_name, exc)
                results.append(
                    BenchmarkResult(
                        task_name=task_name,
                        model_name=self.model_name,
                        metrics={"error": str(exc)},
                        metadata={"duration_s": 0.0},
                    )
                )

        overall_duration = time.time() - overall_start

        # Build summary.
        summary = self._build_summary(results, overall_duration)

        return {
            "model": self.model_name,
            "tasks": self.tasks,
            "results": [r.to_dict() for r in results],
            "summary": summary,
            "total_duration_s": overall_duration,
            "timestamp": time.time(),
        }

    def _build_summary(
        self,
        results: List[BenchmarkResult],
        duration: float,
    ) -> Dict[str, Any]:
        """Build a summary dictionary from results.

        Args:
            results: List of benchmark results.
            duration: Total execution time.

        Returns:
            A summary dictionary.
        """
        num_total = len(results)
        num_success = sum(
            1 for r in results if "error" not in r.metrics
        )
        num_errors = num_total - num_success

        # Extract key metrics for quick reference.
        key_metrics: Dict[str, Any] = {}
        for r in results:
            for metric_name, value in r.metrics.items():
                if metric_name not in ("error",) and isinstance(value, (int, float)):
                    key_metrics[f"{r.task_name}.{metric_name}"] = value

        return {
            "total_tasks": num_total,
            "successful": num_success,
            "failed": num_errors,
            "total_duration_s": duration,
            "key_metrics": key_metrics,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save benchmark results to a JSON file.

        Args:
            results: The results dictionary (from :meth:`run_all`).
            path: Output file path.
        """
        self._logger.info("Saving results to %s", path)

        # Add environment metadata.
        output = {
            "framework": "TorchaVerse",
            "version": "0.3.1",
            "device": str(self._device_manager.get_device()),
            "saved_at": time.time(),
            **results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

        self._logger.info("Results saved to %s", path)

    @staticmethod
    def load_results(path: str) -> Dict[str, Any]:
        """Load benchmark results from a JSON file.

        Args:
            path: Path to the results JSON file.

        Returns:
            The results dictionary.
        """
        logger.info("Loading results from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as a rich-text table.

        Args:
            results: The results dictionary (from :meth:`run_all`).

        Returns:
            A formatted string representation of the results table.
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console(record=True, width=100)
        except ImportError:
            # Fallback to plain text.
            return self._format_results_plain(results)

        # Header.
        console.print(
            Panel(
                f"[bold cyan]TorchaVerse Benchmark Report[/bold cyan]\n"
                f"Model: [bold]{results.get('model', 'unknown')}[/bold]\n"
                f"Duration: {results.get('total_duration_s', 0):.2f}s\n"
                f"Tasks: {', '.join(results.get('tasks', []))}",
                border_style="cyan",
            )
        )

        # Main results table.
        table = Table(title="Benchmark Results", border_style="blue")
        table.add_column("Task", style="cyan", width=20)
        table.add_column("Metric", style="white", width=25)
        table.add_column("Value", style="green", justify="right", width=15)
        table.add_column("Duration (s)", style="dim", justify="right", width=12)

        for result_dict in results.get("results", []):
            task_name = result_dict.get("task_name", "unknown")
            duration = result_dict.get("metadata", {}).get("duration_s", 0.0)
            metrics = result_dict.get("metrics", {})

            if not metrics:
                table.add_row(task_name, "(no metrics)", "-", f"{duration:.2f}")
                continue

            first = True
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                elif isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False)[:30]
                else:
                    value_str = str(value)

                table.add_row(
                    task_name if first else "",
                    metric_name,
                    value_str,
                    f"{duration:.2f}" if first else "",
                )
                first = False

        console.print(table)

        # Summary.
        summary = results.get("summary", {})
        if summary:
            summary_table = Table(title="Summary", border_style="green")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white", justify="right")

            summary_table.add_row("Total tasks", str(summary.get("total_tasks", 0)))
            summary_table.add_row("Successful", str(summary.get("successful", 0)))
            summary_table.add_row("Failed", str(summary.get("failed", 0)))
            summary_table.add_row(
                "Total duration (s)",
                f"{summary.get('total_duration_s', 0):.2f}",
            )

            console.print(summary_table)

            # Key metrics.
            key_metrics = summary.get("key_metrics", {})
            if key_metrics:
                km_table = Table(title="Key Metrics", border_style="yellow")
                km_table.add_column("Metric", style="cyan")
                km_table.add_column("Value", style="green", justify="right")
                for k, v in sorted(key_metrics.items()):
                    km_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
                console.print(km_table)

        return console.export_text()

    def _format_results_plain(self, results: Dict[str, Any]) -> str:
        """Format results as plain text (fallback without rich)."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("TorchaVerse Benchmark Report")
        lines.append(f"Model: {results.get('model', 'unknown')}")
        lines.append(f"Duration: {results.get('total_duration_s', 0):.2f}s")
        lines.append("=" * 60)
        lines.append("")

        for result_dict in results.get("results", []):
            task_name = result_dict.get("task_name", "unknown")
            duration = result_dict.get("metadata", {}).get("duration_s", 0.0)
            metrics = result_dict.get("metrics", {})

            lines.append(f"[{task_name}] ({duration:.2f}s)")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {metric_name}: {value:.4f}")
                else:
                    lines.append(f"  {metric_name}: {value}")
            lines.append("")

        summary = results.get("summary", {})
        lines.append("-" * 60)
        lines.append("Summary:")
        lines.append(f"  Total tasks: {summary.get('total_tasks', 0)}")
        lines.append(f"  Successful: {summary.get('successful', 0)}")
        lines.append(f"  Failed: {summary.get('failed', 0)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # OpenCompass / lm-evaluation-harness style pipeline interface
    # ------------------------------------------------------------------
    def configure_pipeline(
        self,
        task_configs: Dict[str, Dict[str, Any]],
    ) -> None:
        """Configure the evaluation pipeline (OpenCompass-style).

        Allows fine-grained per-task configuration, similar to how
        OpenCompass defines task configs in a YAML/JSON file.

        Args:
            task_configs: A dictionary mapping task names to their
                configuration dictionaries.
        """
        for task_name, task_cfg in task_configs.items():
            if task_name not in self.tasks:
                self.tasks.append(task_name)
            self.config[task_name] = task_cfg

        self._logger.info(
            "Pipeline configured with %d task configs", len(task_configs)
        )

    def run_pipeline(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full evaluation pipeline.

        This is the main entry point for running a configured benchmark
        suite, analogous to ``lm_eval --model ... --tasks ...`` in
        lm-evaluation-harness.

        Args:
            output_path: Optional path to save results as JSON.

        Returns:
            The full results dictionary.
        """
        self._logger.info("Starting evaluation pipeline...")

        results = self.run_all()

        formatted = self.format_results(results)
        print(formatted)

        if output_path:
            self.save_results(results, output_path)

        self._logger.info("Evaluation pipeline complete.")
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def results(self) -> List[BenchmarkResult]:
        """Return all collected results."""
        return list(self._results)

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunner(model={self.model_name!r}, "
            f"tasks={self.tasks})"
        )
