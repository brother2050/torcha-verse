"""Pipeline composer for the TorchaVerse pipeline layer (L5).

This module turns a structural :class:`~pipeline.dag.DAG` into an executable
:class:`Pipeline`.  It is the orchestration heart of the framework: it walks
the graph in dependency order, runs independent nodes concurrently with a
:class:`concurrent.futures.ThreadPoolExecutor`, threads outputs from upstream
nodes into downstream inputs, and reports progress through an optional
callback.

The layer is deliberately *torch-free*: it never imports :mod:`torch`.  Node
executors are resolved lazily through a :class:`NodeContext` -- either from
an explicit per-type callable map or from the :class:`~core.module_bus.ModuleBus`
under the ``node.<node_type>`` namespace.  When no executor is registered for
a node type the pipeline falls back to a *passthrough* that returns the
node's merged inputs as its output, which keeps the orchestration layer
fully exercisable even before the L4 node system is wired up.

Public surface:

* :class:`NodeContext` -- per-run execution context (shared output store,
  executor resolution, metadata).
* :class:`PipelineConfig` -- descriptive metadata for a pipeline.
* :class:`Pipeline` -- the executable pipeline (run / dry_run / validate /
  YAML + dict serialisation, with cancel / pause support).
* :class:`PipelineBuilder` -- a fluent builder producing a :class:`Pipeline`.
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import (
    ALL_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from nodes.base import NodeContext, NodeExecutor
from .dag import DAG, DAGEdge, DAGNode
from .validators import ConnectionValidator

__all__ = [
    "NodeContext",
    "NodeExecutor",
    "PipelineConfig",
    "Pipeline",
    "PipelineBuilder",
]


# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- this layer must not import torch).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("pipeline.composer")


#: Default key used by :meth:`PipelineBuilder.connect` when no output/input
#: key is supplied.
_DEFAULT_OUTPUT_KEY: str = "output"
_DEFAULT_INPUT_KEY: str = "input"

#: Below this many GB of free VRAM the composer emits a warning before
#: executing a node (best-effort; only checked when the budget exposes an
#: ``available()`` method, e.g. a :class:`BudgetTracker`).
_LOW_VRAM_THRESHOLD_GB: float = 1.0
#: Poll interval (seconds) used by the cooperative pause loop.  Kept tiny
#: so that a :meth:`Pipeline.resume` call is observed promptly.
_PAUSE_POLL_INTERVAL_S: float = 0.01

#: Type alias for the progress callback signature.
ProgressCallback = Callable[[int, int, str, str], None]


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Descriptive metadata for a :class:`Pipeline`.

    Attributes:
        name: Human-readable pipeline name.
        description: Free-form description.
        version: Semantic version of the pipeline definition.
        author: Author / owner of the pipeline.
        tags: Free-form tags for discovery and search.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this config to a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Reconstruct a :class:`PipelineConfig` from a serialised dict."""
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0.0"),
            author=d.get("author", ""),
            tags=list(d.get("tags") or []),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    """An executable pipeline backed by a :class:`DAG`.

    A :class:`Pipeline` binds a :class:`PipelineConfig` with a structural
    :class:`DAG` and provides execution, validation, dry-run planning and
    serialisation.  Execution honours the graph's dependency order and runs
    independent nodes concurrently within each parallel layer.

    Run control:

    * :meth:`pause` / :meth:`resume` -- temporarily halt execution between
      layers (cooperative; checked at layer boundaries).
    * :meth:`cancel` -- request early termination; the current layer
      finishes its in-flight nodes and execution stops before the next.

    Args:
        config: The :class:`PipelineConfig` describing the pipeline.
        dag: The :class:`DAG` defining the node graph.
    """

    def __init__(self, config: PipelineConfig, dag: DAG) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig instance.")
        if not isinstance(dag, DAG):
            raise TypeError("dag must be a DAG instance.")
        self._config: PipelineConfig = config
        self._dag: DAG = dag
        self._logger: logging.Logger = _logger
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_max_workers: int = 0

        # Run-control primitives.
        self._cancel_event: threading.Event = threading.Event()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()  # not paused by default
        self._run_lock: threading.RLock = threading.RLock()
        self._running: bool = False

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> PipelineConfig:
        """The pipeline's descriptive config."""
        return self._config

    @property
    def dag(self) -> DAG:
        """The underlying :class:`DAG`."""
        return self._dag

    @property
    def is_running(self) -> bool:
        """``True`` while a run is in progress."""
        return self._running

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        """Request cancellation of the current run.

        Cancellation is cooperative: the currently executing layer is allowed
        to finish its in-flight nodes, after which the run returns the
        results collected so far.
        """
        self._cancel_event.set()
        self._logger.info("Pipeline %r cancellation requested.", self._config.name)

    def pause(self) -> None:
        """Pause execution at the next layer boundary."""
        self._pause_event.clear()
        self._logger.info("Pipeline %r pause requested.", self._config.name)

    def resume(self) -> None:
        """Resume a paused pipeline."""
        self._pause_event.set()
        self._logger.info("Pipeline %r resumed.", self._config.name)

    def _reset_run_state(self) -> None:
        """Reset run-control primitives before a fresh run."""
        self._cancel_event.clear()
        self._pause_event.set()

    def close(self) -> None:
        """Clean up resources (thread pool).

        If the pipeline is currently running this logs a warning but does
        not block -- the executor is shut down without waiting for pending
        tasks.  ``_run_lock`` is an :class:`~threading.RLock` so acquiring
        it here cannot deadlock against :meth:`run` (which only holds it
        briefly to flip the ``_running`` flag).
        """
        with self._run_lock:
            if self._running:
                _logger.warning(
                    "close() called while pipeline is running; "
                    "shutting down executor without waiting."
                )
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
                self._executor_max_workers = 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        ctx: NodeContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline against ``ctx`` and return every node output.

        Nodes are executed in dependency order.  Within each parallel layer
        (see :meth:`DAG.parallel_groups`) independent nodes run concurrently
        on a :class:`ThreadPoolExecutor`.  Upstream outputs are threaded into
        downstream inputs according to the graph's edges.

        Args:
            ctx: The :class:`NodeContext` providing the output store and
                executor resolution.
            progress_callback: Optional callable invoked as
                ``callback(step, total, node_id, status)`` where ``status``
                is one of ``"start"``, ``"done"`` or ``"error"``.

        Returns:
            A mapping of ``node_id -> outputs`` for every node that ran.

        Raises:
            ValueError: If the underlying DAG fails validation.
            RuntimeError: If a node executor raises.
        """
        errors = self.validate()
        if errors:
            raise ValueError(
                "Pipeline DAG is invalid: {}".format("; ".join(errors))
            )

        with self._run_lock:
            if self._running:
                raise RuntimeError("Pipeline is already running.")
            self._running = True
            self._reset_run_state()

        try:
            return self._run(ctx, progress_callback)
        finally:
            with self._run_lock:
                self._running = False

    def _run(
        self,
        ctx: NodeContext,
        progress_callback: Optional[ProgressCallback],
    ) -> Dict[str, Any]:
        """Internal run loop operating on parallel groups."""
        order = self._dag.topological_sort()
        groups = self._dag.parallel_groups()
        total = len(order)
        step = 0
        results: Dict[str, Any] = {}

        # Compute max_workers once (it does not depend on the current group).
        max_workers = max(ctx.max_workers, max((len(g) for g in groups), default=1))

        for group in groups:
            # Cooperative pause: block (with a small spin) until resumed.
            while not self._pause_event.wait(timeout=_PAUSE_POLL_INTERVAL_S):
                if self._cancel_event.is_set():
                    break
            if self._cancel_event.is_set():
                self._logger.info(
                    "Pipeline %r cancelled before layer %d.",
                    self._config.name, step,
                )
                break

            # Run the layer concurrently.
            # 使用可复用线程池
            if self._executor is None or self._executor_max_workers < max_workers:
                if self._executor is not None:
                    self._executor.shutdown(wait=False)
                self._executor = ThreadPoolExecutor(max_workers=max_workers)
                self._executor_max_workers = max_workers
            pool = self._executor
            future_map: Dict[Future, str] = {}
            for node_id in group:
                if progress_callback is not None:
                    progress_callback(step, total, node_id, "start")
                fut = pool.submit(self._run_node, node_id, ctx)
                future_map[fut] = node_id

            for fut in future_map:
                node_id = future_map[fut]
                try:
                    outputs = fut.result()
                except Exception as exc:
                    if progress_callback is not None:
                        progress_callback(step, total, node_id, "error")
                    self._logger.error(
                        "Node %r failed: %s", node_id, exc
                    )
                    # R0-7: 节点失败时保留部分结果。
                    # 等待同层其余 future 全部完成,收集已完成节点的输出,
                    # 写入 ctx 的输出存储,然后抛出 RuntimeError。
                    remaining = [
                        f
                        for f in future_map
                        if f is not fut and not f.done()
                    ]
                    if remaining:
                        wait(remaining, return_when=ALL_COMPLETED)
                    for other_fut, other_id in future_map.items():
                        if other_fut is fut:
                            continue
                        if other_id in results:
                            continue
                        if not other_fut.done():
                            continue
                        try:
                            other_outputs = other_fut.result()
                        except Exception as other_exc:
                            # 该 future 也失败了,跳过(不保留失败结果)。
                            self._logger.debug(
                                "Future %r also failed: %s",
                                other_id, other_exc,
                            )
                            continue
                        results[other_id] = other_outputs
                        ctx.set_output(other_id, other_outputs)
                        step += 1
                        if progress_callback is not None:
                            progress_callback(
                                step, total, other_id, "done"
                            )
                    raise RuntimeError(
                        "Node {!r} failed: {}".format(node_id, exc)
                    ) from exc
                results[node_id] = outputs
                ctx.set_output(node_id, outputs)
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total, node_id, "done")

        return results

    def _run_node(self, node_id: str, ctx: NodeContext) -> Dict[str, Any]:
        """Execute a single node, merging upstream outputs into its inputs."""
        node = self._dag.get_node(node_id)
        inputs: Dict[str, Any] = copy.deepcopy(node.inputs)

        # Thread upstream outputs through the graph's edges.
        for edge in self._dag.get_incoming_edges(node_id):
            if not ctx.has_output(edge.from_node):
                # Upstream did not produce output (e.g. cancelled); skip.
                continue
            value = ctx.get_output(edge.from_node, edge.output_key)
            inputs[edge.input_key] = value

        executor = ctx.resolve_executor(node.node_type)
        if executor is None:
            # Passthrough: return merged inputs so the graph can still flow.
            if ctx.strict_mode:
                raise RuntimeError(
                    "No executor registered for node_type {!r} (strict mode).".format(
                        node.node_type
                    )
                )
            self._logger.warning(
                "No executor for node_type %r; passthrough for %r.",
                node.node_type, node_id,
            )
            return inputs

        # Resource budget pre-check: warn if VRAM is critically low.
        # ``ctx.budget`` is typically a :class:`ResourceBudget` dataclass
        # (which has no ``available()`` method); only a :class:`BudgetTracker`
        # exposes one.  Guard with ``hasattr`` so the check is a no-op for
        # plain budgets instead of raising ``AttributeError`` that was
        # previously swallowed by a bare ``except Exception: pass``.
        budget = ctx.budget
        if budget is not None and hasattr(budget, "available"):
            try:
                available = budget.available()
                vram = available.get("vram_gb", float("inf"))
                if vram < _LOW_VRAM_THRESHOLD_GB:
                    self._logger.warning(
                        "Low VRAM (%.1f GB available) before executing "
                        "node %r (type %r).",
                        vram, node_id, node.node_type,
                    )
            except Exception:
                self._logger.debug(
                    "VRAM pre-check failed for node %r",
                    node_id,
                    exc_info=True,
                )

        return executor(inputs, ctx)

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    def dry_run(self, ctx: Optional[NodeContext] = None) -> Dict[str, Any]:
        """Return an execution plan and resource estimate without running.

        The plan lists every node in topological order with its layer index,
        and the estimate summarises node count, parallelism and the set of
        node types that lack a registered executor (which would fall back to
        passthrough at runtime).

        Args:
            ctx: Optional :class:`NodeContext` used to detect missing
                executors.  When ``None`` every node is reported as missing
                an executor.

        Returns:
            A dictionary with ``plan``, ``estimate``, ``order`` and
            ``groups`` keys.
        """
        errors = self.validate()
        order = self._dag.topological_sort() if not errors else []
        groups = self._dag.parallel_groups() if not errors else []
        max_parallel = max((len(g) for g in groups), default=0)

        plan: List[Dict[str, Any]] = []
        missing_executors: List[str] = []
        for layer_idx, group in enumerate(groups):
            for node_id in group:
                node = self._dag.get_node(node_id)
                has_executor = False
                if ctx is not None:
                    has_executor = ctx.resolve_executor(node.node_type) is not None
                if not has_executor:
                    missing_executors.append(node.node_type)
                plan.append({
                    "step": len(plan) + 1,
                    "node_id": node_id,
                    "node_type": node.node_type,
                    "layer": layer_idx,
                    "has_executor": has_executor,
                    "input_keys": sorted(node.inputs.keys()),
                    "dependencies": list(node.dependencies),
                })

        estimate: Dict[str, Any] = {
            "total_nodes": len(order),
            "parallel_layers": len(groups),
            "max_parallelism": max_parallel,
            "node_types": sorted({self._dag.get_node(n).node_type for n in order}),
            "missing_executor_types": sorted(set(missing_executors)),
            "validation_errors": errors,
        }
        return {
            "plan": plan,
            "estimate": estimate,
            "order": order,
            "groups": groups,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Validate the pipeline's DAG and return a list of error messages.

        An empty list means the pipeline is structurally sound and ready to
        run.
        """
        return self._dag.validate()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the pipeline (config + DAG) to a dictionary."""
        return {
            "config": self._config.to_dict(),
            "dag": self._dag.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pipeline":
        """Reconstruct a :class:`Pipeline` from a serialised dictionary."""
        config = PipelineConfig.from_dict(d["config"])
        dag = DAG.from_dict(d["dag"])
        return cls(config, dag)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the pipeline to a YAML file.

        Args:
            path: Destination file path.

        Returns:
            The resolved path that was written.
        """
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_dict(),
                handle,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        return target

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Pipeline":
        """Load a :class:`Pipeline` from a YAML file.

        Args:
            path: Source YAML file path.

        Returns:
            A reconstructed :class:`Pipeline`.
        """
        source = Path(path).expanduser().resolve()
        with open(source, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(
                "YAML pipeline file {!r} must contain a mapping.".format(source)
            )
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Pipeline(name={!r}, nodes={}, edges={})".format(
            self._config.name, self._dag.node_count, self._dag.edge_count
        )


# ---------------------------------------------------------------------------
# PipelineBuilder
# ---------------------------------------------------------------------------
class PipelineBuilder:
    """Fluent builder for constructing a :class:`Pipeline`.

    The builder accumulates nodes and edges and assembles them into a
    :class:`DAG` (with dependencies derived from the declared edges) when
    :meth:`build` is called.

    Example:
        >>> p = (PipelineBuilder("cinematic_shot")
        ...     .node("image_txt2img", id="shot1", prompt="...", width=1024)
        ...     .node("image_upscale", id="shot1_up", scale=2)
        ...     .connect("shot1", "shot1_up", output_key="image", input_key="image")
        ...     .build())
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        self._config: PipelineConfig = PipelineConfig(
            name=name,
            description=description,
            version=version,
            author=author,
            tags=list(tags) if tags else [],
        )
        self._nodes: List[DAGNode] = []
        self._edges: List[DAGEdge] = []
        self._counter: int = 0
        self._ids: set[str] = set()

    # ------------------------------------------------------------------
    # Fluent API
    # ------------------------------------------------------------------
    def node(self, node_type: str, id: Optional[str] = None, **inputs: Any) -> "PipelineBuilder":
        """Declare a node in the pipeline.

        Args:
            node_type: The executor name (e.g. ``"image_txt2img"``).
            id: Optional explicit node id.  When omitted an id of the form
                ``"<node_type>_<n>"`` is generated.
            **inputs: Static input arguments for the node.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``node_type`` is empty or ``id`` is already used.
        """
        if not node_type or not isinstance(node_type, str):
            raise ValueError("node_type must be a non-empty string.")
        if id is None:
            while True:
                candidate = "{}_{}".format(node_type, self._counter)
                self._counter += 1
                if candidate not in self._ids:
                    id = candidate
                    break
        if id in self._ids:
            raise ValueError("Duplicate node id {!r}.".format(id))
        self._ids.add(id)
        self._nodes.append(
            DAGNode(
                id=id,
                node_type=node_type,
                inputs=dict(inputs),
                dependencies=[],
            )
        )
        return self

    def connect(
        self,
        from_id: str,
        to_id: str,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None,
    ) -> "PipelineBuilder":
        """Declare a data wire between two nodes.

        Calling ``connect(a, b)`` also makes ``b`` depend on ``a``, so the
        resulting DAG's dependency edges stay consistent with its data edges.

        连接声明时会执行完整校验(委托 :class:`ConnectionValidator`):

        * **端点存在性** —— ``from_id`` 与 ``to_id`` 必须已通过 ``node()``
          声明。
        * **自环检测** —— ``from_id != to_id``。
        * **重复边检测** —— 同一条 ``(from, to, output_key, input_key)`` 不能
          重复声明。
        * **端口存在性**(可选,需要 NodeSpec)—— ``output_key`` 是上游节点
          类型的声明输出,``input_key`` 是下游节点类型的声明输入。
        * **类型兼容性**(可选,需要 TypeSystem)—— 输出端口类型与输入端口
          类型兼容。
        * **环检测**(DFS)—— 新边不引入环。

        校验失败时抛出 :class:`ValueError`。

        Args:
            from_id: The producing node id.
            to_id: The consuming node id.
            output_key: Output key on the producer (defaults to ``"output"``).
            input_key: Input key on the consumer (defaults to ``"input"``).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: 如果连接违反上述任一校验规则。
        """
        out_key = output_key or _DEFAULT_OUTPUT_KEY
        in_key = input_key or _DEFAULT_INPUT_KEY

        # 构建校验所需的上下文:已声明节点 id、现有边四元组、节点类型映射。
        existing_edges: List[tuple] = [
            (e.from_node, e.to_node, e.output_key, e.input_key)
            for e in self._edges
        ]
        node_type_map: Dict[str, str] = {
            node.id: node.node_type for node in self._nodes
        }
        # 惰性加载节点规格(注册表不可用时返回 None,跳过端口 / 类型校验)。
        specs = ConnectionValidator.load_specs()

        error = ConnectionValidator.validate_connection(
            from_id=from_id,
            to_id=to_id,
            output_key=out_key,
            input_key=in_key,
            declared_ids=self._ids,
            existing_edges=existing_edges,
            node_type_map=node_type_map,
            specs=specs,
        )
        if error is not None:
            raise ValueError(error)

        self._edges.append(
            DAGEdge(
                from_node=from_id,
                to_node=to_id,
                output_key=out_key,
                input_key=in_key,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> Pipeline:
        """Assemble the accumulated nodes and edges into a :class:`Pipeline`.

        Dependencies are derived from the declared edges: for every edge
        ``a -> b`` the id ``a`` is added to ``b``'s dependencies (de-duplicated,
        preserving insertion order).

        Returns:
            A new :class:`Pipeline`.

        Raises:
            ValueError: If no nodes have been declared.
        """
        if not self._nodes:
            raise ValueError("Cannot build a pipeline with no nodes.")

        # Derive dependencies from edges.
        deps_map: Dict[str, List[str]] = {}
        for edge in self._edges:
            deps = deps_map.setdefault(edge.to_node, [])
            if edge.from_node not in deps:
                deps.append(edge.from_node)

        dag = DAG()
        for node in self._nodes:
            node.dependencies = list(deps_map.get(node.id, []))
            dag.add_node(node)
        for edge in self._edges:
            dag.add_edge(edge)

        return Pipeline(self._config, dag)

    def __repr__(self) -> str:
        return "PipelineBuilder(name={!r}, nodes={}, edges={})".format(
            self._config.name, len(self._nodes), len(self._edges)
        )
