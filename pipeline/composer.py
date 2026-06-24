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

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .dag import DAG, DAGEdge, DAGNode

__all__ = [
    "NodeContext",
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

#: Default maximum number of worker threads used when running a parallel
#: group.  The actual concurrency is the smaller of this value and the group
#: size, so a tiny group never spins up idle workers.
_DEFAULT_MAX_WORKERS: int = 8

#: Sentinel returned by :meth:`NodeContext.get_output` when a key is absent,
#: distinguishing a stored ``None`` from a missing entry.
_MISSING: Any = object()


# Type alias for a node executor callable.
NodeExecutor = Callable[[Dict[str, Any], "NodeContext"], Dict[str, Any]]

#: Type alias for the progress callback signature.
ProgressCallback = Callable[[int, int, str, str], None]


# ---------------------------------------------------------------------------
# NodeContext
# ---------------------------------------------------------------------------
class NodeContext:
    """Per-run execution context shared across all nodes of a pipeline.

    A :class:`NodeContext` carries three responsibilities:

    1. **Output store** -- a thread-safe mapping of ``node_id -> outputs``
       populated as nodes complete, so downstream nodes can fetch upstream
       results.
    2. **Executor resolution** -- a lookup chain that finds the callable to
       run for a given ``node_type``.  Explicit per-type executors take
       precedence, then the :class:`~core.module_bus.ModuleBus` (under the
       ``node.<node_type>`` namespace), then ``None`` (passthrough).
    3. **Metadata** -- an arbitrary, mutable key/value bag for run-level
       configuration (seeds, device hints, asset references, ...).

    The context is intentionally lightweight and dependency-free so that it
    can be constructed in any environment.

    Args:
        bus: Optional :class:`~core.module_bus.ModuleBus` used for executor
            resolution.  When ``None`` only the explicit ``executors`` map
            is consulted.
        executors: Optional mapping of ``node_type -> callable``.  Each
            callable receives ``(inputs, ctx)`` and returns an output dict.
        metadata: Optional initial metadata dictionary.
        max_workers: Default worker cap for parallel execution.
    """

    def __init__(
        self,
        bus: Any = None,
        executors: Optional[Dict[str, NodeExecutor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        strict_mode: bool = False,
        budget: Any = None,
    ) -> None:
        self._bus = bus
        self._executors: Dict[str, NodeExecutor] = dict(executors or {})
        self._metadata: Dict[str, Any] = dict(metadata or {})
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._max_workers: int = max(1, int(max_workers))
        self._strict_mode: bool = bool(strict_mode)
        self._budget: Any = budget
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bus(self) -> Any:
        """The underlying :class:`~core.module_bus.ModuleBus` (or ``None``)."""
        return self._bus

    @property
    def max_workers(self) -> int:
        """The default worker cap for parallel execution."""
        return self._max_workers

    @property
    def strict_mode(self) -> bool:
        """When ``True``, missing executors raise instead of passthrough."""
        return self._strict_mode

    @property
    def budget(self) -> Any:
        """The resource budget tracker (if configured)."""
        return self._budget

    @property
    def metadata(self) -> Dict[str, Any]:
        """The mutable run-level metadata bag."""
        return self._metadata

    # ------------------------------------------------------------------
    # Output store
    # ------------------------------------------------------------------
    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Record the outputs produced by ``node_id``.

        Args:
            node_id: The id of the node that produced the outputs.
            outputs: The output dictionary to store.
        """
        with self._lock:
            self._outputs[node_id] = dict(outputs)

    def get_output(
        self, node_id: str, key: Optional[str] = None
    ) -> Any:
        """Retrieve an output produced by a node.

        Args:
            node_id: The id of the producing node.
            key: When given, return the specific output key; when ``None``,
                return the node's entire output dict.

        Returns:
            The requested value (or output dict).  Missing keys return
            ``None``; missing nodes raise :class:`KeyError`.

        Raises:
            KeyError: If no outputs have been recorded for ``node_id``.
        """
        with self._lock:
            outputs = self._outputs.get(node_id, _MISSING)
        if outputs is _MISSING:
            raise KeyError(
                "No outputs recorded for node {!r}.".format(node_id)
            )
        if key is None:
            return dict(outputs)
        return outputs.get(key)

    def has_output(self, node_id: str) -> bool:
        """Return ``True`` if outputs have been recorded for ``node_id``."""
        with self._lock:
            return node_id in self._outputs

    def all_outputs(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of every recorded node output."""
        with self._lock:
            return {nid: dict(out) for nid, out in self._outputs.items()}

    def reset(self) -> None:
        """Clear all recorded outputs and reset the run state."""
        with self._lock:
            self._outputs.clear()

    # ------------------------------------------------------------------
    # Executor resolution
    # ------------------------------------------------------------------
    def register_executor(self, node_type: str, executor: NodeExecutor) -> None:
        """Register an explicit executor for a node type.

        Args:
            node_type: The node type the executor handles.
            executor: A callable ``(inputs, ctx) -> outputs``.
        """
        with self._lock:
            self._executors[node_type] = executor

    def resolve_executor(self, node_type: str) -> Optional[NodeExecutor]:
        """Resolve the executor callable for ``node_type``.

        Lookup order:

        1. The explicit ``executors`` map.
        2. The :class:`~core.module_bus.ModuleBus` under the ``"node"``
           kind (when a bus is configured).  When the resolved object is a
           :class:`~nodes.base.BaseNode` instance (i.e. has an ``execute``
           method), it is wrapped in an adapter that converts the pipeline
           call signature ``(inputs, ctx)`` into the node signature
           ``execute(ctx, **inputs)``.

        Returns:
            The executor callable, or ``None`` if none is registered (in
            which case the pipeline falls back to a passthrough).
        """
        with self._lock:
            executor = self._executors.get(node_type)
        if executor is not None:
            return executor
        if self._bus is not None:
            try:
                if self._bus.has("node", node_type):  # type: ignore[union-attr]
                    resolved = self._bus.resolve("node", node_type)  # type: ignore[union-attr]
                    # If the resolved object is a BaseNode instance, wrap
                    # it in an adapter that converts (inputs, ctx) ->
                    # node.execute(ctx, **inputs).
                    if hasattr(resolved, "execute") and callable(
                        getattr(resolved, "execute")
                    ):
                        def _node_adapter(
                            inputs: Dict[str, Any],
                            ctx: "NodeContext",
                        ) -> Dict[str, Any]:
                            from nodes.base import NodeContext as L4NodeContext
                            l4_ctx = L4NodeContext(
                                bus=getattr(ctx, "bus", None) or self._bus,
                                config=getattr(ctx, "metadata", {}),
                                budget=getattr(ctx, "budget", None),
                            )
                            return resolved.execute(l4_ctx, **inputs)
                        return _node_adapter
                    return resolved
            except Exception:  # pragma: no cover - defensive
                _logger.debug("ModuleBus lookup failed for %s", node_type, exc_info=True)
        return None

    def __repr__(self) -> str:
        return "NodeContext(outputs={}, executors={})".format(
            len(self._outputs), len(self._executors)
        )


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

        # Run-control primitives.
        self._cancel_event: threading.Event = threading.Event()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()  # not paused by default
        self._run_lock: threading.RLock = threading.RLock()
        self._running: bool = False

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

        for group in groups:
            # Cooperative pause: block (with a small spin) until resumed.
            while not self._pause_event.is_set():
                if self._cancel_event.is_set():
                    break
                time.sleep(0.01)
            if self._cancel_event.is_set():
                self._logger.info(
                    "Pipeline %r cancelled before layer %d.",
                    self._config.name, step,
                )
                break

            # Run the layer concurrently.
            workers = min(ctx.max_workers, len(group)) if group else 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
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
        inputs: Dict[str, Any] = dict(node.inputs)

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
        budget = ctx.budget
        if budget is not None:
            try:
                available = budget.available()
                vram = available.get("vram_gb", float("inf"))
                if vram < 1.0:
                    self._logger.warning(
                        "Low VRAM (%.1f GB available) before executing "
                        "node %r (type %r).",
                        vram, node_id, node.node_type,
                    )
            except Exception:
                pass  # Don't block execution on budget check failure

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

        Args:
            from_id: The producing node id.
            to_id: The consuming node id.
            output_key: Output key on the producer (defaults to ``"output"``).
            input_key: Input key on the consumer (defaults to ``"input"``).

        Returns:
            ``self`` for chaining.
        """
        self._edges.append(
            DAGEdge(
                from_node=from_id,
                to_node=to_id,
                output_key=output_key or _DEFAULT_OUTPUT_KEY,
                input_key=input_key or _DEFAULT_INPUT_KEY,
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
