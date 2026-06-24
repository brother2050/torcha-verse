"""Canvas core for the TorchaVerse v0.3.0 architecture (L6 orchestration).

This module provides the visual *canvas* abstraction that sits above the L5
pipeline layer.  A canvas is a spatial, editable representation of a pipeline
DAG: nodes are placed at 2-D coordinates, connections are drawn between their
ports, and the whole thing can be serialised, versioned, shared and
auto-generated.

The canvas layer is deliberately *torch-free*: it only imports from the L5
pipeline layer (:class:`~pipeline.dag.DAG`, :class:`~pipeline.composer.Pipeline`)
and the Python standard library.  Node executors are referenced by their string
``type`` and resolved lazily at execution time, exactly like the pipeline layer.

Public surface
--------------

* :class:`CanvasNode` -- a single node on the canvas (id, type, position,
  inputs, size).
* :class:`CanvasConnection` -- a typed wire between two canvas nodes
  (from_port -> to_port).
* :class:`CanvasState` -- the full mutable state of a canvas (nodes,
  connections, viewport, zoom).  Supports round-trip serialisation and
  conversion to / from a :class:`~pipeline.dag.DAG`.
* :class:`Canvas` -- the high-level canvas object with add / remove / connect /
  disconnect, validation, serialisation (YAML / JSON), snapshotting, forking,
  merging and auto-layout.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import yaml

from pipeline.composer import Pipeline, PipelineConfig
from pipeline.dag import DAG, DAGEdge, DAGNode

__all__ = [
    "CanvasNode",
    "CanvasConnection",
    "CanvasState",
    "Canvas",
]

# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- this layer must not import torch).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("canvas.canvas")

# ---------------------------------------------------------------------------
# Layout constants (used by auto-layout and from_dag).
# ---------------------------------------------------------------------------
#: Horizontal spacing between topological layers (pixels).
_LAYOUT_LAYER_SPACING: float = 300.0
#: Vertical spacing between nodes within a layer (pixels).
_LAYOUT_NODE_SPACING: float = 150.0
#: X origin for the first layer.
_LAYOUT_ORIGIN_X: float = 100.0
#: Y origin for the first node in each layer.
_LAYOUT_ORIGIN_Y: float = 100.0

#: Default canvas node size (width, height) in pixels.
_DEFAULT_NODE_SIZE: Tuple[float, float] = (200.0, 100.0)
#: Default zoom level.
_DEFAULT_ZOOM: float = 1.0
#: Default viewport dictionary.
_DEFAULT_VIEWPORT: Dict[str, float] = {
    "x": 0.0,
    "y": 0.0,
    "width": 1920.0,
    "height": 1080.0,
}
#: Default pipeline version string.
_DEFAULT_PIPELINE_VERSION: str = "1.0.0"
#: Prefix used when merging foreign nodes to avoid id collisions.
_MERGE_ID_PREFIX: str = "merged_"


# ---------------------------------------------------------------------------
# CanvasNode
# ---------------------------------------------------------------------------
@dataclass
class CanvasNode:
    """A single node placed on the canvas.

    A canvas node is the visual counterpart of a :class:`~pipeline.dag.DAGNode`:
    it carries the same ``id``, ``type`` (mapped to ``node_type`` on the DAG)
    and ``inputs``, but adds spatial information (``position`` and ``size``)
    that is irrelevant to execution but essential for the visual editor.

    Attributes:
        id: Unique node identifier within a canvas.
        type: Executor name resolved at runtime (e.g. ``"text_chat"``,
            ``"image_txt2img"``).  Maps to ``DAGNode.node_type``.
        position: ``(x, y)`` pixel coordinates of the node's top-left corner.
        inputs: Static input arguments for the node (same semantics as
            ``DAGNode.inputs``).
        size: ``(width, height)`` pixel dimensions of the node's visual box.
    """

    id: str
    type: str
    position: Tuple[float, float] = (0.0, 0.0)
    inputs: Dict[str, Any] = field(default_factory=dict)
    size: Tuple[float, float] = _DEFAULT_NODE_SIZE

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this node to a JSON-serialisable dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "position": list(self.position),
            "inputs": dict(self.inputs),
            "size": list(self.size),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasNode":
        """Reconstruct a :class:`CanvasNode` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`CanvasNode` instance.
        """
        pos = d.get("position", (_LAYOUT_ORIGIN_X, _LAYOUT_ORIGIN_Y))
        size = d.get("size", _DEFAULT_NODE_SIZE)
        return cls(
            id=d["id"],
            type=d["type"],
            position=(float(pos[0]), float(pos[1])),
            inputs=dict(d.get("inputs") or {}),
            size=(float(size[0]), float(size[1])),
        )

    def __repr__(self) -> str:
        return "CanvasNode(id={!r}, type={!r}, pos={})".format(
            self.id, self.type, self.position
        )


# ---------------------------------------------------------------------------
# CanvasConnection
# ---------------------------------------------------------------------------
@dataclass
class CanvasConnection:
    """A typed data wire between two :class:`CanvasNode` instances.

    A connection declares that the output produced under ``from_port`` by
    ``from_node`` should be fed as ``to_port`` into ``to_node``.  This maps
    directly to a :class:`~pipeline.dag.DAGEdge` where ``from_port`` becomes
    ``output_key`` and ``to_port`` becomes ``input_key``.

    Attributes:
        id: Unique connection identifier within a canvas.
        from_node: Id of the producing (upstream) node.
        from_port: Output port name on the upstream node
            (maps to ``DAGEdge.output_key``).
        to_node: Id of the consuming (downstream) node.
        to_port: Input port name on the downstream node
            (maps to ``DAGEdge.input_key``).
    """

    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this connection to a JSON-serialisable dictionary."""
        return {
            "id": self.id,
            "from_node": self.from_node,
            "from_port": self.from_port,
            "to_node": self.to_node,
            "to_port": self.to_port,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasConnection":
        """Reconstruct a :class:`CanvasConnection` from a serialised dict.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`CanvasConnection` instance.
        """
        return cls(
            id=d["id"],
            from_node=d["from_node"],
            from_port=d["from_port"],
            to_node=d["to_node"],
            to_port=d["to_port"],
        )

    def __repr__(self) -> str:
        return "CanvasConnection({}.{} -> {}.{})".format(
            self.from_node, self.from_port, self.to_node, self.to_port
        )


# ---------------------------------------------------------------------------
# CanvasState
# ---------------------------------------------------------------------------
@dataclass
class CanvasState:
    """The full mutable state of a canvas.

    A :class:`CanvasState` bundles every node and connection on the canvas
    together with viewport and zoom metadata.  It is the unit of serialisation,
    versioning and sharing: a snapshot of the state fully captures the canvas
    at a point in time.

    The state can be converted to and from a :class:`~pipeline.dag.DAG`:

    * :meth:`to_dag` produces a DAG whose nodes carry dependencies derived
      from the canvas connections, and whose edges mirror the connection
      ports.
    * :meth:`from_dag` produces a canvas state with an auto-computed layered
      layout.

    Attributes:
        nodes: Ordered list of :class:`CanvasNode` instances.
        connections: Ordered list of :class:`CanvasConnection` instances.
        viewport: Viewport dictionary (``x``, ``y``, ``width``, ``height``).
        zoom: Zoom level (``1.0`` = 100%).
    """

    nodes: List[CanvasNode] = field(default_factory=list)
    connections: List[CanvasConnection] = field(default_factory=list)
    viewport: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_VIEWPORT)
    )
    zoom: float = _DEFAULT_ZOOM

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the whole state to a JSON-serialisable dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "viewport": dict(self.viewport),
            "zoom": self.zoom,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasState":
        """Reconstruct a :class:`CanvasState` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`CanvasState` instance.
        """
        viewport = d.get("viewport") or {}
        if not viewport:
            viewport = dict(_DEFAULT_VIEWPORT)
        return cls(
            nodes=[CanvasNode.from_dict(n) for n in d.get("nodes", [])],
            connections=[
                CanvasConnection.from_dict(c) for c in d.get("connections", [])
            ],
            viewport=dict(viewport),
            zoom=float(d.get("zoom", _DEFAULT_ZOOM)),
        )

    # ------------------------------------------------------------------
    # DAG conversion
    # ------------------------------------------------------------------
    def to_dag(self) -> DAG:
        """Convert this canvas state into a :class:`~pipeline.dag.DAG`.

        Each :class:`CanvasNode` becomes a :class:`~pipeline.dag.DAGNode`
        (with ``node_type`` set from ``type`` and ``dependencies`` derived
        from the connections).  Each :class:`CanvasConnection` becomes a
        :class:`~pipeline.dag.DAGEdge` (``from_port`` -> ``output_key``,
        ``to_port`` -> ``input_key``).

        Returns:
            A new :class:`~pipeline.dag.DAG` mirroring this canvas state.
        """
        dag = DAG()

        # Derive dependencies from connections: for every connection
        # from_node -> to_node, add from_node to to_node's dependencies.
        deps_map: Dict[str, List[str]] = {}
        for conn in self.connections:
            deps = deps_map.setdefault(conn.to_node, [])
            if conn.from_node not in deps:
                deps.append(conn.from_node)

        # Add DAG nodes.
        for node in self.nodes:
            dag_node = DAGNode(
                id=node.id,
                node_type=node.type,
                inputs=dict(node.inputs),
                dependencies=list(deps_map.get(node.id, [])),
            )
            dag.add_node(dag_node)

        # Add DAG edges.
        for conn in self.connections:
            dag_edge = DAGEdge(
                from_node=conn.from_node,
                to_node=conn.to_node,
                output_key=conn.from_port,
                input_key=conn.to_port,
            )
            dag.add_edge(dag_edge)

        return dag

    @classmethod
    def from_dag(cls, dag: DAG) -> "CanvasState":
        """Create a canvas state from a :class:`~pipeline.dag.DAG`.

        The canvas nodes are positioned using a simple layered layout:
        nodes are grouped into topological layers (via
        :meth:`~pipeline.dag.DAG.parallel_groups`) and placed left-to-right,
        stacked vertically within each layer.

        Args:
            dag: The source :class:`~pipeline.dag.DAG`.

        Returns:
            A new :class:`CanvasState` with auto-laid-out nodes and
            connections mirroring the DAG's edges.
        """
        dag_dict = dag.to_dict()

        # Compute layered positions.
        position_map: Dict[str, Tuple[float, float]] = {}
        try:
            groups = dag.parallel_groups()
        except ValueError:
            # Cycle in the DAG -- fall back to insertion order.
            groups = [list(dag.node_ids)]

        for layer_idx, group in enumerate(groups):
            for node_idx, node_id in enumerate(group):
                x = _LAYOUT_ORIGIN_X + layer_idx * _LAYOUT_LAYER_SPACING
                y = _LAYOUT_ORIGIN_Y + node_idx * _LAYOUT_NODE_SPACING
                position_map[node_id] = (x, y)

        # Build canvas nodes.
        canvas_nodes: List[CanvasNode] = []
        for node_d in dag_dict.get("nodes", []):
            node_id = node_d["id"]
            pos = position_map.get(
                node_id, (_LAYOUT_ORIGIN_X, _LAYOUT_ORIGIN_Y)
            )
            canvas_nodes.append(
                CanvasNode(
                    id=node_id,
                    type=node_d["node_type"],
                    position=pos,
                    inputs=dict(node_d.get("inputs") or {}),
                )
            )

        # Build canvas connections.
        canvas_conns: List[CanvasConnection] = []
        for edge_d in dag_dict.get("edges", []):
            canvas_conns.append(
                CanvasConnection(
                    id=str(uuid4()),
                    from_node=edge_d["from_node"],
                    from_port=edge_d["output_key"],
                    to_node=edge_d["to_node"],
                    to_port=edge_d["input_key"],
                )
            )

        return cls(nodes=canvas_nodes, connections=canvas_conns)

    def __repr__(self) -> str:
        return "CanvasState(nodes={}, connections={}, zoom={})".format(
            len(self.nodes), len(self.connections), self.zoom
        )


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
class Canvas:
    """A high-level, thread-safe canvas backed by a :class:`CanvasState`.

    A :class:`Canvas` is the user-facing object for building, editing and
    serialising a visual pipeline.  It wraps a :class:`CanvasState` with
    convenience methods for adding / removing nodes, connecting ports,
    validating the graph, converting to an executable
    :class:`~pipeline.composer.Pipeline`, and serialising to / from YAML and
    JSON.

    All public operations are guarded by a re-entrant lock so that a canvas
    may be assembled and inspected concurrently from multiple threads.

    Args:
        name: Human-readable canvas name.
        state: Optional initial :class:`CanvasState`.  When ``None`` an
            empty state is created.
    """

    def __init__(
        self, name: str, state: Optional[CanvasState] = None
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Canvas name must be a non-empty string.")
        self._name: str = name
        self._state: CanvasState = state if state is not None else CanvasState()
        self._lock: threading.RLock = threading.RLock()
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """The canvas name."""
        return self._name

    @property
    def state(self) -> CanvasState:
        """The underlying :class:`CanvasState` (read snapshot)."""
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_type: str,
        id: Optional[str] = None,
        position: Tuple[float, float] = (0.0, 0.0),
        **inputs: Any,
    ) -> CanvasNode:
        """Add a node to the canvas.

        Args:
            node_type: The executor name (e.g. ``"image_txt2img"``).
            id: Optional explicit node id.  When omitted an id of the form
                ``"node_<n>"`` is generated.
            position: ``(x, y)`` pixel coordinates.
            **inputs: Static input arguments for the node.

        Returns:
            The newly created :class:`CanvasNode`.

        Raises:
            ValueError: If ``node_type`` is empty or ``id`` is already used.
        """
        if not node_type or not isinstance(node_type, str):
            raise ValueError("node_type must be a non-empty string.")
        with self._lock:
            if id is None:
                existing_ids = {n.id for n in self._state.nodes}
                while True:
                    candidate = "node_{}".format(self._counter)
                    self._counter += 1
                    if candidate not in existing_ids:
                        id = candidate
                        break
            else:
                existing_ids = {n.id for n in self._state.nodes}
                if id in existing_ids:
                    raise ValueError("Duplicate node id {!r}.".format(id))
            node = CanvasNode(
                id=id,
                type=node_type,
                position=(float(position[0]), float(position[1])),
                inputs=dict(inputs),
            )
            self._state.nodes.append(node)
            _logger.debug("Added node %r (type=%s).", id, node_type)
            return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections from the canvas.

        Args:
            node_id: The id of the node to remove.

        Returns:
            ``True`` if the node was found and removed, ``False`` otherwise.
        """
        with self._lock:
            original_len = len(self._state.nodes)
            self._state.nodes = [
                n for n in self._state.nodes if n.id != node_id
            ]
            removed = len(self._state.nodes) < original_len
            if removed:
                # Remove all connections referencing this node.
                self._state.connections = [
                    c
                    for c in self._state.connections
                    if c.from_node != node_id and c.to_node != node_id
                ]
                _logger.debug("Removed node %r.", node_id)
            return removed

    def move_node(
        self, node_id: str, position: Tuple[float, float]
    ) -> None:
        """Move a node to a new position.

        Args:
            node_id: The id of the node to move.
            position: The new ``(x, y)`` coordinates.

        Raises:
            KeyError: If no node with ``node_id`` exists.
        """
        with self._lock:
            node = self._find_node(node_id)
            if node is None:
                raise KeyError("No CanvasNode with id={!r}.".format(node_id))
            node.position = (float(position[0]), float(position[1]))

    def get_node(self, node_id: str) -> CanvasNode:
        """Return the node registered under ``node_id``.

        Args:
            node_id: The node id to look up.

        Returns:
            The :class:`CanvasNode`.

        Raises:
            KeyError: If no node with that id exists.
        """
        with self._lock:
            node = self._find_node(node_id)
            if node is None:
                raise KeyError("No CanvasNode with id={!r}.".format(node_id))
            return node

    def list_nodes(self) -> List[CanvasNode]:
        """Return a list of all nodes on the canvas (insertion order)."""
        with self._lock:
            return list(self._state.nodes)

    def list_connections(self) -> List[CanvasConnection]:
        """Return a list of all connections on the canvas."""
        with self._lock:
            return list(self._state.connections)

    def _find_node(self, node_id: str) -> Optional[CanvasNode]:
        """Return the node with ``node_id`` or ``None`` (caller holds lock)."""
        for node in self._state.nodes:
            if node.id == node_id:
                return node
        return None

    # ------------------------------------------------------------------
    # Connection operations
    # ------------------------------------------------------------------
    def connect(
        self,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str,
    ) -> CanvasConnection:
        """Connect two nodes on the canvas.

        Args:
            from_node: Id of the producing (upstream) node.
            from_port: Output port name on the upstream node.
            to_node: Id of the consuming (downstream) node.
            to_port: Input port name on the downstream node.

        Returns:
            The newly created :class:`CanvasConnection`.

        Raises:
            ValueError: If any endpoint or port is empty, or if a duplicate
                connection already exists.
        """
        if not from_node or not to_node:
            raise ValueError("Connection endpoints must be non-empty strings.")
        if not from_port or not to_port:
            raise ValueError("Connection ports must be non-empty strings.")
        with self._lock:
            # Check for duplicate connections.
            for conn in self._state.connections:
                if (
                    conn.from_node == from_node
                    and conn.from_port == from_port
                    and conn.to_node == to_node
                    and conn.to_port == to_port
                ):
                    raise ValueError(
                        "Duplicate connection {}.{} -> {}.{}.".format(
                            from_node, from_port, to_node, to_port
                        )
                    )
            connection = CanvasConnection(
                id=str(uuid4()),
                from_node=from_node,
                from_port=from_port,
                to_node=to_node,
                to_port=to_port,
            )
            self._state.connections.append(connection)
            _logger.debug(
                "Connected %s.%s -> %s.%s.",
                from_node, from_port, to_node, to_port,
            )
            return connection

    def disconnect(self, connection_id: str) -> bool:
        """Remove a connection by id.

        Args:
            connection_id: The id of the connection to remove.

        Returns:
            ``True`` if the connection was found and removed.
        """
        with self._lock:
            original_len = len(self._state.connections)
            self._state.connections = [
                c for c in self._state.connections if c.id != connection_id
            ]
            removed = len(self._state.connections) < original_len
            if removed:
                _logger.debug("Disconnected %r.", connection_id)
            return removed

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Validate the canvas and return a list of error messages.

        An empty list means the canvas is structurally sound.  The checks
        performed are:

        * **Dangling connections** -- a connection references a ``from_node``
          or ``to_node`` that does not exist on the canvas.
        * **Self-loops** -- a connection where ``from_node == to_node``.
        * **Duplicate connections** -- the same ``(from, from_port, to,
          to_port)`` wire declared more than once.
        * **Type matching** (best-effort) -- when node specs are available
          (via the L4 node registry), checks that ``from_port`` is a declared
          output of the upstream node's type and ``to_port`` is a declared
          input of the downstream node's type.  Specs that cannot be loaded
          are silently skipped so that validation never fails due to missing
          optional dependencies.

        Returns:
            A list of human-readable error strings.
        """
        errors: List[str] = []
        with self._lock:
            node_ids = {n.id for n in self._state.nodes}
            node_type_map = {n.id: n.type for n in self._state.nodes}

            # Dangling connections and self-loops.
            for conn in self._state.connections:
                if conn.from_node not in node_ids:
                    errors.append(
                        "Connection {!r} references missing from_node "
                        "{!r}.".format(conn.id, conn.from_node)
                    )
                if conn.to_node not in node_ids:
                    errors.append(
                        "Connection {!r} references missing to_node "
                        "{!r}.".format(conn.id, conn.to_node)
                    )
                if conn.from_node == conn.to_node:
                    errors.append(
                        "Connection {!r} is a self-loop on node "
                        "{!r}.".format(conn.id, conn.from_node)
                    )

            # Duplicate connections.
            seen: set[Tuple[str, str, str, str]] = set()
            for conn in self._state.connections:
                key = (
                    conn.from_node,
                    conn.from_port,
                    conn.to_node,
                    conn.to_port,
                )
                if key in seen:
                    errors.append(
                        "Duplicate connection {}.{} -> {}.{}.".format(
                            conn.from_node,
                            conn.from_port,
                            conn.to_node,
                            conn.to_port,
                        )
                    )
                seen.add(key)

        # Best-effort type matching (optional, never raises).
        type_errors = self._validate_port_types(node_type_map)
        errors.extend(type_errors)

        return errors

    def _validate_port_types(
        self, node_type_map: Dict[str, str]
    ) -> List[str]:
        """Best-effort port-type validation using the L4 node registry.

        This method attempts to lazily import the node registry to obtain
        :class:`~nodes.base.NodeSpec` objects.  If the import fails or a
        node type has no spec, the check is silently skipped for that
        connection.

        Args:
            node_type_map: Mapping of ``node_id -> node_type``.

        Returns:
            A list of type-mismatch error strings (may be empty).
        """
        errors: List[str] = []
        try:
            from nodes import NodeRegistry  # type: ignore[import-not-found]

            registry = NodeRegistry()
            specs = {spec.type: spec for spec in registry.list()}
        except Exception:
            # Node registry not available -- skip type checking entirely.
            return errors

        with self._lock:
            for conn in self._state.connections:
                from_type = node_type_map.get(conn.from_node)
                to_type = node_type_map.get(conn.to_node)
                if from_type is None or to_type is None:
                    continue
                from_spec = specs.get(from_type)
                to_spec = specs.get(to_type)
                if from_spec is not None:
                    if conn.from_port not in from_spec.outputs:
                        errors.append(
                            "Port {!r} is not a declared output of node "
                            "type {!r} (node {!r}).".format(
                                conn.from_port, from_type, conn.from_node
                            )
                        )
                if to_spec is not None:
                    if conn.to_port not in to_spec.inputs:
                        errors.append(
                            "Port {!r} is not a declared input of node "
                            "type {!r} (node {!r}).".format(
                                conn.to_port, to_type, conn.to_node
                            )
                        )
        return errors

    # ------------------------------------------------------------------
    # Pipeline conversion
    # ------------------------------------------------------------------
    def to_pipeline(self) -> Pipeline:
        """Convert this canvas into an executable :class:`Pipeline`.

        The canvas state is converted to a :class:`~pipeline.dag.DAG` (via
        :meth:`CanvasState.to_dag`) and wrapped in a
        :class:`~pipeline.composer.PipelineConfig` using the canvas name.

        Returns:
            A new :class:`~pipeline.composer.Pipeline`.
        """
        with self._lock:
            config = PipelineConfig(
                name=self._name,
                version=_DEFAULT_PIPELINE_VERSION,
            )
            dag = self._state.to_dag()
        return Pipeline(config, dag)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the canvas (name + state) to a dictionary."""
        with self._lock:
            return {
                "name": self._name,
                "state": self._state.to_dict(),
            }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Canvas":
        """Reconstruct a :class:`Canvas` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`Canvas` instance.
        """
        name = d.get("name", "canvas")
        state = CanvasState.from_dict(d.get("state") or {})
        return cls(name, state)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the canvas to a YAML file.

        Args:
            path: Destination file path.

        Returns:
            The resolved path that was written.
        """
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = self.to_dict()
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                data,
                handle,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        return target

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Canvas":
        """Load a :class:`Canvas` from a YAML file.

        Args:
            path: Source YAML file path.

        Returns:
            A reconstructed :class:`Canvas`.
        """
        source = Path(path).expanduser().resolve()
        with open(source, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(
                "YAML canvas file {!r} must contain a mapping.".format(source)
            )
        return cls.from_dict(data)

    def to_json(self) -> str:
        """Serialise the canvas to a JSON string.

        Returns:
            A JSON string representation of the canvas.
        """
        with self._lock:
            data = self.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Canvas":
        """Reconstruct a :class:`Canvas` from a JSON string.

        Args:
            json_str: JSON string produced by :meth:`to_json`.

        Returns:
            A new :class:`Canvas` instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Snapshot / fork / merge
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a version-control-friendly snapshot of the canvas.

        The snapshot includes the canvas name, a timestamp, the full
        serialised state, and summary counts.

        Returns:
            A dictionary suitable for storing as a version-control entry.
        """
        with self._lock:
            return {
                "name": self._name,
                "timestamp": time.time(),
                "state": self._state.to_dict(),
                "node_count": len(self._state.nodes),
                "connection_count": len(self._state.connections),
            }

    def fork(self, new_name: str) -> "Canvas":
        """Create a deep copy of this canvas with a new name.

        Args:
            new_name: Name for the forked canvas.

        Returns:
            A new :class:`Canvas` with an independent copy of the state.
        """
        with self._lock:
            state = CanvasState.from_dict(self._state.to_dict())
        return Canvas(new_name, state)

    def merge(self, other: "Canvas") -> "Canvas":
        """Merge another canvas into a new combined canvas.

        Nodes from ``other`` are added with a prefix to avoid id
        collisions.  Connections from ``other`` are rewired to the new
        prefixed ids.  The result is a *new* canvas; neither ``self`` nor
        ``other`` is modified.

        Args:
            other: The canvas to merge into this one.

        Returns:
            A new :class:`Canvas` containing nodes and connections from
            both canvases.
        """
        # Read snapshots from both canvases (thread-safe).
        self_snapshot = self.to_dict()
        other_snapshot = other.to_dict()

        merged = Canvas.from_dict(self_snapshot)
        id_map: Dict[str, str] = {}

        # Add nodes from other with prefixed ids.
        for node_d in other_snapshot["state"]["nodes"]:
            old_id = node_d["id"]
            new_id = _MERGE_ID_PREFIX + old_id
            id_map[old_id] = new_id
            merged.add_node(
                node_d["type"],
                id=new_id,
                position=tuple(node_d.get("position", (0.0, 0.0))),
                **dict(node_d.get("inputs") or {}),
            )

        # Add connections from other with rewired ids.
        for conn_d in other_snapshot["state"]["connections"]:
            new_from = id_map.get(
                conn_d["from_node"], conn_d["from_node"]
            )
            new_to = id_map.get(conn_d["to_node"], conn_d["to_node"])
            merged.connect(
                new_from,
                conn_d["from_port"],
                new_to,
                conn_d["to_port"],
            )

        return merged

    # ------------------------------------------------------------------
    # Auto-layout
    # ------------------------------------------------------------------
    def auto_layout(self) -> None:
        """Re-position all nodes using a simple layered layout.

        Nodes are grouped into topological layers (via the underlying DAG's
        :meth:`~pipeline.dag.DAG.parallel_groups`) and placed left-to-right,
        stacked vertically within each layer.  Nodes that cannot be ordered
        (e.g. due to a cycle) retain their current positions.
        """
        with self._lock:
            dag = self._state.to_dag()
            try:
                groups = dag.parallel_groups()
            except ValueError:
                _logger.warning(
                    "Canvas %r contains a cycle; skipping auto-layout.",
                    self._name,
                )
                return

            position_map: Dict[str, Tuple[float, float]] = {}
            for layer_idx, group in enumerate(groups):
                for node_idx, node_id in enumerate(group):
                    x = _LAYOUT_ORIGIN_X + layer_idx * _LAYOUT_LAYER_SPACING
                    y = _LAYOUT_ORIGIN_Y + node_idx * _LAYOUT_NODE_SPACING
                    position_map[node_id] = (x, y)

            for node in self._state.nodes:
                if node.id in position_map:
                    node.position = position_map[node.id]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            return "Canvas(name={!r}, nodes={}, connections={})".format(
                self._name,
                len(self._state.nodes),
                len(self._state.connections),
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._state.nodes)
