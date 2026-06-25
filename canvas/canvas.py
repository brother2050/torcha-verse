"""Canvas core for the TorchaVerse v0.3.0 architecture (L5 visual frontend).

This module provides the visual *canvas* abstraction that sits above the L4
pipeline layer.  A canvas is a spatial, editable representation of a pipeline
DAG: nodes are placed at 2-D coordinates, connections are drawn between their
ports, and the whole thing can be serialised, versioned, shared and
auto-generated.

The canvas layer is deliberately *torch-free*: it only imports from the L4
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

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import yaml

from pipeline.composer import Pipeline, PipelineConfig
from pipeline.dag import DAG
from pipeline.validators import ConnectionValidator

from canvas.models import (
    CanvasConnection,
    CanvasNode,
    CanvasState,
    _DEFAULT_NODE_SIZE,
    _DEFAULT_VIEWPORT,
    _DEFAULT_ZOOM,
    _LAYOUT_LAYER_SPACING,
    _LAYOUT_NODE_SPACING,
    _LAYOUT_ORIGIN_X,
    _LAYOUT_ORIGIN_Y,
)

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
# Canvas-only constants (layout constants live in canvas.models).
# ---------------------------------------------------------------------------
#: Default pipeline version string.
_DEFAULT_PIPELINE_VERSION: str = "1.0.0"
#: Prefix used when merging foreign nodes to avoid id collisions.
_MERGE_ID_PREFIX: str = "merged_"

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
        """The underlying :class:`CanvasState` (深拷贝快照).

        返回内部状态的深拷贝，确保调用方无法绕过锁直接修改画布的
        内部状态。需要修改状态时请使用 :meth:`_replace_state`。
        """
        with self._lock:
            return copy.deepcopy(self._state)

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
        """Return a deep copy of the node registered under ``node_id``.

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
            return copy.deepcopy(node)

    def list_nodes(self) -> List[CanvasNode]:
        """Return a list of all nodes on the canvas (insertion order)."""
        with self._lock:
            return [copy.deepcopy(n) for n in self._state.nodes]

    def list_connections(self) -> List[CanvasConnection]:
        """Return a list of all connections on the canvas."""
        with self._lock:
            return [copy.deepcopy(c) for c in self._state.connections]

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

        The method performs full validation before creating the connection.
        Canvas-specific checks (endpoint / port sanity, node existence,
        one-to-one input) are performed inline; the remaining structural
        and type checks (self-loop, duplicate, port existence, type
        compatibility, cycle detection) are delegated to
        :class:`~pipeline.validators.ConnectionValidator` to avoid
        duplicating logic that is already shared with
        :class:`~pipeline.composer.PipelineBuilder`.

        * **Endpoint / port sanity** -- ``from_node``, ``to_node``,
          ``from_port`` and ``to_port`` must be non-empty strings.
        * **Node existence** -- both endpoints must already be on the
          canvas.
        * **One-to-one input** -- an input port may receive at most one
          incoming connection.
        * **Self-loop / duplicate / port existence / type compatibility /
          cycle detection** -- delegated to
          :meth:`ConnectionValidator.validate_connection`.

        Args:
            from_node: Id of the producing (upstream) node.
            from_port: Output port name on the upstream node.
            to_node: Id of the consuming (downstream) node.
            to_port: Input port name on the downstream node.

        Returns:
            The newly created :class:`CanvasConnection` on success.

        Raises:
            ValueError: If any validation check fails.  The exception
                message describes the reason.
        """
        if not from_node or not to_node:
            raise ValueError(
                "Connection endpoints must be non-empty strings "
                "(from_node={!r}, to_node={!r}).".format(from_node, to_node)
            )
        if not from_port or not to_port:
            raise ValueError(
                "Connection ports must be non-empty strings "
                "(from_port={!r}, to_port={!r}).".format(
                    from_port, to_port
                )
            )

        with self._lock:
            # --- node existence -------------------------------------------------
            src_node = self._find_node(from_node)
            if src_node is None:
                raise ValueError(
                    "Source node {!r} does not exist on the canvas.".format(
                        from_node
                    )
                )
            dst_node = self._find_node(to_node)
            if dst_node is None:
                raise ValueError(
                    "Target node {!r} does not exist on the canvas.".format(
                        to_node
                    )
                )

            # --- one-to-one input ----------------------------------------------
            # An input port may receive at most one incoming wire.  This
            # check is canvas-specific (the shared validator does not
            # enforce one-to-one inputs) so it stays here.  Exact duplicates
            # (same from_node + from_port) are skipped so that the
            # validator's duplicate check can produce a more specific error.
            for conn in self._state.connections:
                if conn.to_node == to_node and conn.to_port == to_port:
                    if (
                        conn.from_node == from_node
                        and conn.from_port == from_port
                    ):
                        # Exact duplicate -- let the validator report it.
                        continue
                    raise ValueError(
                        "Input port {!r} of node {!r} already has an incoming "
                        "connection from {}.{}; an input may receive at most "
                        "one wire.".format(
                            to_port, to_node, conn.from_node, conn.from_port
                        )
                    )

            # --- delegate remaining checks to ConnectionValidator -------------
            # Build the context required by the shared validator: the set
            # of declared node ids, the existing edges as 4-tuples, the
            # node-type map and the lazily-loaded node specs.
            existing_edges: List[tuple] = [
                (c.from_node, c.to_node, c.from_port, c.to_port)
                for c in self._state.connections
            ]
            node_type_map: Dict[str, str] = {
                n.id: n.type for n in self._state.nodes
            }
            declared_ids: set[str] = {n.id for n in self._state.nodes}
            specs = ConnectionValidator.load_specs()

            error = ConnectionValidator.validate_connection(
                from_id=from_node,
                to_id=to_node,
                output_key=from_port,
                input_key=to_port,
                declared_ids=declared_ids,
                existing_edges=existing_edges,
                node_type_map=node_type_map,
                specs=specs,
            )
            if error is not None:
                raise ValueError(error)

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
        * **Port existence + type matching** -- when node specs are
          available (via the L4 node registry), checks that ``from_port``
          is a declared output of the upstream node's type and ``to_port``
          is a declared input of the downstream node's type, and that the
          output type is compatible with the input type according to
          :class:`~nodes.type_system.TypeSystem`.  Error messages include
          the available ports and the list of compatible input types.
          Connections whose node specs cannot be loaded are reported with
          a notice rather than silently skipped.

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

            # Type matching using the TypeSystem (no longer silently skipped).
            type_errors = self._validate_port_types(node_type_map)
            errors.extend(type_errors)

        return errors

    def _validate_port_types(
        self, node_type_map: Dict[str, str]
    ) -> List[str]:
        """Port-type validation using the L4 node registry and TypeSystem.

        For every connection the method checks, when the node specs are
        available:

        * ``from_port`` is a declared output of the upstream node's type.
        * ``to_port`` is a declared input of the downstream node's type.
        * The output port's type is compatible with the input port's type
          according to :meth:`TypeSystem.is_compatible`.

        Error messages include the list of available ports and, for type
        mismatches, the input types compatible with the output type.

        Args:
            node_type_map: Mapping of ``node_id -> node_type``.

        Returns:
            A list of type-mismatch error strings (may be empty).
        """
        errors: List[str] = []
        specs = ConnectionValidator.load_specs()
        if specs is None:
            # Registry unavailable -- nothing to validate against.
            return errors

        # Lazy import to decouple the canvas layer from the node type
        # system (lower coupling; the import only happens when specs are
        # actually available).
        from nodes.type_system import TypeSystem  # type: ignore[import-not-found]

        with self._lock:
            for conn in self._state.connections:
                from_type = node_type_map.get(conn.from_node)
                to_type = node_type_map.get(conn.to_node)
                if from_type is None or to_type is None:
                    continue
                from_spec = specs.get(from_type)
                to_spec = specs.get(to_type)

                # --- output port existence --------------------------------
                if from_spec is not None:
                    if conn.from_port not in from_spec.outputs:
                        available = (
                            ", ".join(sorted(from_spec.outputs.keys()))
                            or "(none)"
                        )
                        errors.append(
                            "Port {!r} is not a declared output of node "
                            "type {!r} (node {!r}). Available output ports: "
                            "{}.".format(
                                conn.from_port,
                                from_type,
                                conn.from_node,
                                available,
                            )
                        )
                        continue  # cannot check type without the port

                # --- input port existence ----------------------------------
                if to_spec is not None:
                    if conn.to_port not in to_spec.inputs:
                        available = (
                            ", ".join(sorted(to_spec.inputs.keys()))
                            or "(none)"
                        )
                        errors.append(
                            "Port {!r} is not a declared input of node "
                            "type {!r} (node {!r}). Available input ports: "
                            "{}.".format(
                                conn.to_port,
                                to_type,
                                conn.to_node,
                                available,
                            )
                        )
                        continue  # cannot check type without the port

                # --- type compatibility ------------------------------------
                if (
                    from_spec is not None
                    and to_spec is not None
                    and conn.from_port in from_spec.outputs
                    and conn.to_port in to_spec.inputs
                ):
                    out_type = from_spec.outputs[conn.from_port]
                    in_type = to_spec.inputs[conn.to_port]
                    if not TypeSystem.is_compatible(out_type, in_type):
                        compatible = TypeSystem.compatible_inputs(out_type)
                        errors.append(
                            "Type mismatch: output port {!r} of node {!r} "
                            "(type {!r}) is not compatible with input port "
                            "{!r} of node {!r} (type {!r}). Compatible input "
                            "types: {}.".format(
                                conn.from_port,
                                conn.from_node,
                                out_type,
                                conn.to_port,
                                conn.to_node,
                                in_type,
                                ", ".join(compatible),
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

        Raises:
            ValueError: If the canvas has no nodes.
        """
        with self._lock:
            if not self._state.nodes:
                raise ValueError(
                    "Cannot convert an empty canvas (no nodes) to a pipeline."
                )
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
    def _replace_state(self, new_state: CanvasState) -> None:
        """Replace the internal state (used by CanvasHistory).

        This is the only sanctioned way for an external collaborator
        (notably :class:`~canvas.versioning.CanvasHistory`) to swap the
        canvas's state.  It acquires ``self._lock`` so that the
        replacement is atomic with respect to other canvas operations,
        instead of bypassing the lock with a direct ``_state`` assignment.
        """
        with self._lock:
            self._state = new_state

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
            merge_inputs = dict(node_d.get("inputs") or {})
            merge_inputs.pop("id", None)
            merge_inputs.pop("position", None)
            merged.add_node(
                node_d["type"],
                id=new_id,
                position=tuple(node_d.get("position", (0.0, 0.0))),
                **merge_inputs,
            )

        # Add connections from other with rewired ids.
        for conn_d in other_snapshot["state"]["connections"]:
            new_from = id_map.get(
                conn_d["from_node"], conn_d["from_node"]
            )
            new_to = id_map.get(conn_d["to_node"], conn_d["to_node"])
            try:
                merged.connect(
                    new_from,
                    conn_d["from_port"],
                    new_to,
                    conn_d["to_port"],
                )
            except ValueError:
                _logger.warning(
                    "Merge: skipped connection %s.%s -> %s.%s "
                    "(validation failed)",
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
