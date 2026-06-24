"""Canvas data models for the TorchaVerse v0.3.0 architecture (L5).

本模块定义画布层的纯数据模型，与 :class:`~canvas.canvas.Canvas` 的
行为逻辑分离。画布层是 L5 Pipeline 的可视化前端：节点被放置在
二维坐标上，连接在端口之间绘制，整体可被序列化、版本化、共享和自动生成。

数据模型
--------

* :class:`CanvasNode` -- 画布上的单个节点（id、type、position、inputs、size）。
* :class:`CanvasConnection` -- 两个画布节点之间的类型化连线
  （from_port -> to_port）。
* :class:`CanvasState` -- 画布的完整可变状态（节点、连接、视口、缩放）。
  支持往返序列化以及与 :class:`~pipeline.dag.DAG` 的相互转换。

本模块仅依赖 L5 pipeline 层和 Python 标准库，不导入 torch。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from pipeline.dag import DAG, DAGEdge, DAGNode

__all__ = [
    "CanvasNode",
    "CanvasConnection",
    "CanvasState",
]

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
