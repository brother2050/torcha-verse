"""DAG (Directed Acyclic Graph) core for the TorchaVerse pipeline layer (L5).

This module provides the graph primitives that every :class:`Pipeline` is
built upon:

* :class:`DAGNode` -- a single processing step in the graph, identified by a
  unique ``id`` and typed by ``node_type`` (which maps to a registered node
  executor, e.g. ``"image_txt2img"``).  A node carries its static ``inputs``
  and an explicit list of upstream ``dependencies`` (node ids).
* :class:`DAGEdge` -- a typed data wire between two nodes.  It declares that
  the output produced under ``output_key`` by ``from_node`` should be fed as
  ``input_key`` into ``to_node``.
* :class:`DAG` -- the graph container.  It is thread-safe (a re-entrant lock
  guards every mutation and read) and offers topological sorting, cycle
  detection, validation, Mermaid visualisation, serialisation and the ability
  to compute *parallel groups* (sets of nodes that share no dependency and may
  therefore be executed concurrently).

Design notes
------------
The pipeline layer is a pure *orchestration* layer: it never imports
:mod:`torch` directly.  Nodes are referenced by their string ``node_type``
and resolved lazily at execution time (see :mod:`pipeline.composer`).  This
keeps :mod:`pipeline.dag` importable in any environment, including minimal CI
sandboxes, exactly like :mod:`core.module_bus`.
"""

from __future__ import annotations

import copy
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["DAGNode", "DAGEdge", "DAG"]


# ---------------------------------------------------------------------------
# DAGNode
# ---------------------------------------------------------------------------
@dataclass
class DAGNode:
    """A single processing step in a :class:`DAG`.

    A node is the unit of work in a pipeline.  It is identified by a unique
    ``id`` and typed by ``node_type`` (the registered executor name, e.g.
    ``"image_txt2img"``).  Static arguments live in ``inputs`` while
    ``dependencies`` lists the ids of upstream nodes that must complete before
    this node can run.

    Attributes:
        id: Unique node identifier within a DAG.
        node_type: Executor name resolved at runtime (e.g.
            ``"text_chat"``, ``"image_txt2img"``).
        inputs: Static input arguments for the node.  Values fed from
            upstream nodes via :class:`DAGEdge` are merged into this dict at
            execution time, overriding any colliding static key.
        dependencies: Ids of upstream nodes that must finish first.
    """

    id: str
    node_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this node to a JSON-serialisable dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "inputs": dict(self.inputs),
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGNode":
        """Reconstruct a :class:`DAGNode` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`DAGNode` instance.
        """
        return cls(
            id=d["id"],
            node_type=d["node_type"],
            inputs=dict(d.get("inputs") or {}),
            dependencies=list(d.get("dependencies") or []),
        )

    def __repr__(self) -> str:
        return (
            "DAGNode(id={!r}, node_type={!r}, dependencies={!r})".format(
                self.id, self.node_type, self.dependencies
            )
        )


# ---------------------------------------------------------------------------
# DAGEdge
# ---------------------------------------------------------------------------
@dataclass
class DAGEdge:
    """A typed data wire between two :class:`DAGNode` instances.

    An edge declares that the value produced by ``from_node`` under
    ``output_key`` should be injected as ``input_key`` into ``to_node``.
    Edges are what make a pipeline a *graph* rather than a flat list of
    independent steps.

    Attributes:
        from_node: Id of the producing (upstream) node.
        to_node: Id of the consuming (downstream) node.
        output_key: Key in the upstream node's output dict to read from.
        input_key: Key in the downstream node's inputs dict to write to.
    """

    from_node: str
    to_node: str
    output_key: str
    input_key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this edge to a JSON-serialisable dictionary."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "output_key": self.output_key,
            "input_key": self.input_key,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGEdge":
        """Reconstruct a :class:`DAGEdge` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`DAGEdge` instance.
        """
        return cls(
            from_node=d["from_node"],
            to_node=d["to_node"],
            output_key=d["output_key"],
            input_key=d["input_key"],
        )

    def __repr__(self) -> str:
        return (
            "DAGEdge({}.{} -> {}.{})".format(
                self.from_node, self.output_key, self.to_node, self.input_key
            )
        )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
class DAG:
    """A thread-safe directed acyclic graph of :class:`DAGNode` / :class:`DAGEdge`.

    The graph is the structural backbone of a :class:`Pipeline`.  It stores
    nodes keyed by id and a flat list of edges, and derives adjacency on
    demand.  All mutating and read operations are guarded by a single
    :class:`threading.RLock` so that a DAG may be assembled and inspected
    concurrently from multiple threads.

    Key operations:

    * :meth:`add_node` / :meth:`add_edge` -- populate the graph.
    * :meth:`topological_sort` -- return node ids in dependency order,
      raising :class:`ValueError` if a cycle is detected.
    * :meth:`get_dependencies` / :meth:`get_dependents` -- adjacency queries.
    * :meth:`validate` -- return a list of structural problems (missing
      dependencies, cycles, dangling edge endpoints, duplicate ids).
    * :meth:`parallel_groups` -- partition nodes into layers that may run
      concurrently.
    * :meth:`visualize` -- emit a Mermaid flowchart string.
    * :meth:`to_dict` / :meth:`from_dict` -- round-trip serialisation.

    Example:
        >>> dag = DAG()
        >>> dag.add_node(DAGNode(id="a", node_type="text_chat",
        ...                     inputs={"prompt": "hi"}, dependencies=[]))
        >>> dag.add_node(DAGNode(id="b", node_type="text_chat",
        ...                     dependencies=["a"]))
        >>> dag.add_edge(DAGEdge("a", "b", "text", "prompt"))
        >>> dag.topological_sort()
        ['a', 'b']
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: List[DAGEdge] = []
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def node_ids(self) -> List[str]:
        """Return the ids of all nodes (insertion order)."""
        with self._lock:
            return list(self._nodes.keys())

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        with self._lock:
            return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        with self._lock:
            return len(self._edges)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_node(self, node: DAGNode) -> None:
        """Add a node to the graph.

        Re-adding a node with an existing id replaces the previous one.

        Args:
            node: The :class:`DAGNode` to add.

        Raises:
            ValueError: If ``node.id`` is empty.
            TypeError: If ``node`` is not a :class:`DAGNode`.
        """
        if not isinstance(node, DAGNode):
            raise TypeError("node must be a DAGNode instance.")
        if not node.id or not isinstance(node.id, str):
            raise ValueError("DAGNode.id must be a non-empty string.")
        with self._lock:
            self._nodes[node.id] = node

    def add_edge(self, edge: DAGEdge) -> None:
        """Add an edge to the graph.

        Edges may reference nodes that have not been added yet; such dangling
        references are reported by :meth:`validate` rather than raising here.
        This makes it convenient to declare a graph in any order.

        Args:
            edge: The :class:`DAGEdge` to add.

        Raises:
            TypeError: If ``edge`` is not a :class:`DAGEdge`.
            ValueError: If any endpoint or key is empty.
        """
        if not isinstance(edge, DAGEdge):
            raise TypeError("edge must be a DAGEdge instance.")
        if not edge.from_node or not edge.to_node:
            raise ValueError("DAGEdge endpoints must be non-empty strings.")
        if not edge.output_key or not edge.input_key:
            raise ValueError("DAGEdge keys must be non-empty strings.")
        with self._lock:
            self._edges.append(edge)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> DAGNode:
        """Return the node registered under ``node_id``.

        返回节点的一个 **深拷贝** (S3-9),而非内部存储的直接引用。这样调用
        方无法意外修改 DAG 的内部状态,包括嵌套的可变对象(如 ``inputs``
        字典、``dependencies`` 列表)。

        Args:
            node_id: The node id to look up.

        Returns:
            The :class:`DAGNode` 的深拷贝。

        Raises:
            KeyError: If no node with that id exists.
        """
        with self._lock:
            node = self._nodes.get(node_id)
        if node is None:
            raise KeyError("No DAGNode with id={!r}.".format(node_id))
        # 返回深拷贝,防止调用方意外修改 DAG 内部状态(包括嵌套可变对象)。
        return copy.deepcopy(node)

    def has_node(self, node_id: str) -> bool:
        """Return ``True`` if a node with ``node_id`` exists."""
        with self._lock:
            return node_id in self._nodes

    def get_dependencies(self, node_id: str) -> List[str]:
        """Return the ids of the nodes that ``node_id`` depends on.

        Dependencies are derived from the node's explicit ``dependencies``
        list, augmented by any edge whose ``to_node`` is ``node_id`` (the
        ``from_node`` of such an edge is an implicit dependency).

        Args:
            node_id: The node id to query.

        Returns:
            A de-duplicated, insertion-ordered list of upstream node ids.

        Raises:
            KeyError: If ``node_id`` is not registered.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError("No DAGNode with id={!r}.".format(node_id))
            node = self._nodes[node_id]
            deps: List[str] = list(node.dependencies)
            seen = set(deps)
            for edge in self._edges:
                if edge.to_node == node_id and edge.from_node not in seen:
                    deps.append(edge.from_node)
                    seen.add(edge.from_node)
        return deps

    def get_dependents(self, node_id: str) -> List[str]:
        """Return the ids of the nodes that depend on ``node_id``.

        A node ``X`` depends on ``node_id`` if ``node_id`` appears in ``X``'s
        ``dependencies`` list or if an edge ``node_id -> X`` exists.

        Args:
            node_id: The node id to query.

        Returns:
            A de-duplicated, insertion-ordered list of downstream node ids.
        """
        with self._lock:
            dependents: List[str] = []
            seen: set[str] = set()
            for nid, node in self._nodes.items():
                if node_id in node.dependencies and nid not in seen:
                    dependents.append(nid)
                    seen.add(nid)
            for edge in self._edges:
                if edge.from_node == node_id and edge.to_node not in seen:
                    dependents.append(edge.to_node)
                    seen.add(edge.to_node)
        return dependents

    def get_incoming_edges(self, node_id: str) -> List[DAGEdge]:
        """Return all edges whose ``to_node`` is ``node_id``."""
        with self._lock:
            return [e for e in self._edges if e.to_node == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[DAGEdge]:
        """Return all edges whose ``from_node`` is ``node_id``."""
        with self._lock:
            return [e for e in self._edges if e.from_node == node_id]

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def topological_sort(self) -> List[str]:
        """Return node ids in topological (dependency) order.

        Uses Kahn's algorithm.  Nodes with no remaining dependencies are
        emitted first; ties are broken by insertion order for determinism.

        Returns:
            A list of node ids such that every node appears after all of
            its dependencies.

        Raises:
            ValueError: If the graph contains a cycle (i.e. it is not a
                DAG).  The error message lists the nodes involved in the
                cycle.
        """
        with self._lock:
            nodes_snapshot = list(self._nodes.values())
            # Build reverse adjacency once: to_node -> [from_node]
            # This avoids the O(V*E) inner loop of iterating all edges
            # for each node.
            reverse_adj: Dict[str, List[str]] = defaultdict(list)
            for edge in self._edges:
                reverse_adj[edge.to_node].append(edge.from_node)

            # Build forward adjacency: dep -> dependents, and in-degree.
            adjacency: Dict[str, List[str]] = defaultdict(list)
            in_degree: Dict[str, int] = {n.id: 0 for n in nodes_snapshot}
            for node in nodes_snapshot:
                deps = set(node.dependencies)
                if node.id in reverse_adj:
                    deps.update(reverse_adj[node.id])
                for dep in deps:
                    # Only count deps that exist as nodes; missing deps are
                    # a validation concern, not a topology concern.
                    if dep in in_degree:
                        adjacency[dep].append(node.id)
                        in_degree[node.id] += 1

            # Seed the queue with insertion-ordered zero-in-degree nodes.
            queue: deque[str] = deque(
                n.id for n in nodes_snapshot if in_degree[n.id] == 0
            )
            order: List[str] = []
            while queue:
                current = queue.popleft()
                order.append(current)
                for nxt in adjacency[current]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        queue.append(nxt)

            if len(order) != len(nodes_snapshot):
                remaining = [nid for nid, deg in in_degree.items() if deg > 0]
                raise ValueError(
                    "DAG contains a cycle involving nodes: {}".format(
                        ", ".join(sorted(remaining))
                    )
                )
        return order

    def parallel_groups(self) -> List[List[str]]:
        """Partition nodes into layers that may execute concurrently.

        Each layer (group) contains nodes whose dependencies are all
        satisfied by earlier layers.  Nodes within the same group share no
        dependency relationship and may therefore be executed in parallel.
        Within a group, nodes preserve topological / insertion order.

        Returns:
            A list of lists of node ids, ordered from the first executable
            layer to the last.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        order = self.topological_sort()
        # Build dependency map once (O(V+E)) instead of calling
        # get_dependencies() for each node (O(V*E)).
        with self._lock:
            reverse_adj: Dict[str, set] = {nid: set() for nid in order}
            for node in self._nodes.values():
                if node.id in reverse_adj:
                    reverse_adj[node.id].update(node.dependencies)
            for edge in self._edges:
                if edge.to_node in reverse_adj:
                    reverse_adj[edge.to_node].add(edge.from_node)

        depth: Dict[str, int] = {}
        for nid in order:
            deps = reverse_adj.get(nid, set())
            if not deps:
                depth[nid] = 0
            else:
                missing = [d for d in deps if d not in depth]
                if missing:
                    raise ValueError(
                        "Node '{}' depends on non-existent nodes: {}".format(
                            nid, missing
                        )
                    )
                depth[nid] = max(depth[d] + 1 for d in deps)
        if not depth:
            return []
        max_depth = max(depth.values())
        groups: List[List[str]] = [[] for _ in range(max_depth + 1)]
        for nid in order:
            groups[depth[nid]].append(nid)
        return groups

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of structural validation error messages.

        An empty list means the graph is structurally sound.  The checks
        performed are:

        * **Missing dependencies** -- a node lists a dependency id that is
          not registered as a node.
        * **Dangling edge endpoints** -- an edge references a ``from_node``
          or ``to_node`` that is not registered.
        * **Cycles** -- the graph is not acyclic.
        * **Duplicate edges** -- the same ``(from, to, output_key,
          input_key)`` wire is declared more than once.

        Args:
            None.

        Returns:
            A list of human-readable error strings.
        """
        errors: List[str] = []
        with self._lock:
            node_ids = set(self._nodes.keys())

            # Missing dependencies.
            for node in self._nodes.values():
                for dep in node.dependencies:
                    if dep not in node_ids:
                        errors.append(
                            "Node {!r} depends on missing node {!r}.".format(
                                node.id, dep
                            )
                        )

            # Dangling edge endpoints.
            for edge in self._edges:
                if edge.from_node not in node_ids:
                    errors.append(
                        "Edge references missing from_node {!r}.".format(
                            edge.from_node
                        )
                    )
                if edge.to_node not in node_ids:
                    errors.append(
                        "Edge references missing to_node {!r}.".format(
                            edge.to_node
                        )
                    )

            # Duplicate edges.
            seen_edges: set[tuple[str, str, str, str]] = set()
            for edge in self._edges:
                key = (edge.from_node, edge.to_node, edge.output_key, edge.input_key)
                if key in seen_edges:
                    errors.append(
                        "Duplicate edge {} -> {} ({} -> {}).".format(
                            edge.from_node, edge.to_node,
                            edge.output_key, edge.input_key,
                        )
                    )
                seen_edges.add(key)

        # Cycle detection (operates on a snapshot, so safe outside the lock).
        try:
            self.topological_sort()
        except ValueError as exc:
            errors.append(str(exc))

        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the whole graph to a JSON-serialisable dictionary."""
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges],
            }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAG":
        """Reconstruct a :class:`DAG` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`DAG` instance.
        """
        dag = cls()
        for node_d in d.get("nodes", []):
            dag.add_node(DAGNode.from_dict(node_d))
        for edge_d in d.get("edges", []):
            dag.add_edge(DAGEdge.from_dict(edge_d))
        return dag

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def visualize(self) -> str:
        """Return a Mermaid flowchart representation of the graph.

        Nodes are rendered as boxes labelled ``id (node_type)`` and edges as
        arrows annotated with ``output_key -> input_key``.  The output is a
        self-contained ``graph TD`` block suitable for rendering by any
        Mermaid-compatible viewer.

        Returns:
            A Mermaid flowchart string.
        """
        with self._lock:
            lines: List[str] = ["graph TD"]
            # Emit a node declaration for every node (even isolated ones).
            for node in self._nodes.values():
                safe_id = self._mermaid_id(node.id)
                lines.append(
                    '    {}["{} ({})"]'.format(
                        safe_id, node.id, node.node_type
                    )
                )
            # Emit edges.
            if not self._edges:
                # Add an implicit dependency arrow when no explicit edge
                # exists but a node declares a dependency, so the flowchart
                # still reflects execution order.
                for node in self._nodes.values():
                    for dep in node.dependencies:
                        lines.append(
                            "    {} --> {}".format(
                                self._mermaid_id(dep), self._mermaid_id(node.id)
                            )
                        )
            else:
                for edge in self._edges:
                    label = "{} -> {}".format(edge.output_key, edge.input_key)
                    lines.append(
                        "    {} -->|{}| {}".format(
                            self._mermaid_id(edge.from_node),
                            label,
                            self._mermaid_id(edge.to_node),
                        )
                    )
        return "\n".join(lines)

    @staticmethod
    def _mermaid_id(node_id: str) -> str:
        """Sanitise a node id for use as a Mermaid identifier.

        Mermaid node ids may only contain alphanumerics and underscores;
        any other character is replaced with ``_`` and a ``n`` prefix is
        added when the id would otherwise start with a digit.
        """
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in node_id)
        if safe and safe[0].isdigit():
            safe = "n" + safe
        return safe or "n"

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            return "DAG(nodes={}, edges={})".format(
                len(self._nodes), len(self._edges)
            )

    def __len__(self) -> int:
        return self.node_count
