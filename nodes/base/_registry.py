"""Discovery + instantiation facade for nodes, backed by :class:`ModuleBus`.

:class:`NodeRegistry` is a thin wrapper over :class:`ModuleBus`
(the v0.3.0+ single assembly point).  Nodes are registered under
the ``"node"`` kind; this class exposes node-centric operations
(``register``, ``get``, ``list``, ``search``) on top of the bus.

Because :class:`ModuleBus` is a process-wide singleton, a freshly
constructed :class:`NodeRegistry` immediately sees every node
that was registered via the :func:`register_node` decorator at
import time.  A custom bus may be supplied (e.g. for an isolated
test bus).
"""

from __future__ import annotations

from typing import List, Optional

from core.module_bus import ModuleBus, ModuleNotFoundError as _BusNotFoundError

from ._constants import _NODE_KIND
from ._node import BaseNode
from ._registry_index import (
    _NODE_CLASSES,
    _NODE_CLASSES_LOCK,
    _register_node_class,
    _unregister_node_class,
)
from ._spec import NodeSpec

__all__ = ["NodeRegistry"]


class NodeRegistry:
    """Discovery and instantiation facade for nodes, backed by ModuleBus.

    Example::

        @register_node("text_chat")
        class TextNode(BaseNode): ...

        registry = NodeRegistry()
        specs = registry.list()           # all registered nodes
        node = registry.get("text_chat")  # a TextNode instance
        hits = registry.search("image")   # nodes mentioning "image"
    """

    def __init__(self, bus: Optional[ModuleBus] = None) -> None:
        self._bus: ModuleBus = bus if bus is not None else ModuleBus()

    @property
    def bus(self) -> ModuleBus:
        """The underlying :class:`ModuleBus` used for resolution."""
        return self._bus

    # ------------------------------------------------------------------
    def register(self, node_class: type[BaseNode]) -> None:
        """Register a node class with this registry's bus."""
        _register_node_class(node_class, bus=self._bus)

    # ------------------------------------------------------------------
    def unregister(self, node_type: str) -> bool:
        """Remove a node from this registry's bus.

        Args:
            node_type: The node type identifier to remove.

        Returns:
            ``True`` if a node was removed, ``False`` otherwise.
        """
        return _unregister_node_class(node_type, bus=self._bus)

    # ------------------------------------------------------------------
    def get(self, node_type: str) -> BaseNode:
        """Return a (cached) instance of the node registered as ``node_type``.

        Resolution delegates to :class:`ModuleBus`; the factory
        (the node class itself) is invoked at most once per type
        and the instance is cached by the bus.  When the node is
        not on the bus the module-level :data:`_NODE_CLASSES`
        index is consulted as a fallback so that nodes
        registered only in-memory are still reachable.

        Args:
            node_type: The node type identifier (e.g.
                ``"image_txt2img"``).

        Returns:
            A :class:`BaseNode` instance.

        Raises:
            KeyError: If no node is registered for ``node_type``.
        """
        try:
            instance = self._bus.resolve(_NODE_KIND, node_type)
            return instance  # type: ignore[return-value]
        except _BusNotFoundError:
            with _NODE_CLASSES_LOCK:
                cls = _NODE_CLASSES.get(node_type)
            if cls is None:
                raise KeyError(
                    "No node registered for type {!r}.".format(node_type)
                )
            return cls()

    # ------------------------------------------------------------------
    def list(self) -> List[NodeSpec]:
        """Return the :class:`NodeSpec` of every registered node.

        The :class:`ModuleBus` is the authoritative discovery
        surface; the module-level :data:`_NODE_CLASSES` index is
        consulted only to retrieve the :class:`NodeSpec` without
        instantiating the node, and as a defensive fallback for
        nodes registered in-memory only.

        Returns:
            A list of :class:`NodeSpec` sorted by ``type``.
        """
        specs: List[NodeSpec] = []
        seen: set[str] = set()

        for module_spec in self._bus.list(_NODE_KIND):
            if module_spec.kind != _NODE_KIND:
                # Skip nested "node.*" namespaces -- only exact "node".
                continue
            with _NODE_CLASSES_LOCK:
                cls = _NODE_CLASSES.get(module_spec.name)
            if cls is not None:
                specs.append(cls.spec)
                seen.add(module_spec.name)

        # Defensive: include any in-memory-only registrations.
        with _NODE_CLASSES_LOCK:
            node_classes_items = list(_NODE_CLASSES.items())
        for node_type, cls in node_classes_items:
            if node_type not in seen:
                specs.append(cls.spec)
                seen.add(node_type)

        specs.sort(key=lambda s: s.type)
        return specs

    # ------------------------------------------------------------------
    def search(self, query: str) -> List[NodeSpec]:
        """Fuzzy-search nodes by type, name, description or tags.

        The match is case-insensitive and matches any node whose
        type, name, description or any tag *contains* the query
        substring.  An empty query returns every node (same as
        :meth:`list`).

        Args:
            query: Substring to search for.

        Returns:
            A list of matching :class:`NodeSpec` sorted by ``type``.
        """
        needle = (query or "").strip().lower()
        if not needle:
            return self.list()

        results: List[NodeSpec] = []
        for spec in self.list():
            haystack = " ".join(
                [
                    spec.type,
                    spec.name,
                    spec.description,
                    " ".join(spec.tags),
                ]
            ).lower()
            if needle in haystack:
                results.append(spec)
        return results

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "NodeRegistry(bus={!r}, nodes={})".format(
            self._bus, len(self.list())
        )
