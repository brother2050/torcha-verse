"""Module-level index, lock and the ``register_node`` decorator.

The :data:`_NODE_CLASSES` index is *not* authoritative -- the
:class:`ModuleBus` is -- but it is consulted by :class:`NodeRegistry`
to retrieve the :class:`NodeSpec` of a registered node without
having to instantiate the node, and as a defensive fallback for
nodes that were only registered in memory (e.g. in a unit test).

The :func:`register_node` decorator is the public API that
domain-specific node modules use to make themselves discoverable
through :class:`ModuleBus` / :class:`NodeRegistry`.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Callable, Dict, Optional

from core.module_bus import ModuleBus, ModuleNotFoundError as _BusNotFoundError

from ._constants import _NODE_KIND, _logger
from ._node import BaseNode
from ._spec import NodeSpec

__all__ = [
    "_NODE_CLASSES",
    "_NODE_CLASSES_LOCK",
    "_register_node_class",
    "_unregister_node_class",
    "register_node",
]


#: Module-level index ``node_type -> node class``.  Populated by
#: :func:`_register_node_class` / :func:`register_node`; used by
#: :class:`NodeRegistry` to retrieve :class:`NodeSpec` objects
#: without instantiating the node.  The :class:`ModuleBus` remains
#: the authoritative discovery surface.
_NODE_CLASSES: Dict[str, type["BaseNode"]] = {}

#: Re-entrant lock guarding :data:`_NODE_CLASSES` so concurrent
#: registration / unregistration / lookup is safe.
_NODE_CLASSES_LOCK: threading.RLock = threading.RLock()


def _register_node_class(
    cls: type["BaseNode"],
    bus: Optional[ModuleBus] = None,
) -> type["BaseNode"]:
    """Register a node class with the bus and the module index.

    Args:
        cls: The :class:`BaseNode` subclass to register.
        bus: Optional explicit :class:`ModuleBus`.  When ``None``
            the process-wide singleton is used.

    Returns:
        The class unchanged (so it can be used as the decorator
        return value).

    Raises:
        TypeError: If ``cls.spec`` is not a :class:`NodeSpec`.
        ValueError: If ``cls.spec.type`` is empty.
    """
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, NodeSpec):
        raise TypeError(
            "{}.spec must be a NodeSpec instance, got {!r}.".format(
                cls.__name__, spec
            )
        )
    if not spec.type:
        raise ValueError(
            "{}.spec.type must be a non-empty string.".format(cls.__name__)
        )

    registry_bus = bus if bus is not None else ModuleBus()
    with _NODE_CLASSES_LOCK:
        _NODE_CLASSES[spec.type] = cls
    registry_bus.register(
        kind=_NODE_KIND,
        name=spec.type,
        factory=cls,
        description=spec.description,
        tags=list(spec.tags),
    )
    _logger.debug(
        "Registered node type=%s class=%s.", spec.type, cls.__name__
    )
    return cls


def _unregister_node_class(
    node_type: str,
    bus: Optional[ModuleBus] = None,
) -> bool:
    """Remove a node class from the bus and the module index.

    The inverse of :func:`_register_node_class`; used by the
    plugin system to unload a plugin's nodes.  After the call the
    node type is no longer discoverable through
    :class:`NodeRegistry` / :class:`ModuleBus`.

    Args:
        node_type: The node type identifier to remove.
        bus: Optional explicit :class:`ModuleBus`.  When ``None``
            the process-wide singleton is used.

    Returns:
        ``True`` if a node was removed from the bus, ``False``
        otherwise.
    """
    registry_bus = bus if bus is not None else ModuleBus()
    existed = registry_bus.unregister(_NODE_KIND, node_type)
    with _NODE_CLASSES_LOCK:
        cls = _NODE_CLASSES.pop(node_type, None)
    if cls is not None:
        _logger.debug(
            "Unregistered node type=%s class=%s.", node_type,
            getattr(cls, "__name__", "?"),
        )
    return existed or cls is not None


def register_node(
    node_type: str,
) -> Callable[[type["BaseNode"]], type["BaseNode"]]:
    """Class decorator that registers a :class:`BaseNode` subclass.

    The decorated class must define a ``spec`` :class:`NodeSpec`.
    The ``node_type`` argument is authoritative: it is written
    back into ``cls.spec.type`` (via :func:`dataclasses.replace`)
    so the spec and the registration key can never drift apart.

    Example::

        @register_node("text_chat")
        class TextNode(BaseNode):
            spec = NodeSpec(type="text_chat", name="Text Chat", ...)

    Args:
        node_type: The unique node type identifier to register
            under.

    Returns:
        A decorator that registers the class and returns it
        unchanged.

    Raises:
        TypeError: If the decorated object has no valid ``spec``.
        ValueError: If ``node_type`` is empty.
    """
    if not isinstance(node_type, str) or not node_type.strip():
        raise ValueError("node_type must be a non-empty string.")

    def decorator(cls: type["BaseNode"]) -> type["BaseNode"]:
        spec = getattr(cls, "spec", None)
        if not isinstance(spec, NodeSpec):
            raise TypeError(
                "@register_node can only decorate classes with a NodeSpec "
                "'spec' attribute; {} has {!r}.".format(cls.__name__, spec)
            )
        if spec.type != node_type:
            cls.spec = replace(spec, type=node_type)
        _register_node_class(cls)
        return cls

    return decorator
