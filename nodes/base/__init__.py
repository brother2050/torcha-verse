"""L4 capability-layer node base classes (v0.6.x).

The ``nodes/base`` sub-package replaces the v0.4.x single
``nodes/base.py`` (850 lines) with focused sub-modules:

* :mod:`._spec`         -- :class:`NodeSpec` (declarative contract).
* :mod:`._context`      -- :class:`NodeContext` and the
  :data:`NodeExecutor` callable alias.
* :mod:`._node`         -- :class:`BaseNode` and the resource-
  estimation coefficients shared across subclasses.
* :mod:`._registry`     -- :class:`NodeRegistry` (the discovery
  facade over :class:`ModuleBus`).
* :mod:`._registry_index` -- module-level ``_NODE_CLASSES`` index,
  its re-entrant lock, and the ``register_node`` decorator.

This ``__init__`` is a thin facade that re-exports the public
classes so callers that wrote ``from nodes.base import NodeSpec``
keep working unchanged.
"""

from __future__ import annotations

from ..type_system import is_optional  # noqa: F401
from ._context import NodeContext, NodeExecutor
from ._node import BaseNode
from ._registry import NodeRegistry
from ._registry_index import _NODE_CLASSES_LOCK, register_node
from ._spec import NodeSpec

__all__ = [
    "NodeSpec",
    "NodeContext",
    "NodeExecutor",
    "BaseNode",
    "NodeRegistry",
    "register_node",
    "is_optional",
]
