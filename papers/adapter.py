"""Paper implementation adapters for the TorchaVerse paper integration system.

This module defines the contract that a *paper implementation* must
satisfy in order to be driven by the framework.  A :class:`PaperAdapter`
is a thin, two-method object that knows how to (1) load the model
artifacts declared by a :class:`~papers.spec.PaperSpec` and (2) run an
inference step against the loaded model.

Adapters are registered with the :class:`AdapterRegistry` under the
paper name (or any alias) so that the CLI / pipeline layer can resolve
the concrete implementation class for a given paper at runtime.

Public surface
--------------
* :class:`PaperAdapter` -- abstract base class for paper implementations.
* :class:`AdapterRegistry` -- registry of adapter classes keyed by name.
* :class:`AdapterNotFoundError` -- raised when an adapter is missing.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Type

__all__ = [
    "PaperAdapter",
    "AdapterRegistry",
    "AdapterNotFoundError",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class AdapterNotFoundError(KeyError):
    """Raised when an adapter cannot be resolved by :class:`AdapterRegistry`.

    Args:
        name: The adapter name that was requested.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        message = "No adapter registered for name={!r}.".format(name)
        super().__init__(message)

    def __str__(self) -> str:
        return "AdapterNotFoundError: name={!r}".format(self.name)


# ---------------------------------------------------------------------------
# PaperAdapter
# ---------------------------------------------------------------------------
class PaperAdapter(abc.ABC):
    """Abstract base class for a paper implementation adapter.

    A subclass pins :attr:`paper_name` (the :class:`PaperSpec` name it
    implements) and :attr:`node_type` (the framework node it drives),
    then implements :meth:`load_model` and :meth:`infer`.

    The adapter is intentionally minimal -- it does **not** own the model
    lifecycle beyond a single ``load_model`` / ``infer`` pair.  Caching,
    device placement and resource budgeting are delegated to the runtime
    :class:`~nodes.base.NodeContext` handed to :meth:`load_model`.

    Attributes:
        paper_name: Name of the :class:`PaperSpec` this adapter
            implements (e.g. ``"musetalk"``).
        node_type: Framework node type this adapter drives (e.g.
            ``"dh_lip_sync"``).
    """

    paper_name: str = ""
    node_type: str = ""

    @abc.abstractmethod
    def load_model(self, ctx: Any) -> Any:
        """Load and return the model artifacts for this paper.

        Args:
            ctx: The runtime context (a :class:`~nodes.base.NodeContext`
                or compatible object) providing the module bus, asset
                store, device manager and resource budget.

        Returns:
            An opaque, loaded model handle to be passed to :meth:`infer`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run a single inference step against ``model``.

        Args:
            model: The handle returned by :meth:`load_model`.
            **kwargs: Paper-specific inference inputs.

        Returns:
            A dictionary of named outputs.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{}(paper_name={!r}, node_type={!r})".format(
            type(self).__name__, self.paper_name, self.node_type
        )


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------
class AdapterRegistry:
    """Registry of :class:`PaperAdapter` classes keyed by name.

    Unlike the process-wide :class:`~papers.registry.PaperRegistry`
    singleton, :class:`AdapterRegistry` is a plain instantiable class --
    callers may hold several isolated registries (e.g. one per test
    session).  A module-level :data:`default_registry` singleton is
    provided for the common case.

    Example::

        reg = AdapterRegistry()
        reg.register("musetalk", MuseTalkAdapter)
        cls = reg.get("musetalk")
        adapter = cls()
        model = adapter.load_model(ctx)
        out = adapter.infer(model, audio=...)
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Type[PaperAdapter]] = {}

    # ------------------------------------------------------------------
    def register(self, name: str, adapter: Type[PaperAdapter]) -> None:
        """Register an adapter class under ``name``.

        Re-registering an existing name replaces the previous entry.

        Args:
            name: The lookup key (usually the paper name).
            adapter: A :class:`PaperAdapter` subclass.

        Raises:
            ValueError: If ``name`` is empty.
            TypeError: If ``adapter`` is not a :class:`PaperAdapter`
                subclass.
        """
        key = (name or "").strip()
        if not key:
            raise ValueError("Adapter name must be a non-empty string.")
        if not (isinstance(adapter, type) and issubclass(adapter, PaperAdapter)):
            raise TypeError(
                "adapter must be a subclass of PaperAdapter, got {!r}.".format(
                    adapter
                )
            )
        self._adapters[key] = adapter

    # ------------------------------------------------------------------
    def get(self, name: str) -> Type[PaperAdapter]:
        """Return the adapter class registered for ``name``.

        Args:
            name: The lookup key.

        Returns:
            The :class:`PaperAdapter` subclass.

        Raises:
            AdapterNotFoundError: If no adapter is registered.
        """
        key = (name or "").strip()
        adapter = self._adapters.get(key)
        if adapter is None:
            raise AdapterNotFoundError(name)
        return adapter

    # ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        """Return ``True`` if an adapter is registered for ``name``."""
        return (name or "").strip() in self._adapters

    # ------------------------------------------------------------------
    def list(self) -> List[str]:
        """Return the sorted list of registered adapter names."""
        return sorted(self._adapters.keys())

    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of registered adapters."""
        return len(self._adapters)

    def __repr__(self) -> str:
        return "AdapterRegistry(adapters={})".format(len(self._adapters))


#: Process-wide default adapter registry.
default_registry: AdapterRegistry = AdapterRegistry()
