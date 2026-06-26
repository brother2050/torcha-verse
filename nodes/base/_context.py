"""Runtime :class:`NodeContext` for L4 nodes and L5 pipelines (v0.6.x).

The :class:`NodeContext` is the *single* execution context in
v0.3.0+: it merges what used to be the L4 node context and the
L5 pipeline context, eliminating the v0.1.x two-class ambiguity.

L4 responsibilities
    Cross-cutting services handed to every node ``execute`` call:
    :class:`ModuleBus`, :class:`AssetStore`, :class:`ResourceBudget`,
    logger, :class:`AuditLogger`, run config, run id.

L5 responsibilities
    Things the pipeline layer needs to share across nodes during
    one run:

    1. *Output store* -- thread-safe ``node_id -> outputs`` map.
    2. *Executor resolution* -- lookup chain for a given
       ``node_type``: explicit ``executors`` dict first, then
       :class:`ModuleBus` (kind ``"node"``), finally ``None``
       (pipeline falls back to passthrough).
    3. *Metadata bag* -- mutable key/value bag (also exposed as
       ``config`` for legacy callers).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from core.module_bus import ModuleBus
from assets.store import AssetStore
from infrastructure.audit_log import AuditLogger
from infrastructure.logger import get_logger
from infrastructure.resource_budget import ResourceBudget

from ._constants import _NODE_KIND, _logger

__all__ = ["NodeExecutor", "NodeContext"]


#: Sentinel returned by :meth:`NodeContext.get_output` when a node
#: has no entry at all -- lets callers distinguish between a stored
#: ``None`` and a missing key.
_MISSING: Any = object()

#: L5 pipeline layer's default maximum worker count.
_DEFAULT_MAX_WORKERS: int = 4

#: Type alias for the executor callable: ``(inputs, ctx) -> outputs``.
#: Defined before :class:`NodeContext` so the field annotation
#: ``executors: Dict[str, NodeExecutor]`` can reference it directly.
NodeExecutor = Callable[[Dict[str, Any], "NodeContext"], Dict[str, Any]]


@dataclass
class NodeContext:
    """L4 + L5 unified runtime context.

    All fields have reasonable defaults, so the class can be
    constructed with no arguments (handy for tests and dry-runs).

    Attributes:
        bus: Module-assembly bus used to resolve dependencies.
        assets: Tiered asset store (``None`` is fine for dry-runs).
        budget: Hard resource budget for the run.
        logger: Diagnostic logger for nodes.
        audit: Security / ops audit logger.
        config: Free-form run config; nodes read defaults from here
            (e.g. ``"default_text_model"``).
        run_id: Unique identifier of the run.
        executors: ``node_type -> callable`` explicit map; each
            callable receives ``(inputs, ctx)`` and returns outputs.
        max_workers: Default maximum worker count for parallel
            execution.  Normalised to at least 1 in
            :meth:`__post_init__`.
        strict_mode: When ``True`` missing executors raise instead
            of falling back to passthrough.
    """

    # --- L4 node-context fields ---
    bus: ModuleBus = field(default_factory=ModuleBus)
    assets: Optional[AssetStore] = None
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    logger: logging.Logger = field(
        default_factory=lambda: get_logger("nodes.context")
    )
    audit: Optional[AuditLogger] = None
    config: Dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    # --- L5 pipeline-layer fields ---
    executors: Dict[str, "NodeExecutor"] = field(default_factory=dict)
    max_workers: int = _DEFAULT_MAX_WORKERS
    strict_mode: bool = False

    def __post_init__(self) -> None:
        # Output store + re-entrant lock guarding it.
        self._lock: threading.RLock = threading.RLock()
        self._outputs: Dict[str, Dict[str, Any]] = {}
        # Normalise max_workers to at least 1.
        self.max_workers = max(1, int(self.max_workers))

    # ------------------------------------------------------------------
    # Output store
    # ------------------------------------------------------------------
    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Record the outputs produced by ``node_id``."""
        with self._lock:
            self._outputs[node_id] = dict(outputs)

    def get_output(
        self, node_id: str, key: Optional[str] = None
    ) -> Any:
        """Return the outputs for ``node_id``.

        Args:
            node_id: The producer node id.
            key: When given, return that specific output key.  When
                ``None``, return the full output dict.

        Returns:
            The requested value (or output dict).  A missing key
            returns ``None``; a missing node raises
            :class:`KeyError`.

        Raises:
            KeyError: If ``node_id`` has no recorded outputs at all.
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
        """Return whether outputs are recorded for ``node_id``."""
        with self._lock:
            return node_id in self._outputs

    def all_outputs(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of every recorded node output."""
        with self._lock:
            return {nid: dict(out) for nid, out in self._outputs.items()}

    def reset_outputs(self) -> None:
        """Clear every recorded output (reset the run state)."""
        with self._lock:
            self._outputs.clear()

    # ------------------------------------------------------------------
    # Executor resolution
    # ------------------------------------------------------------------
    def register_executor(
        self, node_type: str, executor: "NodeExecutor"
    ) -> None:
        """Register an explicit executor for ``node_type``."""
        with self._lock:
            self.executors[node_type] = executor

    def resolve_executor(
        self, node_type: str
    ) -> Optional["NodeExecutor"]:
        """Resolve the executor for ``node_type``.

        Lookup order:

        1. Explicit :attr:`executors` mapping.
        2. :class:`ModuleBus` ``"node"`` kind, when a bus is set.
           If the resolved object is a :class:`BaseNode` instance
           (i.e. has ``execute``), it is wrapped in an adapter
           closure that turns the pipeline signature
           ``(inputs, ctx)`` into the node signature
           ``execute(ctx, **inputs)``.

        Returns:
            The executor callable, or ``None`` when neither
            source has an entry (the pipeline then falls back
            to passthrough).
        """
        # 1. Explicit executors dict.
        with self._lock:
            executor = self.executors.get(node_type)
        if executor is not None:
            return executor

        # 2. ModuleBus "node" kind.
        if self.bus is not None:
            try:
                if self.bus.has(_NODE_KIND, node_type):
                    resolved = self.bus.resolve(_NODE_KIND, node_type)
                    if hasattr(resolved, "execute") and callable(
                        getattr(resolved, "execute")
                    ):
                        def _node_adapter(
                            inputs: Dict[str, Any],
                            ctx: "NodeContext",
                        ) -> Dict[str, Any]:
                            # Prefer _safe_execute (S2-4) for unified
                            # error handling / logging; fall back to
                            # execute to accommodate non-BaseNode items.
                            if hasattr(resolved, "_safe_execute"):
                                return resolved._safe_execute(ctx, **inputs)
                            return resolved.execute(ctx, **inputs)
                        return _node_adapter
                    return resolved
            except Exception:  # pragma: no cover - defensive
                _logger.debug(
                    "ModuleBus lookup %s failed", node_type, exc_info=True
                )
        return None

    def __repr__(self) -> str:
        return (
            "NodeContext(run_id={!r}, bus={!r}, assets={!r}, "
            "budget={!r}, outputs={}, executors={})".format(
                self.run_id,
                self.bus,
                "set" if self.assets is not None else "None",
                self.budget,
                len(self._outputs),
                len(self.executors),
            )
        )
