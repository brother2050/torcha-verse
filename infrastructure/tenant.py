"""Multi-tenant isolation (v1.0.0 M3a skeleton, shipped in v0.4.2).

Each :class:`Tenant` owns a private :class:`BudgetTracker` and an
isolated :class:`~infrastructure.metrics.MetricsRegistry` namespace.
The :class:`TenantRegistry` is the process-wide index that maps
``tenant_id`` -> :class:`Tenant`; :class:`current_tenant_id` is a
:mod:`contextvars`-backed accessor so request handlers can be
written without an explicit ``tenant_id`` argument at every call
site.

This module does **not** add a network boundary or per-tenant
process; for v0.4.2 it provides the in-process bookkeeping that
the v1.0.0 M3a deliverable will harden (auth, audit log, per-tenant
config overlays, etc.).
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .logger import get_logger
from .metrics import MetricsRegistry
from .resource_budget import ResourceBudget, BudgetTracker

__all__ = [
    "Tenant",
    "TenantRegistry",
    "TenantNotFoundError",
    "current_tenant_id",
    "with_tenant",
    "default_registry",
]


#: Context variable used by :func:`with_tenant` / :func:`current_tenant_id`
#: to thread the active tenant through async / threaded code without an
#: explicit parameter.
_active_tenant: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "torcha_active_tenant", default=None
)


def current_tenant_id() -> Optional[str]:
    """Return the active tenant id (or ``None`` outside ``with_tenant``)."""
    return _active_tenant.get()


def with_tenant(tenant_id: str) -> "_TenantScope":
    """Context manager that binds ``tenant_id`` as the active tenant."""
    return _TenantScope(tenant_id)


@dataclass
class _TenantScope:
    """Result of :func:`with_tenant`; exposes ``__enter__`` / ``__exit__``."""

    tenant_id: str
    _token: Optional[object] = None

    def __enter__(self) -> "_TenantScope":
        if not self.tenant_id:
            raise ValueError("with_tenant() requires a non-empty tenant_id.")
        self._token = _active_tenant.set(self.tenant_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _active_tenant.reset(self._token)  # type: ignore[arg-type]


class TenantNotFoundError(KeyError):
    """Raised by :meth:`TenantRegistry.get` when a tenant is unknown."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(tenant_id)
        self.tenant_id = tenant_id
        self.args = (f"No such tenant: {tenant_id!r}",)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0]


@dataclass
class Tenant:
    """A single tenant's bookkeeping state.

    Attributes:
        tenant_id: Stable, unique identifier (immutable after creation).
        display_name: Human-readable label for logs and dashboards.
        budget: The :class:`ResourceBudget` granted to this tenant.
        budget_tracker: A per-tenant :class:`BudgetTracker` that
            prevents one tenant from starving another.
        metrics: A per-tenant :class:`MetricsRegistry` so counters
            and histograms can be scraped independently.
        tags: Free-form key-value annotations (e.g. ``plan: pro``).
    """

    tenant_id: str
    display_name: str
    budget: ResourceBudget
    budget_tracker: BudgetTracker
    metrics: MetricsRegistry
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__("time").time())

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("Tenant.tenant_id must be a non-empty string.")
        # Tag keys / values are short strings; reject absurdly long
        # inputs to keep the audit log readable.
        for key, value in self.tags.items():
            if not key or not isinstance(key, str):
                raise ValueError(f"Tag keys must be non-empty strings; got {key!r}.")
            if not isinstance(value, str):
                raise ValueError(
                    f"Tag values must be strings; got {type(value).__name__} for {key!r}."
                )
            if len(key) > 64 or len(value) > 256:
                raise ValueError(
                    f"Tag {key!r} too long (key<=64, value<=256)."
                )

    def label(self, metric_name: str) -> str:
        """Return the per-tenant name for ``metric_name``.

        Metric names emitted by the framework are prefixed with
        ``"tenant."`` so the global :data:`METRICS` registry can
        filter by tenant when scraping.
        """
        if not metric_name:
            raise ValueError("metric_name must be a non-empty string.")
        return f"tenant.{self.tenant_id}.{metric_name}"


class TenantRegistry:
    """Process-wide index of :class:`Tenant` objects.

    Lookups are O(1) and protected by a single :class:`threading.RLock`
    so the registry can be safely mutated from request handlers.
    """

    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.tenant.registry")

    def create(
        self,
        tenant_id: Optional[str] = None,
        *,
        display_name: Optional[str] = None,
        budget: Optional[ResourceBudget] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Tenant:
        """Create and register a new :class:`Tenant`.

        Args:
            tenant_id: Optional explicit id; one is auto-generated
                (``"t-" + 12 hex chars``) when omitted.
            display_name: Optional human label; defaults to
                ``tenant_id``.
            budget: :class:`ResourceBudget` for the tenant.  Defaults
                to a small v0.4.x "free tier" budget.
            tags: Optional tag map.
        """
        if tenant_id is None:
            tenant_id = "t-" + uuid.uuid4().hex[:12]
        display_name = display_name or tenant_id
        if budget is None:
            budget = ResourceBudget(
                vram_gb=4.0,
                ram_gb=16.0,
                disk_gb=50.0,
                max_concurrent_models=1,
                max_concurrent_requests=4,
            )
        tags = dict(tags or {})
        tenant = Tenant(
            tenant_id=tenant_id,
            display_name=display_name,
            budget=budget,
            budget_tracker=BudgetTracker(budget),
            metrics=MetricsRegistry(),
            tags=tags,
        )
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"Tenant {tenant_id!r} already exists.")
            self._tenants[tenant_id] = tenant
        self._logger.info(
            "Registered tenant %s (%s) with vram_gb=%.2f",
            tenant_id,
            display_name,
            budget.vram_gb,
        )
        return tenant

    def get(self, tenant_id: str) -> Tenant:
        """Return the tenant registered as ``tenant_id``.

        Raises:
            TenantNotFoundError: if no such tenant exists.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                raise TenantNotFoundError(tenant_id)
            return tenant

    def try_get(self, tenant_id: str) -> Optional[Tenant]:
        """Return the tenant or ``None`` if it is not registered."""
        with self._lock:
            return self._tenants.get(tenant_id)

    def remove(self, tenant_id: str) -> None:
        """Remove the tenant from the registry.

        No-op if the tenant does not exist (so callers can use
        :meth:`remove` from a finally block without checking first).
        """
        with self._lock:
            self._tenants.pop(tenant_id, None)

    def list_ids(self) -> List[str]:
        """Return a sorted list of registered tenant ids."""
        with self._lock:
            return sorted(self._tenants.keys())

    def __contains__(self, tenant_id: object) -> bool:
        if not isinstance(tenant_id, str):
            return False
        with self._lock:
            return tenant_id in self._tenants

    def __len__(self) -> int:
        with self._lock:
            return len(self._tenants)

    def __iter__(self):
        with self._lock:
            return iter(list(self._tenants.values()))


#: Default process-wide :class:`TenantRegistry` used by
#: :func:`default_registry`.  Tests are free to construct their own
#: :class:`TenantRegistry` instead of mutating this global.
_default_registry: TenantRegistry = TenantRegistry()


def default_registry() -> TenantRegistry:
    """Return the process-wide default :class:`TenantRegistry`."""
    return _default_registry
