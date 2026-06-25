"""Tests for the v0.4.2 tenant isolation skeleton.

Covers:

* :class:`Tenant` validation (id, tag length, type).
* :class:`TenantRegistry` CRUD (create, get, try_get, remove,
  contains, len, list_ids, iter).
* :func:`with_tenant` / :func:`current_tenant_id` context-manager
  semantics, including the ``None`` default and the "no-op"
  :meth:`TenantRegistry.remove` path.
"""

from __future__ import annotations

import pytest

from infrastructure.metrics import Counter
from infrastructure.tenant import (
    Tenant,
    TenantNotFoundError,
    TenantRegistry,
    current_tenant_id,
    with_tenant,
)
from infrastructure.resource_budget import ResourceBudget


@pytest.fixture
def registry() -> TenantRegistry:
    return TenantRegistry()


# ---------------------------------------------------------------------------
# Tenant validation
# ---------------------------------------------------------------------------
def test_tenant_requires_id() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        Tenant(
            tenant_id="",
            display_name="",
            budget=ResourceBudget(),
            budget_tracker=None,  # type: ignore[arg-type]
            metrics=None,  # type: ignore[arg-type]
        )


def test_tenant_rejects_non_string_tag_values() -> None:
    with pytest.raises(ValueError, match="strings"):
        Tenant(
            tenant_id="t1",
            display_name="t1",
            budget=ResourceBudget(),
            budget_tracker=None,  # type: ignore[arg-type]
            metrics=None,  # type: ignore[arg-type]
            tags={"plan": 42},  # type: ignore[typeddict-item]
        )


def test_tenant_rejects_oversized_tag_values() -> None:
    with pytest.raises(ValueError, match="too long"):
        Tenant(
            tenant_id="t1",
            display_name="t1",
            budget=ResourceBudget(),
            budget_tracker=None,  # type: ignore[arg-type]
            metrics=None,  # type: ignore[arg-type]
            tags={"k": "v" * 1024},
        )


def test_tenant_label_prefixes_metric_name() -> None:
    tenant = Tenant(
        tenant_id="acme",
        display_name="Acme",
        budget=ResourceBudget(),
        budget_tracker=None,  # type: ignore[arg-type]
        metrics=None,  # type: ignore[arg-type]
    )
    assert tenant.label("requests_total") == "tenant.acme.requests_total"
    with pytest.raises(ValueError, match="non-empty string"):
        tenant.label("")


# ---------------------------------------------------------------------------
# TenantRegistry CRUD
# ---------------------------------------------------------------------------
def test_create_assigns_default_id_when_omitted(
    registry: TenantRegistry,
) -> None:
    tenant = registry.create()
    assert tenant.tenant_id.startswith("t-")
    assert len(tenant.tenant_id) == 14  # "t-" + 12 hex
    assert tenant.display_name == tenant.tenant_id
    # Default budget should be the v0.4.x "free tier".
    assert tenant.budget.vram_gb == pytest.approx(4.0)


def test_create_uses_explicit_id(registry: TenantRegistry) -> None:
    tenant = registry.create(
        tenant_id="acme",
        display_name="Acme Corp",
        tags={"plan": "pro"},
    )
    assert tenant.tenant_id == "acme"
    assert tenant.display_name == "Acme Corp"
    assert tenant.tags == {"plan": "pro"}


def test_create_rejects_duplicate_id(registry: TenantRegistry) -> None:
    registry.create(tenant_id="dup")
    with pytest.raises(ValueError, match="already exists"):
        registry.create(tenant_id="dup")


def test_get_unknown_raises(registry: TenantRegistry) -> None:
    with pytest.raises(TenantNotFoundError) as exc:
        registry.get("missing")
    assert exc.value.tenant_id == "missing"


def test_try_get_returns_none_when_missing(
    registry: TenantRegistry,
) -> None:
    assert registry.try_get("missing") is None
    registry.create(tenant_id="present")
    assert registry.try_get("present") is not None


def test_remove_is_noop_when_missing(registry: TenantRegistry) -> None:
    # Should not raise even when the tenant does not exist.
    registry.remove("ghost")
    assert "ghost" not in registry


def test_contains_and_len(registry: TenantRegistry) -> None:
    registry.create(tenant_id="a")
    registry.create(tenant_id="b")
    assert "a" in registry
    assert "missing" not in registry
    # Non-string keys must return ``False`` rather than raising.
    assert 42 not in registry  # type: ignore[operator]
    assert len(registry) == 2


def test_list_ids_returns_sorted_list(registry: TenantRegistry) -> None:
    registry.create(tenant_id="z")
    registry.create(tenant_id="a")
    registry.create(tenant_id="m")
    assert registry.list_ids() == ["a", "m", "z"]


def test_iter_yields_tenants(registry: TenantRegistry) -> None:
    registry.create(tenant_id="a")
    registry.create(tenant_id="b")
    seen = {t.tenant_id for t in registry}
    assert seen == {"a", "b"}


# ---------------------------------------------------------------------------
# Per-tenant BudgetTracker / MetricsRegistry
# ---------------------------------------------------------------------------
def test_per_tenant_budget_tracker_is_isolated(
    registry: TenantRegistry,
) -> None:
    a = registry.create(tenant_id="a")
    b = registry.create(tenant_id="b")
    a_handle = a.budget_tracker.allocate("model", vram_gb=2.0)
    # The other tenant must not see the first tenant's usage.
    assert b.budget_tracker.used()["vram_gb"] == pytest.approx(0.0)
    a.budget_tracker.release(a_handle)


def test_per_tenant_metrics_isolated(registry: TenantRegistry) -> None:
    a = registry.create(tenant_id="a")
    b = registry.create(tenant_id="b")
    counter_a: Counter = a.metrics.counter("hits_total", "Hits")
    counter_b: Counter = b.metrics.counter("hits_total", "Hits")
    counter_a.inc()
    counter_a.inc()
    # The first tenant's metrics are independent of the second's.
    values_a = list(counter_a._values())
    values_b = list(counter_b._values())
    assert values_a[0][1] == pytest.approx(2.0)
    assert values_b == []  # counter_b has no series yet


# ---------------------------------------------------------------------------
# with_tenant / current_tenant_id
# ---------------------------------------------------------------------------
def test_with_tenant_binds_active_id(registry: TenantRegistry) -> None:
    assert current_tenant_id() is None
    with with_tenant("acme"):
        assert current_tenant_id() == "acme"
    assert current_tenant_id() is None


def test_with_tenant_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        with with_tenant(""):
            pass


def test_with_tenant_restores_after_exception(registry: TenantRegistry) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with with_tenant("acme"):
            assert current_tenant_id() == "acme"
            raise RuntimeError("boom")
    assert current_tenant_id() is None
