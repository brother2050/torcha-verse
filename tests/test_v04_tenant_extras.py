"""Tests for the v0.4.3 Tenant namespace_root + ensure_namespace helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.tenant import Tenant, TenantRegistry
from infrastructure.resource_budget import BudgetTracker, ResourceBudget


def _tracker():
    return BudgetTracker(
        ResourceBudget(
            vram_gb=4.0,
            ram_gb=16.0,
            disk_gb=50.0,
            max_concurrent_models=1,
            max_concurrent_requests=4,
        )
    )


@pytest.fixture
def registry() -> TenantRegistry:
    return TenantRegistry()


# ---------------------------------------------------------------------------
# Tenant.namespace
# ---------------------------------------------------------------------------
def test_namespace_is_none_when_no_root() -> None:
    tenant = Tenant(
        tenant_id="t1",
        display_name="t1",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
    )
    assert tenant.namespace is None


def test_namespace_is_root_slash_tenant_id(tmp_path: Path) -> None:
    tenant = Tenant(
        tenant_id="acme",
        display_name="acme",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
        namespace_root=tmp_path,
    )
    assert tenant.namespace == tmp_path / "acme"


def test_namespace_root_is_normalised_to_path(tmp_path: Path) -> None:
    # Passing a string is supported; ``__post_init__`` should
    # normalise to ``Path`` so callers can rely on ``.joinpath`` /
    # ``.mkdir`` semantics.
    tenant = Tenant(
        tenant_id="acme",
        display_name="acme",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
        namespace_root=str(tmp_path),
    )
    assert isinstance(tenant.namespace_root, Path)


# ---------------------------------------------------------------------------
# Tenant.ensure_namespace
# ---------------------------------------------------------------------------
def test_ensure_namespace_creates_directory(tmp_path: Path) -> None:
    tenant = Tenant(
        tenant_id="acme",
        display_name="acme",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
        namespace_root=tmp_path,
    )
    leaf = tenant.ensure_namespace("assets", "checkpoints")
    assert leaf == tmp_path / "acme" / "assets" / "checkpoints"
    assert leaf.exists()
    assert leaf.is_dir()


def test_ensure_namespace_is_idempotent(tmp_path: Path) -> None:
    tenant = Tenant(
        tenant_id="acme",
        display_name="acme",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
        namespace_root=tmp_path,
    )
    tenant.ensure_namespace("assets")
    # Second call must not raise.
    tenant.ensure_namespace("assets")
    assert (tmp_path / "acme" / "assets").is_dir()


def test_ensure_namespace_returns_none_when_no_root() -> None:
    tenant = Tenant(
        tenant_id="acme",
        display_name="acme",
        budget=ResourceBudget(),
        budget_tracker=_tracker(),
        metrics=None,  # type: ignore[arg-type]
    )
    # ``ensure_namespace`` is a no-op and returns ``None`` so the
    # caller can branch on the result.
    assert tenant.ensure_namespace("assets") is None


# ---------------------------------------------------------------------------
# TenantRegistry.create with namespace_root
# ---------------------------------------------------------------------------
def test_create_with_namespace_root_creates_tenant_directory(
    registry: TenantRegistry, tmp_path: Path
) -> None:
    tenant = registry.create(tenant_id="acme", namespace_root=tmp_path)
    assert tenant.namespace_root == tmp_path
    assert tenant.namespace == tmp_path / "acme"
    # The directory is *not* created eagerly on ``create()``; the
    # caller can opt in via ``ensure_namespace``.
    assert not (tmp_path / "acme").exists()


def test_create_without_namespace_root_keeps_in_memory(
    registry: TenantRegistry,
) -> None:
    tenant = registry.create(tenant_id="acme")
    assert tenant.namespace_root is None
    assert tenant.namespace is None
