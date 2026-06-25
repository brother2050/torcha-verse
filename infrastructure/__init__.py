"""Infrastructure layer for the TorchaVerse framework.

This package groups the foundational, non-domain-specific utilities used
across the framework: configuration management (including the v0.3.0 tiered
:class:`ConfigCenter`), device/distributed abstractions, checkpoint
lifecycle, logging, error handling, rate limiting, caching, audit logging,
resource budgeting, and multi-source model fetching.

Import policy
-------------
To keep ``import infrastructure`` lightweight and free of import-time side
effects (notably :class:`ConfigCenter` singleton initialization, which reads
the on-disk config tree and writes a run snapshot), every submodule is
imported *lazily* on first attribute access (see :pep:`562`).  The inference
defaults exposed by :mod:`infrastructure.defaults` are themselves lazy, so
neither ``import infrastructure`` nor ``import infrastructure.defaults``
triggers :class:`ConfigCenter` initialization -- only reading an actual
default value does.

Concretely::

    import infrastructure                  # succeeds without touching config
    from infrastructure import ConfigCenter # lazily imports config_center
    from infrastructure.defaults import (
        DIFFUSION_STEPS,
    )                                      # lazily reads config on access
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # config
    "ConfigCenter",
    "get_config",
    # inference defaults
    "DIFFUSION_STEPS",
    "DIFFUSION_GUIDANCE_SCALE",
    "DIFFUSION_SCHEDULER",
    "DIFFUSION_ETA",
    "SAMPLING_TEMPERATURE",
    "SAMPLING_TOP_K",
    "SAMPLING_TOP_P",
    "SAMPLING_REPETITION_PENALTY",
    # device / distributed
    "DeviceManager",
    "DTypePolicy",
    "TensorParallel",
    "PipelineParallel",
    # logging & error helper
    "Logger",
    "get_logger",
    "set_log_level",
    "safe_call",
    # lifecycle utilities
    "CacheStore",
    "CheckpointManager",
    "RateLimiter",
    # audit
    "AuditEvent",
    "AuditLogger",
    "EventType",
    "Severity",
    # resource budget
    "ResourceBudget",
    "BudgetTracker",
    "AllocationHandle",
    "BudgetExceededError",
    "FeasibilityEstimate",
]

#: Mapping of public attribute name -> submodule providing it.  These are
#: imported lazily on first access so that ``import infrastructure`` does not
#: eagerly pull in every submodule (and, in particular, does not trigger
#: :class:`ConfigCenter` initialization).
_LAZY_IMPORTS: dict[str, str] = {
    # config_center
    "ConfigCenter": "config_center",
    "get_config": "config_center",
    # defaults (the defaults module is itself lazy via __getattr__)
    "DIFFUSION_STEPS": "defaults",
    "DIFFUSION_GUIDANCE_SCALE": "defaults",
    "DIFFUSION_SCHEDULER": "defaults",
    "DIFFUSION_ETA": "defaults",
    "SAMPLING_TEMPERATURE": "defaults",
    "SAMPLING_TOP_K": "defaults",
    "SAMPLING_TOP_P": "defaults",
    "SAMPLING_REPETITION_PENALTY": "defaults",
    # device_manager
    "DeviceManager": "device_manager",
    "DTypePolicy": "device_manager",
    "TensorParallel": "device_manager",
    "PipelineParallel": "device_manager",
    # logger
    "Logger": "logger",
    "get_logger": "logger",
    "set_log_level": "logger",
    # safe_call (lightweight error helper)
    "safe_call": "error_helper",
    # cache_store
    "CacheStore": "cache_store",
    # checkpoint_manager
    "CheckpointManager": "checkpoint_manager",
    # rate_limiter
    "RateLimiter": "rate_limiter",
    # audit_log
    "AuditEvent": "audit_log",
    "AuditLogger": "audit_log",
    "EventType": "audit_log",
    "Severity": "audit_log",
    # resource_budget
    "ResourceBudget": "resource_budget",
    "BudgetTracker": "resource_budget",
    "AllocationHandle": "resource_budget",
    "BudgetExceededError": "resource_budget",
    "FeasibilityEstimate": "resource_budget",
    # metrics (v1.0.0 M2b, stdlib fallback shipped in v0.4.2)
    "MetricsRegistry": "metrics",
    "Counter": "metrics",
    "Gauge": "metrics",
    "Histogram": "metrics",
    "METRICS": "metrics",
    "render_prometheus": "metrics",
    # tenant (v1.0.0 M3a skeleton, shipped in v0.4.2)
    "Tenant": "tenant",
    "TenantRegistry": "tenant",
    "TenantNotFoundError": "tenant",
    "with_tenant": "tenant",
    "current_tenant_id": "tenant",
    "default_registry": "tenant",
    # scheduler (v1.0.0 M1 skeleton, shipped in v0.4.2)
    "RuntimeScheduler": "scheduler",
    "InlineScheduler": "scheduler",
    "ThreadPoolScheduler": "scheduler",
    "default_scheduler": "scheduler",
}


def __getattr__(name: str) -> Any:
    """Lazily import infrastructure submodules on first attribute access.

    Args:
        name: The attribute being accessed on the :mod:`infrastructure`
            package.

    Returns:
        The requested attribute, imported from its submodule.

    Raises:
        AttributeError: If ``name`` is not a known public attribute.
    """
    submodule = _LAZY_IMPORTS.get(name)
    if submodule is None:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        )
    module = importlib.import_module("." + submodule, __name__)
    value = getattr(module, name)
    # Cache in the module namespace so subsequent accesses are direct.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names of the package (for ``dir(infrastructure)``)."""
    return sorted(set(globals()) | set(__all__))
