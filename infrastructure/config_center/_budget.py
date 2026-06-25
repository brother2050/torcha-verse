"""Configuration-driven :class:`ResourceBudget` builder.

The :meth:`ConfigCenter.resource_budget` accessor is factored out
into this module so the :class:`ConfigCenter` core can stay focused
on load / query / snapshot semantics.  The builder consumes the
``resource_budget`` section of the current configuration and falls
back to sensible defaults for any missing field.

Two lookup patterns are supported, in order of priority:

1. The canonical ResourceBudget field name (e.g. ``vram_gb``).
2. The System-layer ``default_<field>`` convention so that the
   immutable System defaults can seed the budget.

The builder is intentionally pure: it takes a configuration
section (a dict) and returns a :class:`ResourceBudget` instance.
This keeps the module free of any ConfigCenter state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..resource_budget import ResourceBudget

__all__ = ["build_resource_budget"]


def build_resource_budget(
    section: Dict[str, Any],
    logger: logging.Logger,
) -> ResourceBudget:
    """Build a :class:`ResourceBudget` from a config section.

    Args:
        section: The configuration section (typically
            ``cc.get("resource_budget", {})``).
        logger: The logger to emit float-conversion warnings to.

    Returns:
        A :class:`ResourceBudget` instance with all fields populated
        from ``section`` (or sensible defaults).
    """
    if not isinstance(section, dict):
        section = {}

    def _get_float(key: str, default: float) -> float:
        for candidate in (key, f"default_{key}"):
            if candidate in section:
                try:
                    return float(section[candidate])
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "budget[%s] float conversion failed, using default: %s",
                        candidate,
                        exc,
                    )
        return default

    def _get_int(key: str, default: int) -> int:
        for candidate in (key, f"default_{key}"):
            if candidate in section:
                try:
                    return int(section[candidate])
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "budget[%s] int conversion failed, using default: %s",
                        candidate,
                        exc,
                    )
        return default

    return ResourceBudget(
        vram_gb=_get_float("vram_gb", 0.0),
        ram_gb=_get_float("ram_gb", 0.0),
        disk_gb=_get_float("disk_gb", 0.0),
        max_concurrent_models=_get_int("max_concurrent_models", 1),
        max_concurrent_requests=_get_int("max_concurrent_requests", 1),
        kv_cache_gb=_get_float("kv_cache_gb", 0.0),
        activations_gb=_get_float("activations_gb", 0.0),
        offload_to=str(section.get("offload_to", "none")),
    )
