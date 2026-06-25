"""Centralized inference defaults -- single source of truth.

All modules must import inference defaults from here instead of
hardcoding values.  This module reads from ``config/inference_config.yaml``
via :class:`ConfigCenter` lazily (on first attribute access), so importing
``infrastructure`` does not trigger ConfigCenter initialization.

Fallback values (the second arg to ``get``) only apply when the config
file is absent (e.g. minimal test environments); they mirror the YAML
values so behaviour stays consistent.
"""

from __future__ import annotations

from typing import Any

_LAZY_VALUES: dict[str, Any] | None = None


def _ensure_loaded() -> dict[str, Any]:
    global _LAZY_VALUES
    if _LAZY_VALUES is None:
        from infrastructure.config_center import ConfigCenter

        cfg = ConfigCenter()
        _LAZY_VALUES = {
            "DIFFUSION_STEPS": cfg.get("diffusion.default_steps", 30),
            "DIFFUSION_GUIDANCE_SCALE": cfg.get(
                "diffusion.default_guidance_scale", 7.5
            ),
            "DIFFUSION_SCHEDULER": cfg.get("diffusion.scheduler", "dpm_solver"),
            "DIFFUSION_ETA": cfg.get("diffusion.eta", 0.0),
            "SAMPLING_TEMPERATURE": cfg.get("sampling.default.temperature", 0.7),
            "SAMPLING_TOP_K": cfg.get("sampling.default.top_k", 50),
            "SAMPLING_TOP_P": cfg.get("sampling.default.top_p", 0.9),
            "SAMPLING_REPETITION_PENALTY": cfg.get(
                "sampling.default.repetition_penalty", 1.1
            ),
        }
    return _LAZY_VALUES


def __getattr__(name: str) -> Any:
    values = _ensure_loaded()
    if name in values:
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DIFFUSION_STEPS",
    "DIFFUSION_GUIDANCE_SCALE",
    "DIFFUSION_SCHEDULER",
    "DIFFUSION_ETA",
    "SAMPLING_TEMPERATURE",
    "SAMPLING_TOP_K",
    "SAMPLING_TOP_P",
    "SAMPLING_REPETITION_PENALTY",
]
