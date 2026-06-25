"""Centralized inference defaults -- single source of truth.

All modules must import inference defaults from here instead of
hardcoding values.  This module reads from ``config/inference_config.yaml``
via :class:`ConfigCenter` at import time, so changing the YAML file
automatically updates every consumer.

Fallback values (the second arg to ``get``) only apply when the config
file is absent (e.g. minimal test environments); they mirror the YAML
values so behaviour stays consistent.
"""

from __future__ import annotations

from infrastructure.config_center import ConfigCenter

_cfg = ConfigCenter()

# -- diffusion.* ----------------------------------------------------------
DIFFUSION_STEPS: int = _cfg.get("diffusion.default_steps", 30)
DIFFUSION_GUIDANCE_SCALE: float = _cfg.get("diffusion.default_guidance_scale", 7.5)
DIFFUSION_SCHEDULER: str = _cfg.get("diffusion.scheduler", "dpm_solver")
DIFFUSION_ETA: float = _cfg.get("diffusion.eta", 0.0)

# -- sampling.default.* ---------------------------------------------------
SAMPLING_TEMPERATURE: float = _cfg.get("sampling.default.temperature", 0.7)
SAMPLING_TOP_K: int = _cfg.get("sampling.default.top_k", 50)
SAMPLING_TOP_P: float = _cfg.get("sampling.default.top_p", 0.9)
SAMPLING_REPETITION_PENALTY: float = _cfg.get("sampling.default.repetition_penalty", 1.1)
