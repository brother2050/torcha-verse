"""Scheduler / Sampler registry (v0.8.0).

A package of small, self-contained samplers that follow the
diffusers convention of an orthogonal ``Sampler × Scheduler`` pair:

* **Sampler** = the integration method (Euler / DPM++ 2M / DPM-Solver
  / Flow-Match Euler / Flow-Match Heun).  Implemented in
  :mod:`core.schedulers.samplers`.
* **Scheduler** (TimestepSchedule) = the noise schedule and
  time-step distribution.  Implemented in
  :mod:`core.schedulers.schedules`.

This module exposes a :data:`SAMPLER_REGISTRY` mapping the public
name (used by :func:`call_diffusion_scheduler_backend`) to a
factory that produces a ready-to-use sampler instance bound to a
noise schedule.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .samplers import (
    EulerSampler,
    EulerAncestralSampler,
    DPMpp2MSampler,
    DPMSolverSampler,
    FlowMatchEulerSampler,
    FlowMatchHeunSampler,
)
from .schedules import (
    KarrasSchedule,
    LinearSchedule,
    FlowMatchSchedule,
    NormalSchedule,
)

__all__ = [
    "SAMPLER_REGISTRY",
    "EulerSampler",
    "EulerAncestralSampler",
    "DPMpp2MSampler",
    "DPMSolverSampler",
    "FlowMatchEulerSampler",
    "FlowMatchHeunSampler",
    "KarrasSchedule",
    "LinearSchedule",
    "FlowMatchSchedule",
    "NormalSchedule",
]


def _make(sampler_cls: type, schedule_cls: type) -> Callable[..., Any]:
    """Build a factory that returns a ``(sampler, schedule)`` pair."""
    def factory(
        *,
        num_train_timesteps: int = 1000,
        device: str | Any = "cpu",
        **kwargs: Any,
    ) -> tuple:
        schedule = schedule_cls(
            num_train_timesteps=int(num_train_timesteps),
            device=str(device),
        )
        sampler = sampler_cls(schedule, **kwargs)
        return sampler, schedule
    return factory


SAMPLER_REGISTRY: Dict[str, Callable[..., Any]] = {
    # Classic diffusion (DDPM-style) samplers
    "euler":        _make(EulerSampler, NormalSchedule),
    "euler_a":      _make(EulerAncestralSampler, NormalSchedule),
    "dpmpp_2m":     _make(DPMpp2MSampler, KarrasSchedule),
    "dpm_solver":   _make(DPMSolverSampler, KarrasSchedule),
    # Flow-matching samplers (SD3 / FLUX / HunyuanVideo)
    "flow_match_euler": _make(FlowMatchEulerSampler, FlowMatchSchedule),
    "flow_match_heun":  _make(FlowMatchHeunSampler, FlowMatchSchedule),
}


def build_sampler(name: str, **kwargs: Any) -> Any:
    """Resolve a sampler by name from :data:`SAMPLER_REGISTRY`.

    Args:
        name: One of the keys in :data:`SAMPLER_REGISTRY`.
        **kwargs: Forwarded to the factory.

    Returns:
        A ``(sampler, schedule)`` tuple.
    """
    if name not in SAMPLER_REGISTRY:
        raise KeyError(
            f"Unknown sampler {name!r}. "
            f"Available: {sorted(SAMPLER_REGISTRY.keys())}",
        )
    return SAMPLER_REGISTRY[name](**kwargs)
