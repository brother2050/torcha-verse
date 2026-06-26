"""Common scaffolding for the v0.8 samplers (refactored layout).

This module hosts :class:`_BaseSampler` -- the parent of every
concrete sampler in the v0.8.5 / v0.9.0 refactor.  The
:class:`_BaseSampler` API is intentionally small
(``set_timesteps`` / ``step`` / ``_alpha``) so the
:class:`EulerSampler` / :class:`EulerAncestralSampler` /
:class:`DPMpp2MSampler` / :class:`DPMSolverSampler` /
:class:`FlowMatchEulerSampler` / :class:`FlowMatchHeunSampler`
subclasses can live in their own files.

Backward compatibility: ``core.schedulers.samplers`` still
re-exports every public class from this package so callers
importing via the old path keep working.
"""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["_BaseSampler"]


class _BaseSampler:
    """Common scaffolding for the v0.8 samplers.

    The classes here do *not* need to inherit from the legacy
    :class:`core.diffusion_scheduler.Sampler` to stay independent
    of the ``schedule`` argument shape; the parent class accepts
    anything that exposes ``.betas`` / ``.alphas_cumprod``.
    """

    def __init__(self, schedule: Any) -> None:
        self.schedule = schedule
        self.timesteps: torch.Tensor = torch.tensor(
            [], device=schedule.device,
        )

    # ------------------------------------------------------------------
    def set_timesteps(
        self, num_steps: int, *, shift: float = 1.0,
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if shift <= 0.0:
            raise ValueError(f"shift must be > 0, got {shift}")
        t_max = int(self.schedule.num_train_timesteps)
        base = torch.arange(
            0, num_steps, device=self.schedule.device,
        ).float() * (t_max // num_steps)
        if abs(shift - 1.0) > 1e-6:
            base = shift * base / (1.0 + (shift - 1.0) * base / float(t_max))
            base = base.round().clamp_(0, t_max - 1)
        # ``timesteps`` is consumed by ``gather`` (which needs an
        # int index) and by ``scatter`` paths in some v0.8.5
        # callers.  Always cast back to long.
        self.timesteps = base.to(self.schedule.timesteps.dtype).flip(0).long()

    # ------------------------------------------------------------------
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.schedule.alphas_cumprod.to(t.device).gather(0, t)

    # ------------------------------------------------------------------
    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError  # placeholder-registry: ignore
