"""First-order DPM-Solver (``sampler="dpm_solver"``).

Single-step but fast — 10 steps already give good quality.  This
is the Lu et al. 2022 ``DPM-Solver-1`` recipe.
"""
from __future__ import annotations

from typing import Any

import torch

from ._base import _BaseSampler

__all__ = ["DPMSolverSampler"]


class DPMSolverSampler(_BaseSampler):
    """First-order DPM-Solver — single-step but fast (10 steps OK).
    """

    def step(
        self, model_output: torch.Tensor, t: torch.Tensor,
        sample: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        alpha_t = self._alpha(t).view(-1, *([1] * (sample.dim() - 1)))
        alpha_prev = self._alpha(
            (t - (self.timesteps[1] - self.timesteps[0])
             if len(self.timesteps) > 1 else t * 0),
        ).clamp(min=1e-8).view(-1, *([1] * (sample.dim() - 1)))
        lambda_t = (alpha_t / (1.0 - alpha_t)).log()
        lambda_prev = (alpha_prev / (1.0 - alpha_prev)).log()
        h = lambda_prev - lambda_t
        x0_pred = (sample - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt().clamp(min=1e-8)
        return (
            alpha_prev.sqrt() * x0_pred
            + (1.0 - alpha_prev).sqrt() * torch.exp(-h) * model_output
        )
