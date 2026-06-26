"""Euler with ancestral noise (``sampler="euler_a"``).

Slightly more diverse than the deterministic
:class:`core.schedulers.euler.EulerSampler`; slightly less stable.
The implementation follows the original ``diffusers``
``EulerAncestralDiscreteScheduler`` recipe.
"""
from __future__ import annotations

from typing import Any

import torch

from ._base import _BaseSampler

__all__ = ["EulerAncestralSampler"]


class EulerAncestralSampler(_BaseSampler):
    """Euler with ancestral noise (slightly more diverse, slightly
    less stable than the deterministic Euler).
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
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        x0_pred = (sample - sqrt_one_minus * model_output) / sqrt_alpha.clamp(min=1e-8)
        noise = torch.randn_like(sample)
        sigma = ((1.0 - alpha_prev) / (1.0 - alpha_t)).sqrt() * (
            1.0 - alpha_t / alpha_prev.clamp(min=1e-8)
        ).sqrt()
        return (
            alpha_prev.sqrt() * x0_pred
            + (1.0 - alpha_prev - sigma ** 2).clamp(min=0.0).sqrt() * model_output
            + sigma * noise
        )
