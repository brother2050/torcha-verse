"""DPM++ 2M (``sampler="dpmpp_2m"``).

Second-order multistep sampler (Lu et al. 2022).  Simplified
implementation suitable for the v0.8 smoke tests — uses the
previous model output as an extra correction term.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ._base import _BaseSampler

__all__ = ["DPMpp2MSampler"]


class DPMpp2MSampler(_BaseSampler):
    """DPM++ 2M (second-order multistep) sampler.

    Simplified implementation suitable for the v0.8 smoke tests —
    uses the previous model output as an extra correction term.
    """

    def __init__(self, schedule: Any) -> None:
        super().__init__(schedule)
        self._prev_model_output: Optional[torch.Tensor] = None

    def step(
        self, model_output: torch.Tensor, t: torch.Tensor,
        sample: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        alpha_t = self._alpha(t).view(-1, *([1] * (sample.dim() - 1)))
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        x0_pred = (sample - sqrt_one_minus * model_output) / sqrt_alpha.clamp(min=1e-8)
        if self._prev_model_output is None:
            # First step: fall back to Euler.
            self._prev_model_output = model_output
            return x0_pred + (sample - x0_pred) * 0.5
        # Multistep correction.
        delta = (model_output - self._prev_model_output) * 0.5
        self._prev_model_output = model_output
        return x0_pred + delta + (sample - x0_pred) * 0.5
