"""Classic deterministic Euler sampler (v0.9.0 layout).

Update rule::

    x_{t_prev} = sqrt(alpha_prev) * x0_pred
                + sqrt(1 - alpha_prev) * direction

where ``x0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)``
and ``direction = (x_t - x0_pred) / sqrt(1 - alpha_t)``.

This is the workhorse sampler for the v0.8.0 / v0.8.5
``call_diffusion_loop_backend`` default (``sampler="euler"``) and
ships in its own file under the v0.9.0 Sampler × Scheduler
orthogonal layout.
"""
from __future__ import annotations

from typing import Any

import torch

from ._base import _BaseSampler

__all__ = ["EulerSampler"]


class EulerSampler(_BaseSampler):
    """Classic deterministic Euler sampler.

    Update rule (per :class:`_BaseSampler.timesteps` step)::

        x0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        x_{t-dt} = sqrt(alpha_prev) * x0_pred
                 + sqrt(1 - alpha_prev) * eps
    """

    def _prev_t(self, t: torch.Tensor) -> torch.Tensor:
        """Return the previous timestep (``t - dt``) for the current
        step.  ``self.timesteps`` is sorted in **decreasing** order
        (``flip(0)`` of the arange) so ``dt = timesteps[i] - timesteps[i+1]``
        is *positive*.

        We find the index of ``t`` in ``self.timesteps`` and use the
        next entry as the previous timestep.  When ``t`` is the
        last element (i.e. ``t == 0``), the previous timestep is
        zero as well -- this is the standard "absorbing state"
        convention.
        """
        # ``t`` is shape (B,); we work per-batch.
        out = torch.zeros_like(t)
        for b in range(t.shape[0]):
            ti = t[b].item()
            # Linear scan: timesteps are small (<= 1000).
            for i, ts in enumerate(self.timesteps.tolist()):
                if int(ts) == int(ti):
                    if i + 1 < len(self.timesteps):
                        out[b] = self.timesteps[i + 1]
                    else:
                        out[b] = 0
                    break
        return out

    def step(
        self, model_output: torch.Tensor, t: torch.Tensor,
        sample: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        alpha_t = self._alpha(t).view(-1, *([1] * (sample.dim() - 1)))
        prev_t = self._prev_t(t)
        alpha_prev = self._alpha(prev_t).clamp(min=1e-8).view(
            -1, *([1] * (sample.dim() - 1)),
        )
        sqrt_alpha = alpha_t.sqrt().clamp(min=1e-8)
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        x0_pred = (sample - sqrt_one_minus * model_output) / sqrt_alpha
        return (
            alpha_prev.sqrt() * x0_pred
            + (1.0 - alpha_prev).sqrt() * model_output
        )
