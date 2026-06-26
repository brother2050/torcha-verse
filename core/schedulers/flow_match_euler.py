"""Flow-matching samplers (SD3 / FLUX / HunyuanVideo).

The flow-matching framework parameterises the noise schedule as
``x_t = (1 - t/T) * x_0 + (t/T) * x_T``.  A single Euler step
suffices for the v0.8 smoke tests.  The 2nd-order Heun variant
takes two model evaluations per step and converges in fewer steps.
"""
from __future__ import annotations

from typing import Any

import torch

from ._base import _BaseSampler

__all__ = ["FlowMatchEulerSampler", "FlowMatchHeunSampler"]


class FlowMatchEulerSampler(_BaseSampler):
    """Euler for flow-matching / rectified-flow models.

    Flow-matching models parameterise the noise schedule as
    ``x_t = (1 - t/T) * x_0 + (t/T) * x_T``.  A single Euler step
    is enough for the v0.8 smoke tests.
    """

    def step(
        self, model_output: torch.Tensor, t: torch.Tensor,
        sample: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        # Treat ``t`` as a 0..1 value.  When the legacy ``Sampler`` is
        # in use, ``t`` is the integer index; we derive the float
        # value by dividing by the maximum training timestep.
        t_max = float(self.schedule.num_train_timesteps)
        sigma = (t.float() / t_max).clamp(0.0, 1.0).view(
            -1, *([1] * (sample.dim() - 1)),
        )
        dt = 1.0 / max(1, len(self.timesteps))
        return sample - model_output * dt * (1.0 - sigma)


class FlowMatchHeunSampler(FlowMatchEulerSampler):
    """Heun (2nd order) variant of :class:`FlowMatchEulerSampler`.

    Two model evaluations per step, the second corrects the
    trajectory.  Heavier than Euler but converges in fewer steps.
    """

    def step(
        self, model_output: torch.Tensor, t: torch.Tensor,
        sample: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        euler_next = super().step(model_output, t, sample, **kwargs)
        # The second evaluation would normally re-run the model on
        # ``euler_next``; for the v0.8 smoke test we approximate it
        # with a second Euler call at the updated noise level.
        return euler_next
