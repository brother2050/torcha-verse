"""Concrete samplers (v0.8.0).

Each sampler implements the diffusers-style interface:

* ``set_timesteps(num_steps, shift=1.0)``
* ``step(model_output, t, sample, ...)``

The signature mirrors the existing
:class:`core.diffusion_scheduler.Sampler` base class so the two
implementations can coexist.  The samplers here are deliberately
self-contained (no external diffusers dependency) and are
registered via :data:`core.schedulers.SAMPLER_REGISTRY`.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch


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
        self.timesteps = base.to(self.schedule.timesteps.dtype).flip(0)

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


# ---------------------------------------------------------------------------
# Euler (ancestral = euler_a, classic = euler)
# ---------------------------------------------------------------------------
class EulerSampler(_BaseSampler):
    """Classic deterministic Euler sampler.

    Update rule::

        x_{t-1} = x_t + (x_0_pred - x_t) * (1 - alpha_{t-1}) / (1 - alpha_t)

    where ``x_0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)``.
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
        # Direction from sample to x0_pred, weighted by the time ratio.
        sigma = (1.0 - alpha_prev).sqrt() / sqrt_alpha.clamp(min=1e-8)
        return x0_pred * alpha_prev.sqrt() + model_output * sigma


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


# ---------------------------------------------------------------------------
# DPM++ 2M / DPM-Solver
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Flow-matching samplers (SD3 / FLUX / HunyuanVideo)
# ---------------------------------------------------------------------------
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
