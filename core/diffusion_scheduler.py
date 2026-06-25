"""Diffusion process schedulers for TorchaVerse.

This module provides a pure-PyTorch implementation of the diffusion
sampling pipeline.  It contains:

* :class:`NoiseSchedule` -- Computes the noise schedule (betas, alphas,
  alpha-bar) for linear, cosine, and exponential strategies.
* :class:`BaseSampler` and concrete samplers -- DDPM, DDIM, Euler,
  DPM-Solver, and Consistency Model samplers.
* :class:`GuidanceController` -- Classifier-Free Guidance (CFG) and
  Classifier Guidance wrappers.
* :class:`StepController` -- Adaptive step-count scheduling.
* :class:`DiffusionScheduler` -- The top-level orchestrator that ties
  everything together with a uniform ``add_noise`` / ``step`` /
  ``set_timesteps`` API.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from infrastructure.defaults import (
    DIFFUSION_ETA,
    DIFFUSION_GUIDANCE_SCALE,
    DIFFUSION_SCHEDULER,
    DIFFUSION_STEPS,
)
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "NoiseSchedule",
    "BaseSampler",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "DPMSolverSampler",
    "ConsistencySampler",
    "GuidanceController",
    "StepController",
    "DiffusionScheduler",
    "SAMPLER_REGISTRY",
]


# ---------------------------------------------------------------------------
# NoiseSchedule
# ---------------------------------------------------------------------------
class NoiseSchedule:
    """Compute the forward-process noise schedule.

    The schedule defines how noise is added at each timestep ``t`` via
    ``betas[t]``, ``alphas[t] = 1 - betas[t]``, and
    ``alphas_cumprod[t] = prod(alphas[:t+1])``.

    Supported strategies:

    * ``"linear"`` -- Linear interpolation from ``beta_start`` to
      ``beta_end``.
    * ``"cosine"`` -- Cosine schedule (Nichol & Dhariwal, 2021) that
      provides smoother noise at high timesteps.
    * ``"exponential"`` -- Geometric progression of betas.

    Args:
        num_timesteps: Total number of diffusion timesteps (``T``).
        strategy: One of ``"linear"``, ``"cosine"``, ``"exponential"``.
        beta_start: Starting beta value (linear / exponential).
        beta_end: Ending beta value (linear / exponential).
        device: Device for the computed tensors.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        strategy: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be > 0, got {num_timesteps}.")
        if strategy not in ("linear", "cosine", "exponential"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'linear', 'cosine', "
                f"or 'exponential'."
            )

        self.num_timesteps: int = num_timesteps
        self.strategy: str = strategy
        self.beta_start: float = beta_start
        self.beta_end: float = beta_end

        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        self._betas: torch.Tensor = self._compute_betas().to(self._device)
        self._alphas: torch.Tensor = 1.0 - self._betas
        self._alphas_cumprod: torch.Tensor = torch.cumprod(self._alphas, dim=0)
        self._alphas_cumprod_prev: torch.Tensor = F.pad(
            self._alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # Precompute sqrt terms for efficiency.
        self._sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 - self._alphas_cumprod
        )
        self._sqrt_recip_alphas: torch.Tensor = torch.sqrt(1.0 / self._alphas)

    # ------------------------------------------------------------------
    def _compute_betas(self) -> torch.Tensor:
        """Compute the beta schedule based on the selected strategy."""
        if self.strategy == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32
            )
        if self.strategy == "exponential":
            return torch.exp(
                torch.linspace(
                    math.log(self.beta_start),
                    math.log(self.beta_end),
                    self.num_timesteps,
                    dtype=torch.float32,
                )
            )
        # Cosine schedule.
        max_beta: float = 0.999
        steps = self.num_timesteps + 1
        t = torch.linspace(0, math.pi / 2, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(t) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, max_beta)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def betas(self) -> torch.Tensor:
        """Beta values ``(T,)``."""
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        """Alpha values ``(T,)``."""
        return self._alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        """Cumulative product of alphas ``(T,)``."""
        return self._alphas_cumprod

    @property
    def alphas_cumprod_prev(self) -> torch.Tensor:
        """Cumulative product shifted by one ``(T,)``."""
        return self._alphas_cumprod_prev

    @property
    def sqrt_alphas_cumprod(self) -> torch.Tensor:
        """``sqrt(alpha_bar)`` ``(T,)``."""
        return self._sqrt_alphas_cumprod

    @property
    def sqrt_one_minus_alphas_cumprod(self) -> torch.Tensor:
        """``sqrt(1 - alpha_bar)`` ``(T,)``."""
        return self._sqrt_one_minus_alphas_cumprod

    @property
    def device(self) -> torch.device:
        """The device on which the schedule tensors reside."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "NoiseSchedule":
        """Move all schedule tensors to ``device``."""
        self._device = torch.device(device) if isinstance(device, str) else device
        self._betas = self._betas.to(self._device)
        self._alphas = self._alphas.to(self._device)
        self._alphas_cumprod = self._alphas_cumprod.to(self._device)
        self._alphas_cumprod_prev = self._alphas_cumprod_prev.to(self._device)
        self._sqrt_alphas_cumprod = self._sqrt_alphas_cumprod.to(self._device)
        self._sqrt_one_minus_alphas_cumprod = self._sqrt_one_minus_alphas_cumprod.to(self._device)
        self._sqrt_recip_alphas = self._sqrt_recip_alphas.to(self._device)
        return self

    # ------------------------------------------------------------------
    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``alpha_bar`` at timestep ``t``.

        Args:
            t: Timestep indices (``LongTensor``) of arbitrary shape.

        Returns:
            ``alpha_bar`` values with the same shape as ``t``.
        """
        return self._alphas_cumprod[t]


# ---------------------------------------------------------------------------
# BaseSampler
# ---------------------------------------------------------------------------
class BaseSampler(abc.ABC):
    """Abstract base class for all diffusion samplers.

    Each sampler implements :meth:`step` which performs a single
    denoising step, and :meth:`set_timesteps` which configures the
    sampling schedule.

    Args:
        schedule: The :class:`NoiseSchedule` used by the sampler.
    """

    def __init__(self, schedule: NoiseSchedule) -> None:
        self.schedule: NoiseSchedule = schedule
        self.timesteps: torch.Tensor = torch.tensor([], device=schedule.device)

    # ------------------------------------------------------------------
    def set_timesteps(self, num_steps: int) -> None:
        """Configure the sampling timesteps.

        Args:
            num_steps: Number of denoising steps to perform.  Must be
                ``> 0`` and ``<= schedule.num_timesteps``.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")
        if num_steps > self.schedule.num_timesteps:
            raise ValueError(
                f"num_steps {num_steps} exceeds num_timesteps "
                f"{self.schedule.num_timesteps}."
            )
        # Evenly spaced timesteps, descending (from T-1 to 0).
        step_ratio = self.schedule.num_timesteps // num_steps
        self.timesteps = torch.arange(
            0, num_steps, device=self.schedule.device
        ) * step_ratio
        self.timesteps = self.timesteps.flip(0)  # descending order

    @abc.abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform a single denoising step.

        Args:
            model_output: The model's prediction (noise, velocity, or
                sample depending on the parameterisation).
            t: Current timestep index.
            sample: The current noisy sample ``x_t``.

        Returns:
            The previous sample ``x_{t-1}``.
        """
        ...

    def __len__(self) -> int:
        return len(self.timesteps)


# ---------------------------------------------------------------------------
# DDPM Sampler
# ---------------------------------------------------------------------------
class DDPMSampler(BaseSampler):
    """Denoising Diffusion Probabilistic Model sampler (Ho et al., 2020).

    Implements the standard ancestral sampling with stochastic noise
    injection at each step.
    """

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform one DDPM denoising step.

        The update rule is::

            x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1-alpha_bar_t) * eps)
                      + sigma_t * z

        where ``z ~ N(0, I)`` and ``sigma_t`` is the posterior variance.

        Args:
            model_output: Predicted noise ``eps``.
            t: Timestep index (scalar tensor).
            sample: Current sample ``x_t``.

        Returns:
            The previous sample ``x_{t-1}``.
        """
        t = t.to(self.schedule.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        beta_t = self.schedule.betas[t]
        alpha_t = self.schedule.alphas[t]
        alpha_bar_t = self.schedule.alphas_cumprod[t]
        alpha_bar_prev = self.schedule.alphas_cumprod_prev[t]
        sqrt_recip_alpha = self.schedule._sqrt_recip_alphas[t]

        # Predict x_0.
        pred_x0 = (sample - (1 - alpha_bar_t).sqrt() * model_output) / alpha_bar_t.sqrt()

        # Posterior mean.
        mean = sqrt_recip_alpha * (
            sample - beta_t / (1 - alpha_bar_t).sqrt() * model_output
        )

        # Posterior variance (use the clipped value).
        posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        posterior_var = torch.clamp(posterior_var, min=1e-20)

        # No noise at the last step (t=0).
        is_last = (t == 0).all()
        if is_last:
            return mean

        noise = torch.randn_like(sample)
        return mean + posterior_var.sqrt() * noise


# ---------------------------------------------------------------------------
# DDIM Sampler
# ---------------------------------------------------------------------------
class DDIMSampler(BaseSampler):
    """Denoising Diffusion Implicit Model sampler (Song et al., 2021).

    A deterministic variant of DDPM that enables fast sampling with
    fewer steps.  The ``eta`` parameter controls the stochasticity
    (``eta=0`` is fully deterministic).

    Args:
        schedule: The noise schedule.
        eta: Stochasticity parameter in ``[0, 1]``.  ``0`` is fully
            deterministic (DDIM), ``1`` recovers DDPM.
    """

    def __init__(self, schedule: NoiseSchedule, eta: float = DIFFUSION_ETA) -> None:
        super().__init__(schedule)
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}.")
        self.eta: float = eta

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        prev_timestep: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform one DDIM denoising step.

        Args:
            model_output: Predicted noise ``eps``.
            t: Current timestep index.
            sample: Current sample ``x_t``.
            prev_timestep: Explicit previous timestep.  When ``None``
                the previous timestep in ``self.timesteps`` is used.

        Returns:
            The previous sample ``x_{t-1}``.
        """
        t = t.to(self.schedule.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        alpha_bar_t = self.schedule.alphas_cumprod[t]

        # Determine previous alpha_bar.
        if prev_timestep is not None:
            alpha_bar_prev = self.schedule.alphas_cumprod[prev_timestep.to(self.schedule.device)]
        elif (t > 0).all():
            alpha_bar_prev = self.schedule.alphas_cumprod[t - 1]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # Predict x_0.
        pred_x0 = (sample - (1 - alpha_bar_t).sqrt() * model_output) / alpha_bar_t.sqrt()

        # Direction pointing to x_t.
        dir_xt = (1 - alpha_bar_prev).sqrt() * model_output

        # Deterministic part.
        prev_sample = alpha_bar_prev.sqrt() * pred_x0 + dir_xt

        # Stochastic part (when eta > 0).
        if self.eta > 0:
            sigma = self.eta * (
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            ).sqrt()
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma * noise

        return prev_sample


# ---------------------------------------------------------------------------
# Euler Sampler
# ---------------------------------------------------------------------------
class EulerSampler(BaseSampler):
    """Euler (first-order ODE) sampler for flow-matching / diffusion.

    This is the simplest ODE sampler: it takes a single Euler step in the
    direction of the model output at each timestep.  Commonly used with
    rectified-flow and flow-matching models.
    """

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform one Euler step.

        The update rule is::

            x_{t-1} = x_t + (t_{prev} - t) * v

        where ``v`` is the model-predicted velocity.

        Args:
            model_output: Predicted velocity ``v``.
            t: Current timestep index.
            sample: Current sample ``x_t``.

        Returns:
            The previous sample ``x_{t-1}``.
        """
        t = t.to(self.schedule.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Normalise t to [0, 1].
        t_norm = t.float() / self.schedule.num_timesteps
        t_prev_norm = torch.clamp(t_norm - 1.0 / max(len(self.timesteps), 1), min=0.0)

        dt = t_prev_norm - t_norm
        return sample + dt * model_output


# ---------------------------------------------------------------------------
# DPM-Solver Sampler
# ---------------------------------------------------------------------------
class DPMSolverSampler(BaseSampler):
    """DPM-Solver (multistep) sampler (Lu et al., 2022).

    A high-order ODE solver that achieves better sample quality with
    fewer steps than first-order methods.  This implementation uses the
    second-order (DPM-Solver-2) update.

    Args:
        schedule: The noise schedule.
        order: Solver order (1 or 2).
    """

    def __init__(self, schedule: NoiseSchedule, order: int = 2) -> None:
        super().__init__(schedule)
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}.")
        self.order: int = order
        self._prev_output: Optional[torch.Tensor] = None

    def set_timesteps(self, num_steps: int) -> None:
        """Configure timesteps and reset internal state."""
        super().set_timesteps(num_steps)
        self._prev_output = None

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform one DPM-Solver step.

        Args:
            model_output: Predicted noise ``eps``.
            t: Current timestep index.
            sample: Current sample ``x_t``.

        Returns:
            The previous sample ``x_{t-1}``.
        """
        t = t.to(self.schedule.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        alpha_bar_t = self.schedule.alphas_cumprod[t]
        lambda_t = torch.log(alpha_bar_t)

        # Determine previous lambda.
        if (t > 0).all():
            alpha_bar_prev = self.schedule.alphas_cumprod[t - 1]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        lambda_prev = torch.log(alpha_bar_prev)

        # First-order update.
        x_prev = (alpha_bar_prev / alpha_bar_t).sqrt() * sample - (
            alpha_bar_prev / alpha_bar_t
        ).sqrt() * (torch.exp(-lambda_t) - torch.exp(-lambda_prev)) * model_output

        if self.order == 1 or self._prev_output is None:
            self._prev_output = model_output
            return x_prev

        # Second-order correction.
        prev_t = torch.clamp(t - 1, min=0)
        alpha_bar_prev_prev = self.schedule.alphas_cumprod[prev_t] if (t > 1).all() else torch.ones_like(alpha_bar_t)
        lambda_prev_prev = torch.log(alpha_bar_prev_prev)

        h = lambda_prev - lambda_t
        r = (lambda_prev - lambda_t) / (lambda_prev_prev - lambda_t + 1e-8)

        d1 = model_output - self._prev_output
        x_prev = x_prev - 0.5 * h * d1

        self._prev_output = model_output
        return x_prev


# ---------------------------------------------------------------------------
# Consistency Model Sampler
# ---------------------------------------------------------------------------
class ConsistencySampler(BaseSampler):
    """Consistency Model sampler for single-step generation.

    Consistency models (Song et al., 2023) can generate samples in a
    single forward pass by mapping any point on the ODE trajectory
    directly to the origin (clean data).  Multi-step refinement is also
    supported.

    Args:
        schedule: The noise schedule.
        num_steps: Number of refinement steps (1 for single-step).
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        num_steps: int = 1,
    ) -> None:
        super().__init__(schedule)
        self.num_steps: int = num_steps

    def set_timesteps(self, num_steps: int) -> None:
        """Configure timesteps for consistency sampling."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")
        self.num_steps = num_steps
        # Use a few discrete timesteps for multi-step refinement.
        if num_steps == 1:
            self.timesteps = torch.tensor(
                [self.schedule.num_timesteps - 1], device=self.schedule.device
            )
        else:
            step_ratio = self.schedule.num_timesteps // num_steps
            self.timesteps = torch.arange(
                0, num_steps, device=self.schedule.device
            ) * step_ratio
            self.timesteps = self.timesteps.flip(0)

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform one consistency model step.

        The consistency function maps ``x_t`` directly to ``x_0``.

        Args:
            model_output: The consistency function output (predicted
                ``x_0``).
            t: Current timestep index.
            sample: Current sample ``x_t``.

        Returns:
            The denoised sample ``x_0`` (for single-step) or a
            partially denoised sample (for multi-step).
        """
        t = t.to(self.schedule.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Single-step: model output IS the clean sample.
        if self.num_steps == 1:
            return model_output

        # Multi-step: denoise to x_0, then re-noise to the next timestep.
        pred_x0 = model_output

        # Determine the next timestep.
        idx = (self.timesteps == t[0]).nonzero(as_tuple=True)
        if len(idx) > 0 and idx[0] > 0:
            next_t = self.timesteps[idx[0] - 1]
            alpha_bar_next = self.schedule.alphas_cumprod[next_t]
            # Re-noise from x_0 to the next timestep.
            noise = torch.randn_like(pred_x0)
            return alpha_bar_next.sqrt() * pred_x0 + (1 - alpha_bar_next).sqrt() * noise

        return pred_x0


# ---------------------------------------------------------------------------
# GuidanceController
# ---------------------------------------------------------------------------
class GuidanceController:
    """Control guidance during diffusion sampling.

    Supports Classifier-Free Guidance (CFG) and Classifier Guidance.

    Args:
        guidance_scale: Scale factor for guidance.  ``1.0`` means no
            guidance.
        guidance_type: ``"cfg"`` for classifier-free guidance or
            ``"classifier"`` for classifier guidance.
        classifier: Optional classifier model (for classifier guidance).
    """

    def __init__(
        self,
        guidance_scale: float = DIFFUSION_GUIDANCE_SCALE,
        guidance_type: str = "cfg",
        classifier: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {guidance_scale}.")
        if guidance_type not in ("cfg", "classifier", "none"):
            raise ValueError(
                f"guidance_type must be 'cfg', 'classifier', or 'none', "
                f"got '{guidance_type}'."
            )
        if guidance_type == "classifier" and classifier is None:
            raise ValueError("classifier is required when guidance_type='classifier'.")

        self.guidance_scale: float = guidance_scale
        self.guidance_type: str = guidance_type
        self.classifier: Optional[Callable[..., torch.Tensor]] = classifier

    # ------------------------------------------------------------------
    def apply(
        self,
        model: Callable[..., torch.Tensor],
        sample: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        unconditional_conditioning: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply guidance to a model forward pass.

        For CFG, the model is run twice (conditional and unconditional)
        and the outputs are combined::

            output = uncond + scale * (cond - uncond)

        For classifier guidance, gradients of the classifier log-likelihood
        are added to the model output.

        Args:
            model: The diffusion model callable.
            sample: Current sample ``x_t``.
            t: Current timestep.
            conditioning: Conditional input (e.g. text embeddings).
            unconditional_conditioning: Unconditional input for CFG.
            **kwargs: Additional arguments forwarded to ``model``.

        Returns:
            The guided model output.
        """
        if self.guidance_type == "none" or self.guidance_scale == 1.0:
            return model(sample, t, conditioning, **kwargs)

        if self.guidance_type == "cfg":
            return self._apply_cfg(model, sample, t, conditioning, unconditional_conditioning, **kwargs)

        return self._apply_classifier_guidance(model, sample, t, conditioning, **kwargs)

    def _apply_cfg(
        self,
        model: Callable[..., torch.Tensor],
        sample: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        unconditional_conditioning: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply classifier-free guidance."""
        # Concatenate conditional and unconditional for a single forward.
        if conditioning is not None and unconditional_conditioning is not None:
            batch_size = sample.shape[0]
            # Duplicate the sample for both passes.
            sample_doubled = torch.cat([sample, sample], dim=0)
            t_doubled = torch.cat([t, t], dim=0) if t.dim() > 0 else t
            cond_doubled = torch.cat([unconditional_conditioning, conditioning], dim=0)

            output = model(sample_doubled, t_doubled, cond_doubled, **kwargs)
            uncond_out, cond_out = output.chunk(2, dim=0)
            return uncond_out + self.guidance_scale * (cond_out - uncond_out)

        # Fallback: single pass.
        return model(sample, t, conditioning, **kwargs)

    def _apply_classifier_guidance(
        self,
        model: Callable[..., torch.Tensor],
        sample: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply classifier guidance via gradient ascent."""
        sample = sample.detach().requires_grad_(True)
        model_output = model(sample, t, conditioning, **kwargs)

        assert self.classifier is not None
        log_prob = self.classifier(sample, t)
        grad = torch.autograd.grad(log_prob.sum(), sample)[0]
        return model_output + self.guidance_scale * grad


# ---------------------------------------------------------------------------
# StepController
# ---------------------------------------------------------------------------
class StepController:
    """Adaptive step-count controller.

    Dynamically adjusts the number of sampling steps based on the
    desired quality / speed trade-off.  A simple heuristic is used:
    more steps for higher guidance scales and larger spatial resolutions.

    Args:
        min_steps: Minimum number of steps.
        max_steps: Maximum number of steps.
        base_steps: Default number of steps.
    """

    def __init__(
        self,
        min_steps: int = 5,
        max_steps: int = 100,
        base_steps: int = DIFFUSION_STEPS,
    ) -> None:
        if min_steps <= 0 or max_steps < min_steps:
            raise ValueError("Invalid step bounds.")
        self.min_steps: int = min_steps
        self.max_steps: int = max_steps
        self.base_steps: int = base_steps

    def compute_steps(
        self,
        guidance_scale: float = 1.0,
        resolution: int = 512,
        quality: str = "balanced",
    ) -> int:
        """Compute the recommended number of sampling steps.

        Args:
            guidance_scale: Current guidance scale (higher = more steps).
            resolution: Spatial resolution of the sample.
            quality: ``"fast"``, ``"balanced"``, or ``"high"``.

        Returns:
            The recommended step count.
        """
        quality_multipliers = {"fast": 0.5, "balanced": 1.0, "high": 1.5}
        mult = quality_multipliers.get(quality, 1.0)

        # Scale by guidance (higher guidance benefits from more steps).
        guidance_factor = 1.0 + (guidance_scale - 1.0) * 0.1

        # Scale by resolution (larger images need more steps).
        res_factor = max(1.0, resolution / 512.0)

        steps = int(self.base_steps * mult * guidance_factor * res_factor)
        return max(self.min_steps, min(self.max_steps, steps))


# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------
SAMPLER_REGISTRY: Dict[str, type] = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "euler": EulerSampler,
    "dpm_solver": DPMSolverSampler,
    "consistency": ConsistencySampler,
}


# ---------------------------------------------------------------------------
# DiffusionScheduler
# ---------------------------------------------------------------------------
class DiffusionScheduler:
    """Top-level diffusion orchestrator.

    Combines a :class:`NoiseSchedule`, a :class:`BaseSampler`, and a
    :class:`GuidanceController` into a single, easy-to-use API.  This is
    the primary entry point for diffusion sampling in TorchaVerse.

    Args:
        num_timesteps: Number of diffusion timesteps.
        noise_strategy: Noise schedule strategy.
        sampler_name: Name of the sampler to use.
        guidance_scale: CFG guidance scale.
        eta: Stochasticity parameter (DDIM).
        device: Device for computations.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        noise_strategy: str = "linear",
        sampler_name: str = DIFFUSION_SCHEDULER,
        guidance_scale: float = DIFFUSION_GUIDANCE_SCALE,
        eta: float = DIFFUSION_ETA,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._logger = get_logger(self.__class__.__name__)

        # Build the noise schedule.
        self.schedule: NoiseSchedule = NoiseSchedule(
            num_timesteps=num_timesteps,
            strategy=noise_strategy,
            device=self._device,
        )

        # Build the sampler.
        if sampler_name not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown sampler '{sampler_name}'. Available: "
                f"{', '.join(SAMPLER_REGISTRY.keys())}."
            )
        sampler_cls = SAMPLER_REGISTRY[sampler_name]
        if sampler_name == "ddim":
            self.sampler: BaseSampler = sampler_cls(self.schedule, eta=eta)
        elif sampler_name == "dpm_solver":
            self.sampler = sampler_cls(self.schedule, order=2)
        else:
            self.sampler = sampler_cls(self.schedule)

        # Build the guidance controller.
        self.guidance: GuidanceController = GuidanceController(
            guidance_scale=guidance_scale,
            guidance_type="cfg" if guidance_scale > 1.0 else "none",
        )

        # Step controller.
        self.step_controller: StepController = StepController()

        # Sampling state.
        self.num_inference_steps: int = DIFFUSION_STEPS
        self._logger.debug(
            "DiffusionScheduler initialised: strategy=%s, sampler=%s, "
            "guidance_scale=%.1f, device=%s",
            noise_strategy,
            sampler_name,
            guidance_scale,
            self._device,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_timesteps(self, num_steps: int) -> None:
        """Set the number of inference timesteps.

        Args:
            num_steps: Number of denoising steps.
        """
        self.num_inference_steps = num_steps
        self.sampler.set_timesteps(num_steps)

    def set_guidance_scale(self, scale: float) -> None:
        """Update the guidance scale at runtime.

        Args:
            scale: New guidance scale (``>= 1.0``).
        """
        self.guidance.guidance_scale = scale
        self.guidance.guidance_type = "cfg" if scale > 1.0 else "none"

    @property
    def timesteps(self) -> torch.Tensor:
        """The current sampling timesteps."""
        return self.sampler.timesteps

    @property
    def device(self) -> torch.device:
        """The device used for computations."""
        return self._device

    # ------------------------------------------------------------------
    # Forward (add noise)
    # ------------------------------------------------------------------
    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to ``x`` at timestep ``t`` (forward diffusion).

        Implements ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise``.

        Args:
            x: Clean sample ``x_0``.
            t: Timestep indices (``LongTensor``).
            noise: Optional pre-sampled noise.  When ``None`` random
                Gaussian noise is generated.

        Returns:
            The noisy sample ``x_t``.
        """
        if noise is None:
            noise = torch.randn_like(x)

        t = t.to(self._device)
        sqrt_alpha_bar = self.schedule.sqrt_alphas_cumprod[t].to(x.device)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(x.device)

        # Reshape for broadcasting.
        while sqrt_alpha_bar.dim() < x.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_bar * x + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # Backward (denoise step)
    # ------------------------------------------------------------------
    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute one denoising step.

        Args:
            model_output: The model's prediction.
            t: Current timestep.
            sample: Current noisy sample.
            **kwargs: Additional arguments forwarded to the sampler.

        Returns:
            The previous (less noisy) sample.
        """
        return self.sampler.step(model_output, t, sample, **kwargs)

    # ------------------------------------------------------------------
    # Full sampling loop
    # ------------------------------------------------------------------
    def sample(
        self,
        model: Callable[..., torch.Tensor],
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor] = None,
        unconditional_conditioning: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the full reverse diffusion process.

        Args:
            model: The diffusion model callable with signature
                ``model(sample, t, conditioning) -> output``.
            shape: Shape of the sample to generate (without batch dim
                if ``conditioning`` provides it, otherwise full shape).
            conditioning: Conditional input for guided sampling.
            unconditional_conditioning: Unconditional input for CFG.
            num_steps: Override the number of inference steps.
            generator: Optional RNG for reproducibility.
            **kwargs: Additional arguments forwarded to ``model``.

        Returns:
            The generated sample ``x_0``.
        """
        steps = num_steps or self.num_inference_steps
        self.set_timesteps(steps)

        # Start from pure noise.
        if generator is not None:
            sample = torch.randn(shape, generator=generator, device=self._device, dtype=torch.float32)
        else:
            sample = torch.randn(shape, device=self._device, dtype=torch.float32)

        for i, t in enumerate(self.timesteps):
            t_tensor = torch.tensor([t], device=self._device, dtype=torch.long)

            # Apply guidance.
            model_output = self.guidance.apply(
                model,
                sample,
                t_tensor,
                conditioning,
                unconditional_conditioning,
                **kwargs,
            )

            # Denoise step.
            sample = self.sampler.step(model_output, t_tensor, sample)

        self._logger.info("Diffusion sampling completed in %d steps.", steps)
        return sample

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity for flow-matching parameterisation.

        ``v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * sample``

        Args:
            sample: Clean sample ``x_0``.
            noise: Noise tensor.
            t: Timestep indices.

        Returns:
            The velocity tensor.
        """
        t = t.to(self._device)
        sqrt_alpha_bar = self.schedule.sqrt_alphas_cumprod[t].to(sample.device)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(sample.device)

        while sqrt_alpha_bar.dim() < sample.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_bar * noise - sqrt_one_minus * sample

    def predict_x0_from_eps(
        self,
        sample: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``x_0`` from the noise prediction.

        ``x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)``

        Args:
            sample: Noisy sample ``x_t``.
            eps: Predicted noise.
            t: Timestep indices.

        Returns:
            The predicted clean sample ``x_0``.
        """
        t = t.to(self._device)
        sqrt_alpha_bar = self.schedule.sqrt_alphas_cumprod[t].to(sample.device)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(sample.device)

        while sqrt_alpha_bar.dim() < sample.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return (sample - sqrt_one_minus * eps) / sqrt_alpha_bar.clamp(min=1e-8)
