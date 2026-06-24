"""Unified sampler abstraction layer for TorchaVerse v0.3.0.

This module provides a registry-driven, unified sampler abstraction that
extracts the common sampling logic previously embedded in
:mod:`core.diffusion_scheduler`.  It decouples the *sampling algorithm*
(DDPM, DDIM, Euler, DPM-Solver++, Consistency) from the *noise schedule*
and the *guidance strategy*, exposing a single :meth:`BaseSampler.sample`
entry point.

Key components:

* :class:`SamplerConfig` -- dataclass holding all sampling hyper-parameters.
* :class:`BaseSampler` -- abstract base class with a uniform ``sample``
  API that every concrete sampler implements.
* :class:`DDPMSampler` / :class:`DDIMSampler` / :class:`EulerSampler` /
  :class:`DPMSolverSampler` / :class:`ConsistencySampler` -- concrete
  samplers.
* :class:`SamplerRegistry` -- thread-safe registry backed by
  :class:`~core.module_bus.ModuleBus` for discovery and instantiation.
* :func:`register_sampler` -- decorator for registering sampler classes.

Design notes
------------
The ``noise_scheduler`` argument accepted by :meth:`BaseSampler.sample`
is expected to be a *schedule-like* object exposing at least
``alphas_cumprod`` (a 1-D :class:`torch.Tensor`), ``betas``, ``alphas``
and ``num_timesteps``.  The existing :class:`~core.diffusion_scheduler.NoiseSchedule`
satisfies this contract.  All samplers are device-agnostic: computation
follows the device of the supplied ``latents`` tensor.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import torch

from core.module_bus import ModuleBus
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "SamplerConfig",
    "BaseSampler",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "DPMSolverSampler",
    "ConsistencySampler",
    "SamplerRegistry",
    "register_sampler",
]


# ---------------------------------------------------------------------------
# Module-level constants (no hard-coded magic numbers inside functions)
# ---------------------------------------------------------------------------
#: Default number of denoising steps when :attr:`SamplerConfig.steps` is unset.
_DEFAULT_STEPS: int = 50

#: Default classifier-free guidance scale (``1.0`` = no guidance).
_DEFAULT_GUIDANCE_SCALE: float = 7.5

#: Default DDIM stochasticity parameter (``0`` = deterministic).
_DEFAULT_ETA: float = 0.0

#: Numerical-stability epsilon for division / variance clamping.
_EPSILON: float = 1e-8

#: Minimum value for posterior variance to avoid zero-variance noise.
_MIN_VARIANCE: float = 1e-20

#: ModuleBus kind under which samplers are registered.
_SAMPLER_KIND: str = "sampler"

#: Default version string for sampler registrations.
_SAMPLER_VERSION: str = "1.0.0"

#: Type alias for the progress callback ``(step, total, latents) -> None``.
SamplerCallback = Optional[Callable[[int, int, torch.Tensor], None]]

#: Type alias for a noise-schedule-like object.
NoiseSchedulerLike = Any


# ---------------------------------------------------------------------------
# SamplerConfig
# ---------------------------------------------------------------------------
@dataclass
class SamplerConfig:
    """Configuration for a single sampling run.

    Attributes:
        steps: Number of denoising steps to perform (``> 0``).
        guidance_scale: Classifier-free guidance scale.  ``1.0`` disables
            guidance; values ``> 1`` amplify the conditional direction.
        negative_prompt: Optional negative prompt string (kept for
            reference; the actual negative conditioning tensor is passed
            separately to :meth:`BaseSampler.sample`).
        seed: Optional integer seed for reproducible noise generation.
            When ``None`` the global RNG state is used.
        eta: Stochasticity parameter in ``[0, 1]``.  Only used by the
            DDIM sampler; ``0`` is fully deterministic, ``1`` recovers
            DDPM-level stochasticity.
    """

    steps: int = _DEFAULT_STEPS
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    eta: float = _DEFAULT_ETA

    def __post_init__(self) -> None:
        """Validate configuration fields after dataclass init."""
        if self.steps <= 0:
            raise ValueError(
                "steps must be > 0, got {}.".format(self.steps)
            )
        if self.guidance_scale < 1.0:
            raise ValueError(
                "guidance_scale must be >= 1.0, got {}.".format(
                    self.guidance_scale
                )
            )
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(
                "eta must be in [0, 1], got {}.".format(self.eta)
            )


# ---------------------------------------------------------------------------
# BaseSampler
# ---------------------------------------------------------------------------
class BaseSampler(abc.ABC):
    """Abstract base class for all unified samplers.

    Each concrete sampler implements :meth:`sample` which runs the full
    reverse diffusion (or flow-matching) process from initial latents
    to a clean sample.  Common helpers for timestep scheduling,
    classifier-free guidance, and progress callbacks are provided here
    so that subclasses can focus on the per-step update rule.

    The sampler is stateless between calls; all configuration is passed
    via :meth:`sample` arguments.
    """

    def __init__(self) -> None:
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = self._device_manager.get_device()
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the full sampling loop and return the denoised latents.

        Args:
            model: Diffusion model callable with signature
                ``model(latents, t, cond) -> noise_prediction``.
            latents: Initial noisy latents (typically pure noise).
            noise_scheduler: Schedule-like object providing
                ``alphas_cumprod``, ``betas``, ``alphas`` and
                ``num_timesteps``.
            cond: Conditional embedding (e.g. text embeddings).
            neg_cond: Negative conditional embedding for CFG.  When
                ``None`` or when ``guidance_scale == 1`` guidance is
                disabled.
            config: Sampling configuration.  Defaults to
                :class:`SamplerConfig` defaults when ``None``.
            callback: Optional progress callback
                ``callback(step, total, latents)`` invoked after each
                denoising step.

        Returns:
            The denoised sample tensor with the same shape as
            ``latents``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_config(config: Optional[SamplerConfig]) -> SamplerConfig:
        """Return *config* or a fresh default :class:`SamplerConfig`."""
        return config if config is not None else SamplerConfig()

    def _make_generator(
        self, config: SamplerConfig, device: torch.device
    ) -> Optional[torch.Generator]:
        """Create a seeded :class:`torch.Generator` or ``None``."""
        if config.seed is None:
            return None
        try:
            gen = torch.Generator(device=device)
        except RuntimeError:
            # Fallback for devices that don't support Generator (e.g. MPS).
            gen = torch.Generator()
        gen.manual_seed(config.seed)
        return gen

    @staticmethod
    def _get_schedule_attr(
        noise_scheduler: NoiseSchedulerLike,
        name: str,
        default: Any = None,
    ) -> Any:
        """Safely retrieve an attribute from the noise scheduler."""
        return getattr(noise_scheduler, name, default)

    def _build_timesteps(
        self,
        noise_scheduler: NoiseSchedulerLike,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a descending timestep schedule.

        Timesteps are evenly spaced from ``T-1`` down to ``0`` with
        ``num_steps`` entries, mirroring the logic in
        :class:`~core.diffusion_scheduler.BaseSampler.set_timesteps`.
        """
        num_timesteps: int = int(
            self._get_schedule_attr(noise_scheduler, "num_timesteps", num_steps)
        )
        step_ratio: int = max(1, num_timesteps // num_steps)
        timesteps = (
            torch.arange(0, num_steps, device=device, dtype=torch.long)
            * step_ratio
        )
        return timesteps.flip(0)  # descending: T-1 ... 0

    def _model_forward(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor],
        neg_cond: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the model with optional classifier-free guidance.

        When ``guidance_scale > 1`` and ``neg_cond`` is provided, the
        model is run on a doubled batch (unconditional + conditional)
        and the outputs are combined as::

            output = uncond + scale * (cond - uncond)

        Otherwise a single forward pass is performed.
        """
        if guidance_scale <= 1.0 or neg_cond is None:
            return model(latents, t, cond)

        # Batched CFG: concatenate unconditional and conditional.
        latent_input = torch.cat([latents, latents], dim=0)
        t_input = torch.cat([t, t], dim=0) if t.dim() > 0 else t
        cond_input = torch.cat([neg_cond, cond], dim=0)

        output = model(latent_input, t_input, cond_input)
        uncond_out, cond_out = output.chunk(2, dim=0)
        return uncond_out + guidance_scale * (cond_out - uncond_out)

    @staticmethod
    def _predict_x0(
        sample: torch.Tensor,
        eps: torch.Tensor,
        alpha_bar_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``x_0`` from a noise prediction.

        ``x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)``
        """
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar_t).sqrt()
        # Broadcast to sample shape.
        while sqrt_alpha_bar.dim() < sample.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return (sample - sqrt_one_minus * eps) / sqrt_alpha_bar.clamp(
            min=_EPSILON
        )

    def _invoke_callback(
        self,
        callback: SamplerCallback,
        step: int,
        total: int,
        latents: torch.Tensor,
    ) -> None:
        """Safely invoke the progress callback, swallowing errors."""
        if callback is None:
            return
        try:
            callback(step, total, latents)
        except Exception:  # noqa: BLE001 - callback errors must not abort sampling
            self._logger.debug("Sampler callback raised at step %d.", step)


# ---------------------------------------------------------------------------
# SamplerRegistry (defined before concrete samplers so the decorator works)
# ---------------------------------------------------------------------------
class SamplerRegistry:
    """Thread-safe registry for sampler classes, backed by ModuleBus.

    Samplers are registered as factories under the ``"sampler"`` kind in
    the global :class:`~core.module_bus.ModuleBus`.  The registry
    provides convenient ``register`` / ``get`` / ``list_names`` methods.

    Example:
        >>> reg = SamplerRegistry()
        >>> reg.register("custom", MySampler)
        >>> sampler = reg.get("custom")
        >>> "ddpm" in reg.list_names()
        True
    """

    def __init__(self, bus: Optional[ModuleBus] = None) -> None:
        self._bus: ModuleBus = bus if bus is not None else ModuleBus()
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        sampler_class: Type[BaseSampler],
        version: str = _SAMPLER_VERSION,
        description: str = "",
    ) -> None:
        """Register a sampler class under *name*.

        Args:
            name: Unique sampler name (e.g. ``"ddpm"``).
            sampler_class: A subclass of :class:`BaseSampler`.
            version: Semantic version string.
            description: Human-readable description.

        Raises:
            TypeError: If *sampler_class* is not a :class:`BaseSampler`
                subclass.
            ValueError: If *name* is empty.
        """
        if not name or not name.strip():
            raise ValueError("Sampler name must be a non-empty string.")
        if not (
            isinstance(sampler_class, type)
            and issubclass(sampler_class, BaseSampler)
        ):
            raise TypeError(
                "sampler_class must be a subclass of BaseSampler, got {}.".format(
                    type(sampler_class).__name__
                )
            )

        factory: Callable[..., BaseSampler] = (
            lambda cls=sampler_class: cls()
        )
        self._bus.register(
            kind=_SAMPLER_KIND,
            name=name,
            factory=factory,
            version=version,
            description=description or sampler_class.__doc__ or "",
        )
        self._logger.debug(
            "Registered sampler '%s' -> %s.", name, sampler_class.__name__
        )

    # ------------------------------------------------------------------
    def get(self, name: str) -> BaseSampler:
        """Resolve and return a sampler instance by name.

        The instance is cached by :class:`ModuleBus`; call
        :meth:`invalidate` on the bus to force re-instantiation.

        Args:
            name: Registered sampler name.

        Returns:
            A :class:`BaseSampler` instance.

        Raises:
            ModuleNotFoundError: If no sampler is registered for *name*.
        """
        return self._bus.resolve(_SAMPLER_KIND, name)

    # ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        """Return ``True`` if a sampler is registered for *name*."""
        return self._bus.has(_SAMPLER_KIND, name)

    # ------------------------------------------------------------------
    def list_names(self) -> List[str]:
        """Return a sorted list of registered sampler names."""
        with self._lock:
            return sorted(
                spec.name for spec in self._bus.list(_SAMPLER_KIND)
            )

    # ------------------------------------------------------------------
    def invalidate(self, name: str) -> None:
        """Invalidate the cached instance for *name*."""
        self._bus.invalidate(_SAMPLER_KIND, name)

    def __repr__(self) -> str:
        return "SamplerRegistry(samplers={})".format(self.list_names())


# ---------------------------------------------------------------------------
# Decorator (must precede concrete sampler definitions)
# ---------------------------------------------------------------------------
def register_sampler(
    name: str,
    *,
    version: str = _SAMPLER_VERSION,
    description: str = "",
) -> Callable[[Type[BaseSampler]], Type[BaseSampler]]:
    """Decorator that registers a sampler class with the global registry.

    Usage::

        @register_sampler("ddpm")
        class DDPMSampler(BaseSampler):
            ...

    The decorated class is returned unchanged so it can still be
    instantiated directly.

    Args:
        name: Unique sampler name.
        version: Semantic version string.
        description: Human-readable description.

    Returns:
        A class decorator that registers the sampler and returns it
        unchanged.
    """

    def _decorator(cls: Type[BaseSampler]) -> Type[BaseSampler]:
        SamplerRegistry().register(
            name=name,
            sampler_class=cls,
            version=version,
            description=description,
        )
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# DDPMSampler
# ---------------------------------------------------------------------------
@register_sampler("ddpm")
class DDPMSampler(BaseSampler):
    """Standard DDPM (ancestral) sampler (Ho et al., 2020).

    Implements stochastic ancestral sampling with posterior noise
    injection at every step except the last.
    """

    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the full DDPM reverse process.

        The per-step update is::

            x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1-alpha_bar_t) * eps)
                      + sigma_t * z

        where ``z ~ N(0, I)`` and ``sigma_t`` is the posterior variance.
        """
        cfg = self._resolve_config(config)
        device = latents.device
        generator = self._make_generator(cfg, device)
        timesteps = self._build_timesteps(noise_scheduler, cfg.steps, device)
        total = len(timesteps)

        alphas_cumprod = self._get_schedule_attr(
            noise_scheduler, "alphas_cumprod"
        )
        alphas_cumprod_prev = self._get_schedule_attr(
            noise_scheduler, "alphas_cumprod_prev"
        )
        betas = self._get_schedule_attr(noise_scheduler, "betas")
        alphas = self._get_schedule_attr(noise_scheduler, "alphas")

        sample = latents
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            eps = self._model_forward(
                model, sample, t_tensor, cond, neg_cond, cfg.guidance_scale
            )

            beta_t = betas[t_tensor]
            alpha_t = alphas[t_tensor]
            alpha_bar_t = alphas_cumprod[t_tensor]
            alpha_bar_prev = (
                alphas_cumprod_prev[t_tensor]
                if alphas_cumprod_prev is not None
                else torch.ones_like(alpha_bar_t)
            )
            sqrt_recip_alpha = (1.0 / alpha_t.clamp(min=_EPSILON)).sqrt()

            # Posterior mean.
            mean = sqrt_recip_alpha * (
                sample - beta_t / (1.0 - alpha_bar_t).sqrt() * eps
            )

            # Posterior variance.
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (
                1.0 - alpha_bar_t
            )
            posterior_var = torch.clamp(posterior_var, min=_MIN_VARIANCE)

            is_last = (t_tensor == 0).all()
            if is_last:
                sample = mean
            else:
                noise = torch.randn(
                    sample.shape,
                    generator=generator,
                    device=device,
                    dtype=sample.dtype,
                )
                sample = mean + posterior_var.sqrt() * noise

            self._invoke_callback(callback, i + 1, total, sample)

        self._logger.debug("DDPM sampling completed in %d steps.", total)
        return sample


# ---------------------------------------------------------------------------
# DDIMSampler
# ---------------------------------------------------------------------------
@register_sampler("ddim")
class DDIMSampler(BaseSampler):
    """DDIM sampler (Song et al., 2021).

    A deterministic (when ``eta=0``) variant of DDPM that supports fast
    sampling with fewer steps.  The ``eta`` parameter controls
    stochasticity: ``0`` is fully deterministic, ``1`` recovers DDPM.
    """

    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the full DDIM reverse process.

        The per-step update is::

            x_{t-1} = sqrt(alpha_bar_prev) * x0 + sqrt(1 - alpha_bar_prev) * dir_xt
                      + sigma * z   (when eta > 0)

        where ``dir_xt = sqrt(1 - alpha_bar_prev) * eps`` and
        ``sigma = eta * sqrt((1-alpha_bar_prev)/(1-alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))``.
        """
        cfg = self._resolve_config(config)
        device = latents.device
        generator = self._make_generator(cfg, device)
        timesteps = self._build_timesteps(noise_scheduler, cfg.steps, device)
        total = len(timesteps)

        alphas_cumprod = self._get_schedule_attr(
            noise_scheduler, "alphas_cumprod"
        )

        sample = latents
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            eps = self._model_forward(
                model, sample, t_tensor, cond, neg_cond, cfg.guidance_scale
            )

            alpha_bar_t = alphas_cumprod[t_tensor]

            # Determine previous alpha_bar.
            if i + 1 < total:
                prev_t = timesteps[i + 1]
                alpha_bar_prev = alphas_cumprod[
                    torch.tensor([prev_t], device=device, dtype=torch.long)
                ]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # Predict x_0.
            pred_x0 = self._predict_x0(sample, eps, alpha_bar_t)

            # Direction pointing to x_t.
            dir_xt = (1.0 - alpha_bar_prev).sqrt() * eps

            # Deterministic part.
            prev_sample = alpha_bar_prev.sqrt() * pred_x0 + dir_xt

            # Stochastic part (when eta > 0).
            if cfg.eta > 0:
                sigma = cfg.eta * (
                    (1.0 - alpha_bar_prev)
                    / (1.0 - alpha_bar_t)
                    * (
                        1.0
                        - alpha_bar_t / alpha_bar_prev.clamp(min=_EPSILON)
                    )
                ).sqrt()
                noise = torch.randn(
                    sample.shape,
                    generator=generator,
                    device=device,
                    dtype=sample.dtype,
                )
                prev_sample = prev_sample + sigma * noise

            sample = prev_sample
            self._invoke_callback(callback, i + 1, total, sample)

        self._logger.debug("DDIM sampling completed in %d steps.", total)
        return sample


# ---------------------------------------------------------------------------
# EulerSampler
# ---------------------------------------------------------------------------
@register_sampler("euler")
class EulerSampler(BaseSampler):
    """Euler (first-order ODE) sampler for flow-matching / rectified flow.

    Takes a single Euler step in the direction of the model-predicted
    velocity at each timestep.  Commonly used with flow-matching and
    rectified-flow models where the model predicts velocity ``v``.
    """

    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the full Euler ODE integration.

        The per-step update is::

            x_{t-1} = x_t + (t_prev - t) * v

        where ``v`` is the model-predicted velocity and ``t`` is
        normalised to ``[0, 1]``.
        """
        cfg = self._resolve_config(config)
        device = latents.device
        timesteps = self._build_timesteps(noise_scheduler, cfg.steps, device)
        total = len(timesteps)

        num_timesteps: int = int(
            self._get_schedule_attr(noise_scheduler, "num_timesteps", total)
        )

        sample = latents
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            velocity = self._model_forward(
                model, sample, t_tensor, cond, neg_cond, cfg.guidance_scale
            )

            # Normalise t to [0, 1].
            t_norm = t_tensor.float() / max(num_timesteps, 1)
            if i + 1 < total:
                t_prev_norm = timesteps[i + 1].float() / max(
                    num_timesteps, 1
                )
            else:
                t_prev_norm = torch.zeros_like(t_norm)

            dt = t_prev_norm - t_norm
            # Broadcast dt to sample shape.
            while dt.dim() < sample.dim():
                dt = dt.unsqueeze(-1)

            sample = sample + dt * velocity
            self._invoke_callback(callback, i + 1, total, sample)

        self._logger.debug("Euler sampling completed in %d steps.", total)
        return sample


# ---------------------------------------------------------------------------
# DPMSolverSampler (DPM-Solver++ 2M)
# ---------------------------------------------------------------------------
@register_sampler("dpm_solver")
class DPMSolverSampler(BaseSampler):
    """DPM-Solver++ (2M) sampler (Lu et al., 2022).

    A high-order ODE solver that achieves better sample quality with
    fewer steps than first-order methods.  This implementation uses the
    multistep second-order (2M) update in the data-prediction (``++``)
    parameterisation.

    The solver maintains the previous step's ``x_0`` prediction to
    compute a second-order correction.  On the first step only the
    first-order update is used.
    """

    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the full DPM-Solver++ 2M reverse process.

        First-order update::

            x_prev = (sigma_prev / sigma_t) * x_t + alpha_prev * (1 - e^{-h}) * x0

        Second-order (2M) correction::

            x_prev -= sigma_prev * Omega2 * (x0_cur - x0_prev)

        where ``h`` is the current log-SNR step, ``r1`` is the ratio of
        the previous step to the current step, and ``Omega2`` is the
        second-order coefficient.
        """
        cfg = self._resolve_config(config)
        device = latents.device
        timesteps = self._build_timesteps(noise_scheduler, cfg.steps, device)
        total = len(timesteps)

        alphas_cumprod = self._get_schedule_attr(
            noise_scheduler, "alphas_cumprod"
        )

        def _lambda(alpha_bar: torch.Tensor) -> torch.Tensor:
            """Log half-SNR: ``lambda = 0.5 * log(alpha_bar / (1 - alpha_bar))``."""
            return 0.5 * torch.log(
                alpha_bar / (1.0 - alpha_bar).clamp(min=_EPSILON)
            )

        def _alpha_sigma(
            alpha_bar: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return ``(alpha, sigma)`` = ``(sqrt(alpha_bar), sqrt(1-alpha_bar))``."""
            return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()

        def _broadcast(val: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            """Broadcast *val* to match the dimensionality of *ref*."""
            while val.dim() < ref.dim():
                val = val.unsqueeze(-1)
            return val

        sample = latents
        prev_x0: Optional[torch.Tensor] = None
        prev_lambda: Optional[torch.Tensor] = None

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            eps = self._model_forward(
                model, sample, t_tensor, cond, neg_cond, cfg.guidance_scale
            )

            alpha_bar_t = alphas_cumprod[t_tensor]
            alpha_t, sigma_t = _alpha_sigma(alpha_bar_t)
            lambda_t = _lambda(alpha_bar_t)

            # Predict x_0.
            x0 = self._predict_x0(sample, eps, alpha_bar_t)

            # Determine target (previous) alpha_bar.
            if i + 1 < total:
                prev_t = timesteps[i + 1]
                alpha_bar_prev = alphas_cumprod[
                    torch.tensor([prev_t], device=device, dtype=torch.long)
                ]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            alpha_prev, sigma_prev = _alpha_sigma(alpha_bar_prev)
            lambda_prev = _lambda(alpha_bar_prev)

            # Current step in log-SNR space (h > 0 since prev is less noisy).
            h = lambda_prev - lambda_t

            # First-order update.
            omega1 = 1.0 - torch.exp(-h)
            ratio = sigma_prev / sigma_t.clamp(min=_EPSILON)

            ratio_b = _broadcast(ratio, sample)
            omega1_b = _broadcast(omega1, sample)
            alpha_prev_b = _broadcast(alpha_prev, sample)

            prev_sample = ratio_b * sample + alpha_prev_b * omega1_b * x0

            # Second-order (2M) correction when previous x0 is available.
            if prev_x0 is not None and prev_lambda is not None:
                h_prev = lambda_t - prev_lambda  # previous step (> 0)
                r1 = h_prev / h.clamp(min=_EPSILON)
                # Omega2 = (1 / r1) * (e^{-h} - 1 + h) / h^2
                omega2 = (
                    (1.0 / r1.clamp(min=_EPSILON))
                    * (torch.exp(-h) - 1.0 + h)
                    / (h * h).clamp(min=_EPSILON)
                )
                omega2_b = _broadcast(omega2, sample)
                sigma_prev_b = _broadcast(sigma_prev, sample)
                prev_sample = prev_sample - sigma_prev_b * omega2_b * (
                    x0 - prev_x0
                )

            # Update state.
            prev_x0 = x0
            prev_lambda = lambda_t
            sample = prev_sample

            self._invoke_callback(callback, i + 1, total, sample)

        self._logger.debug(
            "DPM-Solver++ 2M sampling completed in %d steps.", total
        )
        return sample


# ---------------------------------------------------------------------------
# ConsistencySampler
# ---------------------------------------------------------------------------
@register_sampler("consistency")
class ConsistencySampler(BaseSampler):
    """Consistency Model sampler (Song et al., 2023).

    Consistency models can generate samples in a single forward pass by
    mapping any point on the ODE trajectory directly to the origin
    (clean data).  Multi-step refinement is also supported: the model
    denoises to ``x_0``, then re-noises to the next timestep and
    repeats.
    """

    def sample(
        self,
        model: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        noise_scheduler: NoiseSchedulerLike,
        cond: Optional[torch.Tensor] = None,
        neg_cond: Optional[torch.Tensor] = None,
        config: Optional[SamplerConfig] = None,
        callback: SamplerCallback = None,
    ) -> torch.Tensor:
        """Run the consistency-model sampling loop.

        For single-step (``config.steps == 1``) the model output is
        returned directly as ``x_0``.  For multi-step, each step
        denoises to ``x_0`` and re-noises to the next timestep.
        """
        cfg = self._resolve_config(config)
        device = latents.device
        generator = self._make_generator(cfg, device)
        timesteps = self._build_timesteps(noise_scheduler, cfg.steps, device)
        total = len(timesteps)

        alphas_cumprod = self._get_schedule_attr(
            noise_scheduler, "alphas_cumprod"
        )

        sample = latents
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            # The consistency function output IS the predicted x_0.
            x0 = self._model_forward(
                model, sample, t_tensor, cond, neg_cond, cfg.guidance_scale
            )

            is_last = i + 1 >= total
            if is_last:
                sample = x0
            else:
                # Re-noise from x_0 to the next timestep.
                next_t = timesteps[i + 1]
                alpha_bar_next = alphas_cumprod[
                    torch.tensor([next_t], device=device, dtype=torch.long)
                ]
                sqrt_ab = _broadcast_scalar(alpha_bar_next.sqrt(), x0)
                sqrt_omab = _broadcast_scalar(
                    (1.0 - alpha_bar_next).sqrt(), x0
                )
                noise = torch.randn(
                    x0.shape,
                    generator=generator,
                    device=device,
                    dtype=x0.dtype,
                )
                sample = sqrt_ab * x0 + sqrt_omab * noise

            self._invoke_callback(callback, i + 1, total, sample)

        self._logger.debug(
            "Consistency sampling completed in %d steps.", total
        )
        return sample


# ---------------------------------------------------------------------------
# Local helper (kept at the bottom to avoid polluting the class namespace)
# ---------------------------------------------------------------------------
def _broadcast_scalar(val: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a 0-D or 1-D tensor to match *ref*'s dimensionality."""
    while val.dim() < ref.dim():
        val = val.unsqueeze(-1)
    return val
