"""Noise schedule implementations (v0.8.0).

A noise schedule wraps a :class:`torch.Tensor` of
``alphas_cumprod`` (a.k.a. ``alpha_bar``) values.  Different
samplers prefer different shapes:

* :class:`NormalSchedule` — linear beta schedule, the default for
  ``euler`` / ``euler_a``.
* :class:`KarrasSchedule` — Karras et al. (2022) sigma schedule,
  good for ``dpmpp_2m`` / ``dpm_solver``.
* :class:`FlowMatchSchedule` — flow-matching linear schedule, the
  default for SD3 / FLUX / HunyuanVideo.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class _BaseSchedule:
    """Shared boilerplate for the v0.8 schedules."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        device: str = "cpu",
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be > 0, got {num_train_timesteps}."
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.device = str(device)
        self.betas = self._make_betas().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Alias used by the samplers in :mod:`core.schedulers.samplers`.
        self.timesteps = self.alphas_cumprod

    # ------------------------------------------------------------------
    def _make_betas(self) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError  # placeholder-registry: ignore


# ---------------------------------------------------------------------------
# Normal (linear DDPM-style) schedule
# ---------------------------------------------------------------------------
class NormalSchedule(_BaseSchedule):
    """Linear ``beta`` schedule from DDPM / SD1.5.

    ``beta_t`` grows linearly from ``beta_start`` to ``beta_end``
    over ``num_train_timesteps`` steps.
    """

    def _make_betas(
        self, beta_start: float = 0.00085, beta_end: float = 0.012,
    ) -> torch.Tensor:
        return torch.linspace(
            beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Karras schedule
# ---------------------------------------------------------------------------
class KarrasSchedule(_BaseSchedule):
    """Karras et al. (2022) sigma schedule.

    The original schedule is defined in terms of ``sigma_t`` rather
    than ``alpha_bar_t``; here we recover ``betas`` by
    ``alpha_bar = 1 / (1 + sigma^2)`` and ``beta = 1 - alpha_t /
    alpha_{t-1}``.
    """

    def _make_betas(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> torch.Tensor:
        ramp = torch.linspace(0, 1, self.num_train_timesteps, dtype=torch.float32)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        alphas_cumprod = 1.0 / (1.0 + sigmas ** 2)
        alphas_cumprod = alphas_cumprod.clamp(min=1e-8, max=1.0)
        alphas = alphas_cumprod / torch.cat([
            torch.ones(1, dtype=alphas_cumprod.dtype),
            alphas_cumprod[:-1],
        ])
        betas = 1.0 - alphas
        return betas.clamp(min=0.0, max=0.999)


# ---------------------------------------------------------------------------
# Flow-match schedule (SD3 / FLUX / HunyuanVideo)
# ---------------------------------------------------------------------------
class FlowMatchSchedule(_BaseSchedule):
    """Linear flow-matching schedule (``alpha_bar = 1 - t / T``).

    For flow-matching models the noise at timestep ``t`` is::

        x_t = (1 - t / T) * x_0 + (t / T) * noise

    which gives ``alpha_bar = 1 - t / T``.
    """

    def _make_betas(self) -> torch.Tensor:
        ramp = torch.linspace(
            0, 1, self.num_train_timesteps, dtype=torch.float32,
        )
        alpha_bar = 1.0 - ramp
        alpha_bar = alpha_bar.clamp(min=1e-8, max=1.0)
        alphas = alpha_bar / torch.cat([
            torch.ones(1, dtype=alpha_bar.dtype), alpha_bar[:-1],
        ])
        return (1.0 - alphas).clamp(min=0.0, max=0.999)
