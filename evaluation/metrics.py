"""Image-quality metrics for the TorchaVerse evaluation framework (v0.4.0).

This module provides classical full-reference image-quality metrics that
can be used to score the output of the L4 generation nodes (text / image
/ video) against reference images.  All metrics are implemented in pure
PyTorch (and a thin standard-library helper layer), so they run in any
environment that has ``torch`` installed -- no ``scipy``,
``torchmetrics``, ``pytorch-fid`` or other third-party metric packages
are required.

Three metrics are exposed:

* :func:`psnr` -- Peak Signal-to-Noise Ratio (in dB, ``higher = better``).
* :func:`ssim` -- Structural Similarity Index (in ``[0, 1]``,
  ``higher = better``).
* :func:`lpips` -- Learned Perceptual Image Patch Similarity.  This is
  shipped as a **structural placeholder** that returns a deterministic
  cosine-distance score in ``[0, 1]`` derived from the existing
  :class:`consistency.score.ScoreCalculator` placeholder backbone.  It
  keeps the public API stable; swapping in a real LPIPS network later
  is a drop-in change behind the same function signature.

The module reuses the image-conversion helpers from
:mod:`consistency.score` so the public API accepts the same flexible
inputs (tensor / PIL / numpy / placeholder descriptor).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``evaluation`` (this module) -- full-reference image metrics.

The functions are thread-safe; they do not maintain mutable state.
"""

from __future__ import annotations

import math
import threading
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger
from consistency.score import (
    _to_tensor,
    _resize_tensor,
    _gaussian_kernel,
)

__all__ = ["psnr", "ssim", "lpips", "LpipPlaceholder"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default data range used by :func:`psnr` / :func:`ssim` for tensors in
#: ``[0, 1]``.  PSNR is technically ``10 * log10(MAX^2 / MSE)`` where
#: ``MAX`` is the maximum representable value of the input (``1.0`` for
#: float images in ``[0, 1]``).
_PSNR_MAX_DEFAULT: float = 1.0

#: PSNR value returned when two images are bit-for-bit identical (MSE
#: is zero and a finite cap prevents ``log10(0)``).
_PSNR_INF: float = 100.0

#: SSIM sliding-window size, mirroring ``consistency.score``.
_SSIM_WINDOW: int = 11

#: SSIM stability constant ``C1`` (``L = 1.0`` for ``[0, 1]`` images).
_SSIM_C1: float = (0.01) ** 2

#: SSIM stability constant ``C2`` (``L = 1.0`` for ``[0, 1]`` images).
_SSIM_C2: float = (0.03) ** 2

#: Numerical floor to avoid division-by-zero.
_EPS: float = 1e-12

#: Default square size used for the LPIPS placeholder backbone.
_LPIPS_IMAGE_SIZE: int = 32

#: Default feature dimension for the LPIPS placeholder backbone.
_LPIPS_FEATURE_DIM: int = 64

#: Module-level logger.
_logger = get_logger("evaluation.metrics")


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def psnr(
    image1: Any,
    image2: Any,
    max_value: float = _PSNR_MAX_DEFAULT,
) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Both inputs are converted to ``(C, H, W)`` float tensors in
    ``[0, max_value]`` (default ``[0, 1]``) and, if their shapes differ,
    ``image2`` is bilinearly resized to match ``image1``.  The PSNR is
    ``10 * log10(max_value ** 2 / MSE)`` where ``MSE`` is the
    channel-averaged mean-squared error.

    When ``image1`` and ``image2`` are identical, ``MSE`` is zero and
    the function returns :data:`_PSNR_INF` (a finite cap, so callers do
    not need to handle ``inf``).

    Args:
        image1: The first image (tensor / PIL / numpy / descriptor).
        image2: The second image (any compatible format).
        max_value: The maximum representable value of the inputs
            (``1.0`` for ``[0, 1]`` floats, ``255.0`` for uint8).

    Returns:
        A float in ``[0, +inf)`` measured in decibels.  Higher is
        better; ``> 40 dB`` typically indicates the images are
        visually indistinguishable.
    """
    t1 = _to_tensor(image1)
    t2 = _to_tensor(image2)
    t1, t2 = _align_shapes(t1, t2)
    mse = (t1 - t2).pow(2).mean().item()
    if mse <= _EPS:
        return float(_PSNR_INF)
    value = 10.0 * math.log10((max_value * max_value) / mse)
    return float(value)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def ssim(
    image1: Any,
    image2: Any,
    window_size: int = _SSIM_WINDOW,
) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images.

    Mirrors the implementation in :mod:`consistency.score` but is
    exposed as a free function so it can be used without instantiating
    a :class:`ScoreCalculator`.  The result is the mean SSIM over all
    pixels and all channels, clamped to ``[0, 1]``.

    Args:
        image1: The first image (tensor / PIL / numpy / descriptor).
        image2: The second image.
        window_size: Side length of the Gaussian sliding window.
            Defaults to ``11`` (the classical SSIM configuration).

    Returns:
        A float in ``[0, 1]``.  ``1`` = structurally identical;
        ``0`` = unrelated.
    """
    t1 = _to_tensor(image1)
    t2 = _to_tensor(image2)
    t1, t2 = _align_shapes(t1, t2)
    return _compute_ssim(t1, t2, window_size=window_size)


def _compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = _SSIM_WINDOW,
) -> float:
    """Compute SSIM between two ``(C, H, W)`` tensors in ``[0, 1]``."""
    c = img1.shape[0]
    kernel = _gaussian_kernel(
        window_size, sigma=1.5, channels=c
    ).to(img1.device)
    pad = window_size // 2

    x1 = img1.unsqueeze(0)
    x2 = img2.unsqueeze(0)

    mu1 = F.conv2d(x1, kernel, padding=pad, groups=c)
    mu2 = F.conv2d(x2, kernel, padding=pad, groups=c)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x1 * x1, kernel, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, kernel, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(x1 * x2, kernel, padding=pad, groups=c) - mu1_mu2

    num = (2.0 * mu1_mu2 + _SSIM_C1) * (2.0 * sigma12 + _SSIM_C2)
    den = (
        mu1_sq + mu2_sq + _SSIM_C1
    ) * (sigma1_sq + sigma2_sq + _SSIM_C2)
    ssim_map = num / (den + _EPS)
    value = ssim_map.mean().item()
    return float(max(0.0, min(1.0, value)))


# ---------------------------------------------------------------------------
# LPIPS placeholder
# ---------------------------------------------------------------------------
class LpipPlaceholder(nn.Module):
    """Lightweight LPIPS-style perceptual feature extractor (placeholder).

    The real LPIPS model is a learned CNN (typically VGG/AlexNet
    backbones) trained to mimic human perceptual judgments.  For the
    initial v0.4.0 milestone we ship a *structural* placeholder: a tiny
    randomly-initialised convolutional backbone followed by a fixed
    linear projection.  The placeholder produces a deterministic
    perceptual feature vector for each image, and the public :func:`lpips`
    function returns ``1 - cos_sim(f1, f2)`` -- a value in ``[0, 1]``
    where ``0`` means the two images are perceptually identical in the
    placeholder's feature space.

    This keeps the public API stable across the v0.4.x series so that
    callers can wire :func:`lpips` into their evaluation pipelines
    today, and a future swap-in of a pretrained LPIPS backbone becomes
    a single-class change behind the same function signature.

    The network is randomly initialised (not pretrained) so the
    resulting LPIPS distances are *relative* rather than absolute, but
    they remain useful for comparing outputs produced under the same
    conditions.
    """

    def __init__(
        self,
        feature_dim: int = _LPIPS_FEATURE_DIM,
        image_size: int = _LPIPS_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.image_size = int(image_size)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, self.feature_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised perceptual features.

        Args:
            x: Input tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            Feature tensor ``(N, feature_dim)`` with unit-norm rows.
        """
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        feat = self.proj(feat)
        return F.normalize(feat, dim=-1)


# Module-level lazy singleton for the LPIPS placeholder so we don't
# rebuild the network on every call.
_lpips_instance: Optional[LpipPlaceholder] = None
_lpips_lock: threading.Lock = threading.Lock()


def _get_lpips_placeholder() -> LpipPlaceholder:
    """Return the lazily-initialised module-level LPIPS placeholder."""
    global _lpips_instance
    if _lpips_instance is None:
        with _lpips_lock:
            if _lpips_instance is None:
                net = LpipPlaceholder().eval()
                _lpips_instance = net
    return _lpips_instance  # type: ignore[return-value]


def lpips(image1: Any, image2: Any) -> float:
    """Compute the LPIPS perceptual distance between two images.

    The current implementation is a *placeholder* -- a small
    randomly-initialised convolutional backbone that produces
    deterministic perceptual features.  See :class:`LpipPlaceholder`
    for the rationale and the migration path to a real LPIPS network.

    Args:
        image1: The first image (tensor / PIL / numpy / descriptor).
        image2: The second image.

    Returns:
        A float in ``[0, 1]`` where ``0`` = perceptually identical
        and ``1`` = maximally different in the placeholder feature
        space.  (``cosine distance`` of unit-norm features is in
        ``[0, 1]``.)
    """
    t1 = _to_tensor(image1)
    t2 = _to_tensor(image2)
    t1, t2 = _align_shapes(t1, t2)
    net = _get_lpips_placeholder()
    with torch.no_grad():
        f1 = net(t1.unsqueeze(0))
        f2 = net(t2.unsqueeze(0))
        cos_sim = F.cosine_similarity(f1, f2, dim=-1).item()
    distance = max(0.0, min(1.0, 1.0 - cos_sim))
    return float(distance)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _align_shapes(
    t1: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align two image tensors to a common ``(C, H, W)`` shape.

    The first tensor's spatial size always wins; ``t2`` is bilinearly
    resized to match it.  Channel counts must already match (no
    padding/trimming of channels is performed -- that is a separate
    operation the caller must do explicitly).
    """
    if t1.shape[-2:] != t2.shape[-2:]:
        t2 = _resize_tensor(t2, (t1.shape[-2], t1.shape[-1]))
    if t1.shape[0] != t2.shape[0]:
        # Mismatched channel counts: take the minimum and slice both.
        c = min(t1.shape[0], t2.shape[0])
        t1 = t1[:c]
        t2 = t2[:c]
    return t1.contiguous(), t2.contiguous()
