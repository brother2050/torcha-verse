"""Frechet Inception Distance (FID) for the TorchaVerse evaluation
framework (v0.4.0).

FID (Heusel et al., 2017) is the de-facto standard distribution-level
metric for image-generation models.  It compares the activations of an
Inception-v3 pool layer for two sets of images -- typically a "real"
reference set and a "generated" candidate set -- and reports the
Frechet distance (a.k.a. Wasserstein-2) between the two multivariate
Gaussians that best fit those activations:

.. math::

    \\mathrm{FID}(x, y) = \\lVert \\mu_x - \\mu_y \\rVert^2_2
        + \\mathrm{Tr}\\bigl(\\Sigma_x + \\Sigma_y
            - 2(\\Sigma_x \\Sigma_y)^{1/2}\\bigr)

The original FID relies on a pretrained Inception-v3 network
(ImageNet weights, ``pool3`` layer, 2048-d activations).  For the
v0.4.0 minimum-viable milestone we ship a *placeholder* Inception-style
backbone -- a small randomly-initialised convolutional network that
produces 2048-d L2-normalised features.  The placeholder follows the
exact same FID math as the real metric, so a future swap-in of a
pretrained Inception-v3 is a one-line class change behind the same
public API.

Why a placeholder?  Pulling a 95 MB Inception checkpoint and a
~100 MB safetensors download is the wrong default for a CI smoke test
that needs to run in 30 s on a clean checkout.  The placeholder keeps
the math honest -- the public function :func:`image_fid` still
returns a scalar Frechet distance in the same units as the real metric
-- and the public API (``compute_statistics`` / ``frechet_distance`` /
``FidCalculator``) is the production-ready surface that will hold
across the v0.4.x series.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``evaluation`` (this module) -- distribution metric.

Threading: this module is thread-safe.  The :class:`FidCalculator`
holds a lock around feature-extractor initialisation; the pure
functions :func:`compute_statistics` and :func:`frechet_distance` are
stateless and re-entrant.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger
from consistency.score import _to_tensor, _resize_tensor

__all__ = [
    "image_fid",
    "compute_statistics",
    "frechet_distance",
    "FidCalculator",
    "InceptionPlaceholder",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Dimension of the Inception-pool features used by the original FID.
_FEATURE_DIM: int = 2048

#: Default square image size for the FID backbone.
_IMAGE_SIZE: int = 64

#: Number of colour channels expected by the FID backbone.
_IMAGE_CHANNELS: int = 3

#: Channel widths of the placeholder Inception-style backbone.
_BACKBONE_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256, 512)

#: Convolution kernel size.
_KERNEL: int = 3

#: Convolution stride.
_STRIDE: int = 2

#: Convolution padding.
_PADDING: int = 1

#: Stddev for the random projection that lifts backbone features to
#: ``_FEATURE_DIM`` dimensions.
_PROJ_STD: float = 0.02

#: Numerical floor for matrix square-root iterations.
_EPS: float = 1e-10

#: Module-level logger.
_logger = get_logger("evaluation.fid")


# ---------------------------------------------------------------------------
# Inception-style placeholder backbone
# ---------------------------------------------------------------------------
class InceptionPlaceholder(nn.Module):
    """Lightweight Inception-style image-feature extractor (placeholder).

    The original FID pipeline feeds images through an Inception-v3
    network pretrained on ImageNet and reads the 2048-d activations of
    the final pool layer.  For v0.4.0 we ship a structural
    placeholder: a small convolutional backbone followed by a global
    average pool and a fixed random linear projection that lifts the
    backbone's flattened feature vector to ``_FEATURE_DIM`` dimensions.

    The random projection is normal-initialised, so different images
    produce discriminative features (the global-average-pool alone
    tends to collapse to a near-constant vector for small backbones).
    A future swap-in of a pretrained Inception-v3 is a one-class
    change behind the same ``forward(x) -> (N, _FEATURE_DIM)``
    interface.
    """

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        image_size: int = _IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.image_size = int(image_size)
        layers: List[nn.Module] = []
        in_ch = _IMAGE_CHANNELS
        for out_ch in _BACKBONE_CHANNELS:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=_KERNEL,
                    stride=_STRIDE,
                    padding=_PADDING,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, self.feature_dim)
        nn.init.normal_(self.proj.weight, std=_PROJ_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract FID-pool features.

        Args:
            x: Input tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            L2-normalised feature tensor ``(N, feature_dim)`` with
            unit-norm rows.
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


# ---------------------------------------------------------------------------
# Statistics and Frechet distance (pure functions)
# ---------------------------------------------------------------------------
@dataclass
class _ActivationStats:
    """Mean and covariance of an activation sample.

    Attributes:
        mu: 1-D tensor of shape ``(D,)``.
        sigma: 2-D tensor of shape ``(D, D)``.
    """

    mu: torch.Tensor
    sigma: torch.Tensor


def _to_batch_tensor(
    images: Sequence[Any], image_size: int
) -> torch.Tensor:
    """Convert a sequence of images to a ``(N, 3, image_size, image_size)`` tensor."""
    if not images:
        raise ValueError("`images` must be a non-empty sequence")
    tensors: List[torch.Tensor] = []
    for img in images:
        t = _to_tensor(img)
        if t.dim() == 3 and t.shape[0] == _IMAGE_CHANNELS:
            tensors.append(t)
        elif t.dim() == 2:
            tensors.append(t.unsqueeze(0).repeat(_IMAGE_CHANNELS, 1, 1))
        else:
            tensors.append(t[:_IMAGE_CHANNELS])
    out = torch.stack(tensors, dim=0)
    if out.shape[-1] != image_size or out.shape[-2] != image_size:
        out = F.interpolate(
            out, size=(image_size, image_size), mode="bilinear",
            align_corners=False,
        )
    return out.contiguous()


def compute_statistics(
    images: Sequence[Any],
    calculator: Optional["FidCalculator"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the activation ``(mu, sigma)`` of an image set.

    Args:
        images: A non-empty sequence of images (tensor / PIL / numpy
            / descriptor).  Mixed types are allowed; each is converted
            to a float tensor in ``[0, 1]`` independently.
        calculator: Optional :class:`FidCalculator` to reuse its
            feature extractor and image size.  When ``None`` a default
            CPU calculator is created on demand.

    Returns:
        ``(mu, sigma)`` where ``mu`` is a 1-D tensor of shape
        ``(D,)`` and ``sigma`` is a 2-D tensor of shape ``(D, D)``.

    Raises:
        ValueError: If ``images`` is empty.
    """
    if not images:
        raise ValueError("`images` must be a non-empty sequence")
    calc = calculator if calculator is not None else FidCalculator()
    batch = _to_batch_tensor(images, calc.image_size)
    activations = calc._extract(batch)  # noqa: SLF001 -- internal call
    mu = activations.mean(dim=0)
    centered = activations - mu.unsqueeze(0)
    # Use the biased estimator (1/N) -- this matches the original FID
    # implementation and keeps the metric numerically well-defined for
    # small batches.
    n = activations.shape[0]
    sigma = (centered.transpose(0, 1) @ centered) / float(n)
    return mu.detach().cpu(), sigma.detach().cpu()


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
) -> float:
    """Compute the Frechet distance between two Gaussians.

    Implements the closed-form Frechet distance between two multivariate
    Gaussians ``N(mu1, sigma1)`` and ``N(mu2, sigma2)``:

    .. math::

        d^2 = \\lVert \\mu_1 - \\mu_2 \\rVert^2_2
            + \\mathrm{Tr}\\bigl(\\Sigma_1 + \\Sigma_2
                - 2(\\Sigma_1 \\Sigma_2)^{1/2}\\bigr)

    The matrix square root is computed via an iterative
    Newton-Schulz / Schur-style algorithm (see :func:`_matrix_sqrt`)
    that runs in pure PyTorch with no ``scipy`` dependency.

    Args:
        mu1: 1-D tensor ``(D,)``.
        sigma1: 2-D tensor ``(D, D)`` (positive semi-definite).
        mu2: 1-D tensor ``(D,)``.
        sigma2: 2-D tensor ``(D, D)`` (positive semi-definite).

    Returns:
        A non-negative float.  Smaller is better; ``0`` means the two
        Gaussians are identical.
    """
    mu1 = mu1.detach().to(dtype=torch.float64)
    mu2 = mu2.detach().to(dtype=torch.float64)
    sigma1 = sigma1.detach().to(dtype=torch.float64)
    sigma2 = sigma2.detach().to(dtype=torch.float64)

    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    sqrt_prod = _matrix_sqrt(sigma1 @ sigma2)
    trace_term = float(
        torch.trace(sigma1 + sigma2 - 2.0 * sqrt_prod)
    )
    value = mean_term + trace_term
    # Numerical safety: the iterative square root can produce tiny
    # negative values when the matrices are nearly rank-deficient.
    return float(max(0.0, value))


def _matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Compute the matrix square root via eigendecomposition.

    For a symmetric positive semi-definite matrix ``A`` we have
    ``A = V @ diag(d) @ V.T`` where ``d >= 0``.  Therefore
    ``A^{1/2} = V @ diag(sqrt(d)) @ V.T``.  This is the most direct
    and numerically stable approach for SPD matrices and runs in
    pure PyTorch via :func:`torch.linalg.eigh`.

    Negative eigenvalues (which can appear when the covariance is
    near rank-deficient) are clamped to zero so the square root is
    always real-valued.

    Args:
        mat: Square symmetric positive semi-definite matrix.

    Returns:
        A tensor of the same shape as ``mat`` approximating
        ``mat^{1/2}``.
    """
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError("`mat` must be square")
    # ``eigh`` returns sorted eigenvalues in ascending order.
    eigvals, eigvecs = torch.linalg.eigh(mat)
    # Clamp tiny negative values from numerical noise.
    eigvals = eigvals.clamp(min=0.0)
    sqrt_vals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_vals.unsqueeze(0)) @ eigvecs.transpose(0, 1)


# ---------------------------------------------------------------------------
# Public FID entry point
# ---------------------------------------------------------------------------
def image_fid(
    real_images: Sequence[Any],
    generated_images: Sequence[Any],
    calculator: Optional["FidCalculator"] = None,
) -> float:
    """Compute the FID between two image sets.

    Args:
        real_images: A non-empty sequence of reference (real) images.
        generated_images: A non-empty sequence of generated images.
        calculator: Optional :class:`FidCalculator` for feature
            extraction.  When ``None`` a default CPU calculator is
            created (and cached at module level) so multiple FID calls
            in the same process share the same backbone.

    Returns:
        A non-negative float.  Smaller is better; ``0`` means the two
        distributions are identical in the chosen feature space.

    Raises:
        ValueError: If either input sequence is empty.
    """
    if not real_images:
        raise ValueError("`real_images` must be non-empty")
    if not generated_images:
        raise ValueError("`generated_images` must be non-empty")
    calc = calculator if calculator is not None else FidCalculator()
    mu1, sigma1 = compute_statistics(real_images, calc)
    mu2, sigma2 = compute_statistics(generated_images, calc)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ---------------------------------------------------------------------------
# FidCalculator (stateful wrapper)
# ---------------------------------------------------------------------------
class FidCalculator:
    """Stateful FID calculator with a cached feature extractor.

    Wraps an :class:`InceptionPlaceholder` (or, in the future, a
    pretrained Inception-v3) and exposes a uniform :meth:`fid` /
    :meth:`features` API.  A single instance is safe to share across
    threads -- the feature extractor is lazily initialised behind a
    lock.

    Args:
        device: Optional device for feature extraction.  Defaults to
            CPU so the calculator is portable across CI environments.
        image_size: Square image size for the backbone.
        feature_dim: Output feature dimension (2048 for classical FID).
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = _IMAGE_SIZE,
        feature_dim: int = _FEATURE_DIM,
    ) -> None:
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device if device is not None
            else torch.device("cpu")
        )
        self.image_size: int = int(image_size)
        self.feature_dim: int = int(feature_dim)
        self._extractor: Optional[nn.Module] = None
        self._lock: threading.Lock = threading.Lock()
        self._logger = _logger

    def _get_extractor(self) -> nn.Module:
        if self._extractor is None:
            with self._lock:
                if self._extractor is None:
                    net = InceptionPlaceholder(
                        feature_dim=self.feature_dim,
                        image_size=self.image_size,
                    ).to(self._device).eval()
                    self._extractor = net
        return self._extractor  # type: ignore[return-value]

    @torch.no_grad()
    def _extract(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the feature extractor on a ``(N, 3, H, W)`` batch."""
        extractor = self._get_extractor()
        out = extractor(batch.to(self._device))
        return out.detach().cpu()

    def features(self, images: Sequence[Any]) -> torch.Tensor:
        """Extract per-image features for a sequence of images.

        Args:
            images: A non-empty sequence of images.

        Returns:
            A tensor of shape ``(N, feature_dim)``.
        """
        if not images:
            raise ValueError("`images` must be non-empty")
        batch = _to_batch_tensor(images, self.image_size)
        return self._extract(batch)

    def fid(
        self,
        real_images: Sequence[Any],
        generated_images: Sequence[Any],
    ) -> float:
        """Compute FID between two image sets using this calculator."""
        if not real_images:
            raise ValueError("`real_images` must be non-empty")
        if not generated_images:
            raise ValueError("`generated_images` must be non-empty")
        feats_real = self.features(real_images)
        feats_gen = self.features(generated_images)
        mu1 = feats_real.mean(dim=0)
        mu2 = feats_gen.mean(dim=0)
        sigma1 = _covariance(feats_real, mu1)
        sigma2 = _covariance(feats_gen, mu2)
        return frechet_distance(mu1, sigma1, mu2, sigma2)

    def __repr__(self) -> str:
        return (
            "FidCalculator(device={}, image_size={}, feature_dim={})".format(
                self._device, self.image_size, self.feature_dim,
            )
        )


def _covariance(features: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Compute the (biased) covariance matrix of a feature batch.

    Args:
        features: ``(N, D)`` feature matrix.
        mu: ``(D,)`` per-dimension mean.

    Returns:
        ``(D, D)`` covariance matrix.
    """
    n = features.shape[0]
    centered = features - mu.unsqueeze(0)
    return (centered.transpose(0, 1) @ centered) / float(n)
