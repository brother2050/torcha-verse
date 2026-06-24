"""Consistency scoring for the TorchaVerse consistency framework (v0.3.0).

This module provides the scoring primitives used by the consistency
pipeline to quantify how well a generated image (or video frame)
preserves the identity, garment, scene and depth signals of the
reference assets.

Two kinds of primitives are exposed:

* :class:`ConsistencyScore` -- a dataclass holding the per-axis scores
  (character / outfit / scene / depth / temporal) plus an aggregate
  ``overall`` field.  It round-trips through ``to_dict`` / ``from_dict``
  so that scores can be persisted alongside generation outputs.
* :class:`ScoreCalculator` -- the engine that computes a
  :class:`ConsistencyScore` from a generation output and a set of
  reference images.  It exposes three distance metrics:

  - :meth:`ScoreCalculator.clip_i_distance` -- CLIP-I image-feature
    cosine distance (``0`` = identical, ``1`` = orthogonal).
  - :meth:`ScoreCalculator.dinov2_distance` -- DINOv2-style feature
    cosine distance.
  - :meth:`ScoreCalculator.ssim` -- structural similarity index
    (``1`` = identical, ``0`` = unrelated).

The CLIP / DINOv2 feature extractors are lightweight, randomly
initialised :class:`torch.nn.Module` placeholders -- they do not depend
on the ``transformers`` library.  They produce deterministic, fixed-dim
feature vectors so that the distance metrics are reproducible within a
process, while remaining a drop-in target for a future pretrained
backbone.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L2 ``assets`` -- asset types (referenced for type hints only).
* L6 ``consistency`` (this module) -- scoring primitives.

This module depends on :mod:`torch` for tensor operations and feature
extraction.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger

__all__ = ["ConsistencyScore", "ScoreCalculator"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Dimension of the placeholder CLIP image-feature vector.
_CLIP_FEATURE_DIM: int = 512

#: Dimension of the placeholder DINOv2 image-feature vector.
_DINO_FEATURE_DIM: int = 768

#: Default square image size used when resizing inputs for feature
#: extraction.
_DEFAULT_IMAGE_SIZE: int = 224

#: Number of colour channels expected by the feature extractors.
_IMAGE_CHANNELS: int = 3

#: Convolution kernel size used by the placeholder feature extractors.
_CONV_KERNEL: int = 3

#: Convolution stride used by the placeholder feature extractors.
_CONV_STRIDE: int = 2

#: Convolution padding used by the placeholder feature extractors.
_CONV_PADDING: int = 1

#: Intermediate channel widths for the CLIP placeholder backbone.
_CLIP_CONV_CHANNELS: Tuple[int, ...] = (32, 64, 128)

#: Intermediate channel widths for the DINOv2 placeholder backbone.
_DINO_CONV_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)

#: Spatial pool size for the CLIP feature extractor (preserves more
#: spatial information than a global average pool, making features more
#: discriminative across different inputs).
_CLIP_POOL_SIZE: int = 7

#: Spatial pool size for the DINOv2 feature extractor.
_DINO_POOL_SIZE: int = 7

#: Standard deviation for the random-projection initialisation.
_PROJ_INIT_STD: float = 0.02

#: SSIM sliding-window size.
_SSIM_WINDOW: int = 11

#: SSIM stability constant C1 (L = 1.0 for [0, 1] images).
_SSIM_C1: float = (0.01) ** 2

#: SSIM stability constant C2 (L = 1.0 for [0, 1] images).
_SSIM_C2: float = (0.03) ** 2

#: Small epsilon to avoid division-by-zero in distance computations.
_DISTANCE_EPS: float = 1e-8

#: Default per-axis weights used by :meth:`ScoreCalculator.calculate`
#: when the caller does not supply explicit reference images for every
#: axis.  The weights mirror :class:`~consistency.profile.ConsistencyProfile`
#: defaults but are kept independent so this module has no circular
#: import on the profile package.
_DEFAULT_CHARACTER_WEIGHT: float = 0.8
_DEFAULT_OUTFIT_WEIGHT: float = 0.7
_DEFAULT_SCENE_WEIGHT: float = 0.6
_DEFAULT_DEPTH_WEIGHT: float = 0.5
_DEFAULT_TEMPORAL_WEIGHT: float = 0.0

#: Module-level logger.
_logger = get_logger("consistency.score")


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------
def _to_tensor(image: Any) -> torch.Tensor:
    """Convert a PIL image, numpy array, tensor, or other object to a tensor.

    The returned tensor is a float tensor normalised to ``[0, 1]``.
    When the input is a tensor, PIL image, or numpy array it is converted
    directly.  For any other type (e.g. a placeholder descriptor dict or
    an asset object) a deterministic tensor is generated from the
    object's content via :func:`_tensor_from_any`, so that the scoring
    pipeline works with placeholder outputs.

    Args:
        image: A :class:`torch.Tensor`, PIL image, numpy array, or any
            other object (fallback to deterministic generation).

    Returns:
        A float tensor of shape ``(C, H, W)`` in ``[0, 1]``.
    """
    if isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        # Normalise from [-1, 1] to [0, 1] if needed.
        if tensor.min() < 0:
            tensor = (tensor + 1.0) / 2.0
        return tensor.clamp(0.0, 1.0)

    # PIL image.
    try:
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(image, PILImage.Image):
            arr = np.array(image.convert("RGB")).astype("float32") / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
    except ImportError:
        pass

    # Numpy array.
    import numpy as np

    if isinstance(image, np.ndarray):
        arr = image.astype("float32")
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[-1] == _IMAGE_CHANNELS:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[None, ...]
        return torch.from_numpy(arr).clamp(0.0, 1.0)

    # Fallback: generate a deterministic tensor from any other object
    # (e.g. placeholder dicts, asset objects, strings).  This ensures
    # that the scoring pipeline works with placeholder outputs while
    # remaining discriminative: the same object always maps to the same
    # tensor, and different objects map to different tensors.
    return _tensor_from_any(image)


def _tensor_from_any(obj: Any) -> torch.Tensor:
    """Generate a deterministic ``(C, H, W)`` tensor from any object.

    This fallback is used when the input is not a tensor, PIL image, or
    numpy array (e.g. a placeholder descriptor dict or an asset object).
    It hashes the object's string representation and uses the digest to
    seed a deterministic random tensor, so that:

    * The same object always maps to the same tensor (reproducible).
    * Different objects map to different tensors (discriminative).

    Args:
        obj: Any Python object.

    Returns:
        A float tensor of shape ``(3, 8, 8)`` in ``[0, 1]``.
    """
    import hashlib

    # Try to extract a meaningful identifier from common placeholder
    # types (dicts with 'kind'/'id' keys, objects with an 'id' attr).
    if isinstance(obj, dict):
        identifier = str(obj.get("id", obj.get("kind", "")))
        identifier += str(obj.get("character_id", ""))
        identifier += str(obj.get("outfit_id", ""))
        identifier += str(obj.get("scene_id", ""))
    elif hasattr(obj, "id"):
        identifier = str(getattr(obj, "id", ""))
    else:
        identifier = str(obj)

    if not identifier:
        identifier = repr(obj)

    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    # Use the first 8 hex chars (32 bits) as a seed.
    seed = int(digest[:8], 16)
    gen = torch.Generator()
    gen.manual_seed(seed)
    tensor = torch.rand(_IMAGE_CHANNELS, 8, 8, generator=gen)
    return tensor


def _resize_tensor(
    tensor: torch.Tensor, size: Tuple[int, int]
) -> torch.Tensor:
    """Resize a ``(C, H, W)`` tensor to ``size`` using bilinear interpolation.

    Args:
        tensor: Input tensor ``(C, H, W)``.
        size: Target ``(height, width)``.

    Returns:
        Resized tensor ``(C, H, W)``.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    resized = F.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)


def _gaussian_kernel(
    window_size: int, sigma: float, channels: int
) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for SSIM.

    Args:
        window_size: Side length of the square kernel.
        sigma: Standard deviation of the Gaussian.
        channels: Number of channels (kernel is replicated per channel).

    Returns:
        A tensor of shape ``(channels, 1, window_size, window_size)``.
    """
    coords = torch.arange(window_size, dtype=torch.float32) - (
        window_size - 1
    ) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel2d = g[:, None] * g[None, :]
    kernel2d = kernel2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel2d


# ---------------------------------------------------------------------------
# Placeholder feature extractors
# ---------------------------------------------------------------------------
class _CLIPFeatureExtractor(nn.Module):
    """A lightweight CLIP-style image feature extractor (placeholder).

    This network produces a fixed-dimensional, L2-normalised feature
    vector from an input image.  It uses a small convolutional backbone
    followed by an adaptive average pool (to a non-trivial spatial size
    so that spatial information is preserved) and a random linear
    projection.  The projection weights are initialised from a normal
    distribution so that different images produce discriminative
    features (unlike a global-average-pool backbone whose output is
    dominated by bias terms and collapses for different inputs).

    The network is randomly initialised (not pretrained) so the
    resulting CLIP-I distances are *relative* rather than absolute,
    but they remain useful for comparing outputs produced under the
    same conditions.  It is a drop-in target for a future pretrained
    CLIP visual encoder.
    """

    def __init__(self, feature_dim: int = _CLIP_FEATURE_DIM) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = _IMAGE_CHANNELS
        for out_ch in _CLIP_CONV_CHANNELS:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=_CONV_KERNEL,
                    stride=_CONV_STRIDE,
                    padding=_CONV_PADDING,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(_CLIP_POOL_SIZE)
        flattened_dim = in_ch * _CLIP_POOL_SIZE * _CLIP_POOL_SIZE
        self.proj = nn.Linear(flattened_dim, feature_dim)
        self.feature_dim = feature_dim
        # Normal-initialise the projection so different images yield
        # discriminative features.
        nn.init.normal_(self.proj.weight, std=_PROJ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised features.

        Args:
            x: Input tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            Feature tensor ``(N, feature_dim)`` with unit-norm rows.
        """
        feat = self.backbone(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        feat = self.proj(feat)
        return F.normalize(feat, dim=-1)


class _DINOFeatureExtractor(nn.Module):
    """A lightweight DINOv2-style image feature extractor (placeholder).

    Similar to :class:`_CLIPFeatureExtractor` but with a deeper backbone
    (four conv layers instead of three) and a wider feature dimension,
    mimicking the DINOv2 architecture profile.  Like the CLIP extractor
    it uses a non-trivial adaptive pool and a normal-initialised random
    projection so that different images produce discriminative features.
    It is a drop-in target for a future pretrained DINOv2 backbone.
    """

    def __init__(self, feature_dim: int = _DINO_FEATURE_DIM) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = _IMAGE_CHANNELS
        for out_ch in _DINO_CONV_CHANNELS:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=_CONV_KERNEL,
                    stride=_CONV_STRIDE,
                    padding=_CONV_PADDING,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(_DINO_POOL_SIZE)
        flattened_dim = in_ch * _DINO_POOL_SIZE * _DINO_POOL_SIZE
        self.proj = nn.Linear(flattened_dim, feature_dim)
        self.feature_dim = feature_dim
        nn.init.normal_(self.proj.weight, std=_PROJ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised features.

        Args:
            x: Input tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            Feature tensor ``(N, feature_dim)`` with unit-norm rows.
        """
        feat = self.backbone(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        feat = self.proj(feat)
        return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# ConsistencyScore
# ---------------------------------------------------------------------------
@dataclass
class ConsistencyScore:
    """Per-axis and aggregate consistency scores for a generation output.

    Every score is a float in ``[0, 1]`` where ``1`` means perfect
    consistency with the reference and ``0`` means no consistency.  The
    :attr:`overall` field is a weighted aggregate of the per-axis
    scores; it is stored explicitly so that it can be set by the caller
    (e.g. by :meth:`ScoreCalculator.calculate`) rather than recomputed
    on every access.

    Attributes:
        character_score: Character identity consistency (0-1).
        outfit_score: Outfit / garment consistency (0-1).
        scene_score: Scene / environment consistency (0-1).
        depth_score: Depth-map consistency (0-1).
        temporal_score: Temporal consistency across frames (0-1);
            ``0`` for single-image outputs.
        overall: Aggregate consistency score (0-1).
    """

    character_score: float
    outfit_score: float
    scene_score: float
    depth_score: float
    temporal_score: float
    overall: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this score to a JSON-serialisable dictionary.

        Returns:
            A dictionary with all six score fields.
        """
        return {
            "character_score": self.character_score,
            "outfit_score": self.outfit_score,
            "scene_score": self.scene_score,
            "depth_score": self.depth_score,
            "temporal_score": self.temporal_score,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsistencyScore":
        """Reconstruct a :class:`ConsistencyScore` from a serialised dict.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`ConsistencyScore` instance.
        """
        return cls(
            character_score=float(d.get("character_score", 0.0)),
            outfit_score=float(d.get("outfit_score", 0.0)),
            scene_score=float(d.get("scene_score", 0.0)),
            depth_score=float(d.get("depth_score", 0.0)),
            temporal_score=float(d.get("temporal_score", 0.0)),
            overall=float(d.get("overall", 0.0)),
        )

    def __repr__(self) -> str:
        return (
            "ConsistencyScore(character={:.3f}, outfit={:.3f}, "
            "scene={:.3f}, depth={:.3f}, temporal={:.3f}, "
            "overall={:.3f})".format(
                self.character_score,
                self.outfit_score,
                self.scene_score,
                self.depth_score,
                self.temporal_score,
                self.overall,
            )
        )


# ---------------------------------------------------------------------------
# ScoreCalculator
# ---------------------------------------------------------------------------
class ScoreCalculator:
    """Engine that computes :class:`ConsistencyScore` from generation outputs.

    The calculator wraps two placeholder feature extractors (CLIP-style
    and DINOv2-style) and exposes three distance metrics:

    * :meth:`clip_i_distance` -- CLIP-I cosine distance (0 = identical).
    * :meth:`dinov2_distance` -- DINOv2 cosine distance (0 = identical).
    * :meth:`ssim` -- structural similarity (1 = identical).

    The :meth:`calculate` method combines these into a single
    :class:`ConsistencyScore` by comparing a generation output against a
    set of reference images keyed by axis (``character``, ``outfit``,
    ``scene``, ``depth``).

    The feature extractors are lazily initialised and cached behind a
    :class:`threading.Lock` so that the calculator is safe to share
    across threads.

    Args:
        device: Optional device for feature extraction.  When ``None``
            the default CPU device is used.
        image_size: Square image size used when resizing inputs for
            feature extraction.  Defaults to 224.
        clip_feature_dim: Dimension of the CLIP placeholder features.
        dino_feature_dim: Dimension of the DINOv2 placeholder features.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        clip_feature_dim: int = _CLIP_FEATURE_DIM,
        dino_feature_dim: int = _DINO_FEATURE_DIM,
    ) -> None:
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device if device is not None
            else torch.device("cpu")
        )
        self._image_size: int = int(image_size)
        self._clip_feature_dim: int = int(clip_feature_dim)
        self._dino_feature_dim: int = int(dino_feature_dim)

        self._clip_extractor: Optional[_CLIPFeatureExtractor] = None
        self._dino_extractor: Optional[_DINOFeatureExtractor] = None
        self._lock: threading.Lock = threading.Lock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """The device on which feature extraction runs."""
        return self._device

    @property
    def image_size(self) -> int:
        """The square image size used for feature extraction."""
        return self._image_size

    # ------------------------------------------------------------------
    # Lazy feature-extractor initialisation
    # ------------------------------------------------------------------
    def _get_clip_extractor(self) -> _CLIPFeatureExtractor:
        """Return the (lazily initialised) CLIP feature extractor."""
        if self._clip_extractor is None:
            with self._lock:
                if self._clip_extractor is None:
                    extractor = _CLIPFeatureExtractor(
                        feature_dim=self._clip_feature_dim
                    ).to(self._device)
                    extractor.eval()
                    self._clip_extractor = extractor
        return self._clip_extractor  # type: ignore[return-value]

    def _get_dino_extractor(self) -> _DINOFeatureExtractor:
        """Return the (lazily initialised) DINOv2 feature extractor."""
        if self._dino_extractor is None:
            with self._lock:
                if self._dino_extractor is None:
                    extractor = _DINOFeatureExtractor(
                        feature_dim=self._dino_feature_dim
                    ).to(self._device)
                    extractor.eval()
                    self._dino_extractor = extractor
        return self._dino_extractor  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Distance metrics
    # ------------------------------------------------------------------
    def clip_i_distance(
        self, image1: Any, image2: Any
    ) -> float:
        """Compute the CLIP-I cosine distance between two images.

        CLIP-I distance is ``1 - cos_sim(f1, f2)`` where ``f1`` and
        ``f2`` are CLIP image-feature vectors.  A distance of ``0``
        means the two images are identical in feature space; ``1``
        means they are orthogonal.

        Args:
            image1: The first image (tensor / PIL / numpy).
            image2: The second image (tensor / PIL / numpy).

        Returns:
            A float in ``[0, 2]`` (typically ``[0, 1]`` for normalised
            features).
        """
        t1 = self._prepare_image(image1)
        t2 = self._prepare_image(image2)
        extractor = self._get_clip_extractor()
        with torch.no_grad():
            f1 = extractor(t1.unsqueeze(0).to(self._device))
            f2 = extractor(t2.unsqueeze(0).to(self._device))
            cos_sim = F.cosine_similarity(f1, f2, dim=-1)
            distance = 1.0 - cos_sim.item()
        return float(max(0.0, distance))

    def dinov2_distance(
        self, image1: Any, image2: Any
    ) -> float:
        """Compute the DINOv2 cosine distance between two images.

        DINOv2 distance is ``1 - cos_sim(f1, f2)`` where ``f1`` and
        ``f2`` are DINOv2 image-feature vectors.  Semantically identical
        to :meth:`clip_i_distance` but uses a different (deeper, wider)
        feature space.

        Args:
            image1: The first image (tensor / PIL / numpy).
            image2: The second image (tensor / PIL / numpy).

        Returns:
            A float in ``[0, 2]`` (typically ``[0, 1]``).
        """
        t1 = self._prepare_image(image1)
        t2 = self._prepare_image(image2)
        extractor = self._get_dino_extractor()
        with torch.no_grad():
            f1 = extractor(t1.unsqueeze(0).to(self._device))
            f2 = extractor(t2.unsqueeze(0).to(self._device))
            cos_sim = F.cosine_similarity(f1, f2, dim=-1)
            distance = 1.0 - cos_sim.item()
        return float(max(0.0, distance))

    def ssim(
        self, image1: Any, image2: Any
    ) -> float:
        """Compute the Structural Similarity Index (SSIM) between two images.

        SSIM ranges from ``-1`` to ``1`` where ``1`` means the two
        images are structurally identical.  This implementation uses a
        Gaussian-weighted sliding window.

        Args:
            image1: The first image (tensor / PIL / numpy).
            image2: The second image (tensor / PIL / numpy).

        Returns:
            A float in ``[-1, 1]`` (clamped to ``[0, 1]``).
        """
        t1 = self._prepare_image(image1)
        t2 = self._prepare_image(image2)
        return self._compute_ssim(t1, t2)

    # ------------------------------------------------------------------
    # Aggregate scoring
    # ------------------------------------------------------------------
    def calculate(
        self,
        output: Any,
        references: Dict[str, Any],
    ) -> ConsistencyScore:
        """Compute a full :class:`ConsistencyScore` for ``output``.

        Each axis (``character``, ``outfit``, ``scene``, ``depth``) is
        scored by comparing ``output`` against the corresponding
        reference image in ``references`` using the CLIP-I distance
        (converted to a similarity in ``[0, 1]``).  When a reference is
        missing the axis defaults to ``0``.  The temporal axis is
        scored from ``references["frames"]`` when present (average
        pairwise CLIP-I similarity); otherwise it defaults to ``0``.

        The ``overall`` field is a weighted average of the per-axis
        scores using the default weights defined at module level.

        Args:
            output: The generated image to score.
            references: A dictionary mapping axis names to reference
                images.  Recognised keys: ``"character"``,
                ``"outfit"``, ``"scene"``, ``"depth"``, ``"frames"``.

        Returns:
            A :class:`ConsistencyScore` with all six fields populated.
        """
        character_ref = references.get("character")
        outfit_ref = references.get("outfit")
        scene_ref = references.get("scene")
        depth_ref = references.get("depth")
        frames_ref = references.get("frames")

        character_score = self._axis_similarity(output, character_ref)
        outfit_score = self._axis_similarity(output, outfit_ref)
        scene_score = self._axis_similarity(output, scene_ref)
        depth_score = self._axis_similarity(output, depth_ref)
        temporal_score = self._temporal_similarity(output, frames_ref)

        overall = self._aggregate(
            character_score,
            outfit_score,
            scene_score,
            depth_score,
            temporal_score,
        )

        score = ConsistencyScore(
            character_score=character_score,
            outfit_score=outfit_score,
            scene_score=scene_score,
            depth_score=depth_score,
            temporal_score=temporal_score,
            overall=overall,
        )
        self._logger.debug(
            "Calculated consistency score: %s", score
        )
        return score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Convert and resize an image to the feature-extraction format.

        Args:
            image: A tensor / PIL / numpy image.

        Returns:
            A float tensor of shape ``(C, image_size, image_size)``.
        """
        tensor = _to_tensor(image)
        if tensor.shape[-2:] != (self._image_size, self._image_size):
            tensor = _resize_tensor(
                tensor, (self._image_size, self._image_size)
            )
        return tensor

    def _axis_similarity(
        self, output: Any, reference: Optional[Any]
    ) -> float:
        """Compute a ``[0, 1]`` similarity for a single axis.

        Uses CLIP-I distance converted to similarity (``1 - distance``).
        When ``reference`` is ``None`` the axis defaults to ``0``.

        Args:
            output: The generated image.
            reference: The reference image (or ``None``).

        Returns:
            A float in ``[0, 1]``.
        """
        if reference is None:
            return 0.0
        distance = self.clip_i_distance(output, reference)
        similarity = 1.0 - distance
        return float(max(0.0, min(1.0, similarity)))

    def _temporal_similarity(
        self, output: Any, frames: Optional[Sequence[Any]]
    ) -> float:
        """Compute the average pairwise CLIP-I similarity across frames.

        When ``frames`` is ``None`` or has fewer than two entries the
        temporal score defaults to ``0``.

        Args:
            output: The primary output frame.
            frames: A sequence of additional frames.

        Returns:
            A float in ``[0, 1]``.
        """
        if frames is None or len(frames) < 2:
            return 0.0
        all_frames: List[Any] = [output] + list(frames)
        similarities: List[float] = []
        for i in range(len(all_frames)):
            for j in range(i + 1, len(all_frames)):
                distance = self.clip_i_distance(
                    all_frames[i], all_frames[j]
                )
                similarities.append(1.0 - distance)
        if not similarities:
            return 0.0
        avg = sum(similarities) / len(similarities)
        return float(max(0.0, min(1.0, avg)))

    def _aggregate(
        self,
        character: float,
        outfit: float,
        scene: float,
        depth: float,
        temporal: float,
    ) -> float:
        """Compute the weighted aggregate of per-axis scores.

        Uses the module-level default weights.  The temporal weight is
        only non-zero when the temporal score is meaningful (i.e. when
        frames were provided).

        Args:
            character: Character-axis score.
            outfit: Outfit-axis score.
            scene: Scene-axis score.
            depth: Depth-axis score.
            temporal: Temporal-axis score.

        Returns:
            A float in ``[0, 1]``.
        """
        weights = {
            "character": _DEFAULT_CHARACTER_WEIGHT,
            "outfit": _DEFAULT_OUTFIT_WEIGHT,
            "scene": _DEFAULT_SCENE_WEIGHT,
            "depth": _DEFAULT_DEPTH_WEIGHT,
            "temporal": _DEFAULT_TEMPORAL_WEIGHT,
        }
        scores = {
            "character": character,
            "outfit": outfit,
            "scene": scene,
            "depth": depth,
            "temporal": temporal,
        }
        total_weight = sum(weights.values())
        if total_weight <= _DISTANCE_EPS:
            return 0.0
        weighted_sum = sum(
            scores[axis] * weights[axis] for axis in weights
        )
        return float(max(0.0, min(1.0, weighted_sum / total_weight)))

    def _compute_ssim(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> float:
        """Compute SSIM between two ``(C, H, W)`` tensors in ``[0, 1]``.

        Args:
            img1: First image tensor ``(C, H, W)``.
            img2: Second image tensor ``(C, H, W)``.

        Returns:
            A float in ``[0, 1]``.
        """
        if img1.shape != img2.shape:
            target = (img1.shape[-2], img1.shape[-1])
            img2 = _resize_tensor(img2, target)

        c = img1.shape[0]
        kernel = _gaussian_kernel(
            _SSIM_WINDOW, sigma=1.5, channels=c
        ).to(img1.device)
        pad = _SSIM_WINDOW // 2

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

        num = (2 * mu1_mu2 + _SSIM_C1) * (2 * sigma12 + _SSIM_C2)
        den = (mu1_sq + mu2_sq + _SSIM_C1) * (
            sigma1_sq + sigma2_sq + _SSIM_C2
        )
        ssim_map = num / (den + _DISTANCE_EPS)
        value = ssim_map.mean().item()
        return float(max(0.0, min(1.0, value)))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "ScoreCalculator(device={}, image_size={}, "
            "clip_dim={}, dino_dim={})".format(
                self._device,
                self._image_size,
                self._clip_feature_dim,
                self._dino_feature_dim,
            )
        )
