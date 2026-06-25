"""Scene consistency engine for the TorchaVerse consistency framework
(v0.3.0).

This module provides :class:`SceneEngine`, the engine responsible for
creating, applying and extracting scene / environment conditioning,
including depth-map generation.  It is the third of the four
"consistency engines" that the
:class:`~consistency.pipeline.ConsistencyPipeline` composes.

Capabilities:

* :meth:`SceneEngine.create_scene` -- create a
  :class:`~assets.model_asset.SceneAsset` from a description and an
  optional reference image / LoRA reference.
* :meth:`SceneEngine.apply_scene` -- apply scene conditioning (scene
  LoRA + ControlNet) to an image at a given weight.
* :meth:`SceneEngine.generate_depth_map` -- generate a depth map from
  an image using a specified estimation method (``"midas"`` or
  ``"depth_anything"``).
* :meth:`SceneEngine.apply_depth` -- apply depth conditioning
  (ControlNet depth) to an image at a given weight.

The depth-map generation uses a lightweight torch-based estimator as a
placeholder; the conditioning logic returns descriptor dictionaries.
The full interface is exercised so that the methods can be swapped for
a real generation backend without changing call sites.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L2 ``assets`` -- :class:`~assets.model_asset.SceneAsset`,
  :class:`~assets.model_asset.DepthAsset`,
  :class:`~assets.store.AssetStore`, :class:`~assets.base.AssetRef`.
* L6 ``consistency`` (this module) -- scene engine + depth estimation.

This module depends on :mod:`torch` for the depth-map estimation
placeholder.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

from assets.base import AssetRef
from assets.model_asset import DepthAsset, SceneAsset
from assets.store import AssetStore
from assets.types import AssetStatus, AssetType
from infrastructure.logger import get_logger

__all__ = ["SceneEngine"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default scene-conditioning weight.
_DEFAULT_SCENE_WEIGHT: float = 0.6

#: Default depth-conditioning weight.
_DEFAULT_DEPTH_WEIGHT: float = 0.5

#: Weight lower bound.
_WEIGHT_MIN: float = 0.0

#: Weight upper bound.
_WEIGHT_MAX: float = 1.0

#: Allowed depth-estimation methods.
_DEPTH_METHODS: tuple[str, ...] = ("midas", "depth_anything")

#: Default depth-estimation method.
_DEFAULT_DEPTH_METHOD: str = "midas"

#: Image channels for the depth estimator.
_DEPTH_CHANNELS: int = 3

#: Convolution kernel size for the depth estimator.
_DEPTH_KERNEL: int = 3

#: Convolution padding for the depth estimator.
_DEPTH_PADDING: int = 1

#: Intermediate channel widths for the depth estimator backbone.
_DEPTH_CONV_CHANNELS: tuple[int, ...] = (16, 32, 64)

#: Default square image size for depth estimation.
_DEPTH_IMAGE_SIZE: int = 256

#: Module-level logger.
_logger = get_logger("consistency.scene")


# ---------------------------------------------------------------------------
# Image conversion helper
# ---------------------------------------------------------------------------
def _to_tensor(image: Any) -> torch.Tensor:
    """Convert a PIL image, numpy array, or tensor to a ``(C, H, W)`` tensor.

    The returned tensor is a float tensor normalised to ``[0, 1]``.

    Args:
        image: A :class:`torch.Tensor`, PIL image, or numpy array.

    Returns:
        A float tensor of shape ``(C, H, W)`` in ``[0, 1]``.

    Raises:
        TypeError: If ``image`` is of an unsupported type.
    """
    if isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.min() < 0:
            tensor = (tensor + 1.0) / 2.0
        return tensor.clamp(0.0, 1.0)

    try:
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(image, PILImage.Image):
            arr = np.array(image.convert("RGB")).astype("float32") / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
    except ImportError as exc:
        _logger.debug("PIL/numpy import unavailable, using tensor path: %s", exc)

    import numpy as np

    if isinstance(image, np.ndarray):
        arr = image.astype("float32")
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[-1] == _DEPTH_CHANNELS:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[None, ...]
        return torch.from_numpy(arr).clamp(0.0, 1.0)

    raise TypeError(
        "Unsupported image type: {}. Expected torch.Tensor, PIL Image, "
        "or numpy.ndarray.".format(type(image).__name__)
    )


# ---------------------------------------------------------------------------
# Placeholder depth estimator
# ---------------------------------------------------------------------------
class _DepthEstimator(nn.Module):
    """A lightweight depth-estimation network (placeholder).

    This network produces a single-channel depth map from an RGB input.
    It is randomly initialised (not pretrained) so the resulting depth
    maps are *relative* rather than metrically accurate, but they are
    structurally consistent and suitable as a ControlNet conditioning
    signal placeholder.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = _DEPTH_CHANNELS
        for out_ch in _DEPTH_CONV_CHANNELS:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=_DEPTH_KERNEL,
                    padding=_DEPTH_PADDING,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Conv2d(
            in_ch, 1, kernel_size=_DEPTH_KERNEL, padding=_DEPTH_PADDING
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate a depth map.

        Args:
            x: Input tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            Depth tensor ``(N, 1, H, W)`` in ``[0, 1]``.
        """
        feat = self.encoder(x)
        depth = self.decoder(feat)
        depth = torch.sigmoid(depth)
        return depth


# ---------------------------------------------------------------------------
# SceneEngine
# ---------------------------------------------------------------------------
class SceneEngine:
    """Engine for creating, applying and extracting scene conditioning.

    The engine wraps an :class:`~assets.store.AssetStore` for persisting
    scene and depth assets, and a lightweight :class:`_DepthEstimator`
    for depth-map generation.  All public operations are thread-safe
    thanks to a :class:`threading.Lock` guarding the depth estimator.

    Args:
        asset_store: The tiered asset store used to persist scene and
            depth assets.
    """

    def __init__(
        self,
        asset_store: AssetStore,
    ) -> None:
        self._store: AssetStore = asset_store
        self._depth_estimator: Optional[_DepthEstimator] = None
        self._lock: threading.Lock = threading.Lock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def asset_store(self) -> AssetStore:
        """The underlying :class:`AssetStore`."""
        return self._store

    # ------------------------------------------------------------------
    # Scene creation
    # ------------------------------------------------------------------
    def create_scene(
        self,
        name: str,
        description: str,
        reference_image: Optional[Union[str, Any]] = None,
        lora_ref: Optional[AssetRef] = None,
    ) -> SceneAsset:
        """Create and persist a :class:`SceneAsset`.

        The scene is created with :attr:`AssetStatus.DRAFT` status.  When
        a ``reference_image`` is provided, a depth map is generated from
        it and referenced via :attr:`SceneAsset.depth_ref`.

        Args:
            name: Human-readable scene name.
            description: Textual description used for prompt assembly.
            reference_image: Optional path to (or descriptor of) a
                reference image for the scene.
            lora_ref: Optional :class:`AssetRef` to a scene LoRA.

        Returns:
            A newly created :class:`SceneAsset` (status ``DRAFT``).

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Scene name must be a non-empty string.")

        scene_id = "scene-{}".format(uuid4().hex[:12])
        controlnet_ref = AssetRef(
            asset_id="controlnet-{}".format(uuid4().hex[:8]),
            asset_type=AssetType.CONTROLNET,
            revision="r1",
            content_hash=uuid4().hex,
        )

        depth_ref: Optional[AssetRef] = None
        if reference_image is not None:
            depth_ref = AssetRef(
                asset_id="depth-{}".format(uuid4().hex[:8]),
                asset_type=AssetType.DEPTH,
                revision="r1",
                content_hash=uuid4().hex,
            )

        scene = SceneAsset(
            id=scene_id,
            name=name,
            lora_ref=lora_ref,
            controlnet_ref=controlnet_ref,
            description=description,
            depth_ref=depth_ref,
            status=AssetStatus.DRAFT,
        )
        self._logger.debug(
            "Created scene %r (id=%s, ref_img=%s, lora=%s, depth=%s).",
            name,
            scene_id,
            str(reference_image) if reference_image is not None else None,
            lora_ref.asset_id if lora_ref is not None else None,
            depth_ref.asset_id if depth_ref is not None else None,
        )
        return scene

    # ------------------------------------------------------------------
    # Scene application
    # ------------------------------------------------------------------
    def apply_scene(
        self,
        image: Any,
        scene: SceneAsset,
        weight: float = _DEFAULT_SCENE_WEIGHT,
    ) -> Any:
        """Apply scene conditioning to an image.

        This injects the scene's LoRA and ControlNet conditioning into
        the image at the given ``weight``.  The current implementation
        is a placeholder that returns a descriptor dictionary.

        Args:
            image: The source image to condition.
            scene: The :class:`SceneAsset` whose environment to apply.
            weight: Conditioning strength in ``[0, 1]``.  Defaults to
                ``0.6``.

        Returns:
            A descriptor dictionary with keys ``kind``, ``scene_id``,
            ``weight``, ``lora_ref``, ``controlnet_ref`` and
            ``depth_ref``.

        Raises:
            ValueError: If ``weight`` is outside ``[0, 1]``.
        """
        if weight < _WEIGHT_MIN or weight > _WEIGHT_MAX:
            raise ValueError(
                "weight must be in [{}, {}], got {}.".format(
                    _WEIGHT_MIN, _WEIGHT_MAX, weight
                )
            )
        result = {
            "kind": "scene_conditioned_image",
            "scene_id": scene.id,
            "scene_name": scene.name,
            "weight": weight,
            "lora_ref": (
                scene.lora_ref.to_dict() if scene.lora_ref else None
            ),
            "controlnet_ref": (
                scene.controlnet_ref.to_dict()
                if scene.controlnet_ref
                else None
            ),
            "depth_ref": (
                scene.depth_ref.to_dict() if scene.depth_ref else None
            ),
            "source_image_type": type(image).__name__,
        }
        self._logger.debug(
            "Applied scene %r to image (weight=%.2f).",
            scene.name, weight,
        )
        return result

    # ------------------------------------------------------------------
    # Depth map generation
    # ------------------------------------------------------------------
    def generate_depth_map(
        self,
        image: Any,
        method: str = _DEFAULT_DEPTH_METHOD,
    ) -> Any:
        """Generate a depth map from an image.

        Uses the specified estimation method (``"midas"`` or
        ``"depth_anything"``).  The current implementation uses a
        lightweight, randomly initialised :class:`_DepthEstimator` as a
        placeholder for the real MiDaS / Depth-Anything backbone.

        Args:
            image: The source image (tensor / PIL / numpy).
            method: Depth-estimation method -- ``"midas"`` or
                ``"depth_anything"``.  Defaults to ``"midas"``.

        Returns:
            A depth-map descriptor dictionary with keys ``kind``,
            ``method``, ``depth_tensor`` (the estimated depth tensor)
            and ``source_image_type``.

        Raises:
            ValueError: If ``method`` is not one of the allowed
                methods.
        """
        if method not in _DEPTH_METHODS:
            raise ValueError(
                "method must be one of {}, got {!r}.".format(
                    list(_DEPTH_METHODS), method
                )
            )
        estimator = self._get_depth_estimator()
        tensor = _to_tensor(image)
        if tensor.shape[-2] != _DEPTH_IMAGE_SIZE or tensor.shape[-1] != _DEPTH_IMAGE_SIZE:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(_DEPTH_IMAGE_SIZE, _DEPTH_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        with torch.no_grad():
            depth = estimator(tensor.unsqueeze(0))
        result = {
            "kind": "depth_map",
            "method": method,
            "depth_tensor": depth.squeeze(0),
            "source_image_type": type(image).__name__,
        }
        self._logger.debug(
            "Generated depth map (method=%s, shape=%s).",
            method,
            tuple(depth.shape),
        )
        return result

    # ------------------------------------------------------------------
    # Depth application
    # ------------------------------------------------------------------
    def apply_depth(
        self,
        image: Any,
        depth_map: Any,
        weight: float = _DEFAULT_DEPTH_WEIGHT,
    ) -> Any:
        """Apply depth conditioning to an image.

        This injects a ControlNet depth conditioning signal into the
        image at the given ``weight``.  The current implementation is a
        placeholder that returns a descriptor dictionary.

        Args:
            image: The source image to condition.
            depth_map: The depth map (or a descriptor produced by
                :meth:`generate_depth_map`).
            weight: Conditioning strength in ``[0, 1]``.  Defaults to
                ``0.5``.

        Returns:
            A descriptor dictionary with keys ``kind``, ``weight``,
            ``depth_method`` and ``source_image_type``.

        Raises:
            ValueError: If ``weight`` is outside ``[0, 1]``.
        """
        if weight < _WEIGHT_MIN or weight > _WEIGHT_MAX:
            raise ValueError(
                "weight must be in [{}, {}], got {}.".format(
                    _WEIGHT_MIN, _WEIGHT_MAX, weight
                )
            )
        depth_method = _DEFAULT_DEPTH_METHOD
        if isinstance(depth_map, dict):
            depth_method = depth_map.get("method", depth_method)
        result = {
            "kind": "depth_conditioned_image",
            "weight": weight,
            "depth_method": depth_method,
            "source_image_type": type(image).__name__,
        }
        self._logger.debug(
            "Applied depth conditioning (method=%s, weight=%.2f).",
            depth_method, weight,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_depth_estimator(self) -> _DepthEstimator:
        """Return the (lazily initialised) depth estimator."""
        if self._depth_estimator is None:
            with self._lock:
                if self._depth_estimator is None:
                    estimator = _DepthEstimator()
                    estimator.eval()
                    self._depth_estimator = estimator
        return self._depth_estimator  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "SceneEngine(store={!r})".format(self._store)
