"""Consistency-conditioning nodes for the TorchaVerse L4 capability layer.

This module implements the v0.3.0 "consistency framework" -- the four
asset types (character / outfit / scene / depth) plus the multi-view
helper that the v0.1.0 engines had no support for at all.  These nodes
turn :class:`AssetRef` handles into conditioning signals that downstream
generation nodes (e.g. :class:`nodes.image.ImageTxt2ImgNode`) consume.

* :class:`CharacterApplyNode` (``character_apply``) -- apply a character
  asset to a prompt, producing a character-conditioned image.
* :class:`OutfitApplyNode` (``outfit_apply``) -- apply an outfit asset to
  an existing image.
* :class:`SceneApplyNode` (``scene_apply``) -- apply a scene asset to an
  existing image.
* :class:`DepthConditionNode` (``depth_condition``) -- extract a depth map
  from an image or scene via MiDaS / Depth-Anything.
* :class:`FiveViewNode` (``character_five_view``) -- produce a 5-view
  reference sheet (front / back / left / right / 3-quarter) for a
  character.

All five nodes carry a real :meth:`validate_inputs` (dimension ranges,
enum membership for ``method``, non-empty prompts / names) and a real
:meth:`estimate_resources`.  Their :meth:`execute` bodies are
placeholder stubs returning deterministic mock data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from assets.base import AssetRef

from .base import BaseNode, NodeContext, NodeSpec, register_node

__all__ = [
    "CharacterApplyNode",
    "OutfitApplyNode",
    "SceneApplyNode",
    "DepthConditionNode",
    "FiveViewNode",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
#: Inclusive lower bound for image width / height (pixels).
_CONSISTENCY_MIN_DIM: int = 64
#: Inclusive upper bound for image width / height (pixels).
_CONSISTENCY_MAX_DIM: int = 2048
#: Allowed depth-extraction methods.
_DEPTH_METHODS: tuple[str, ...] = ("midas", "depth_anything")
#: VRAM (GB) for the consistency / IP-Adapter model.
_CONSISTENCY_MODEL_VRAM_GB: float = 2.0
#: VRAM (GB) for a depth-extraction model.
_DEPTH_MODEL_VRAM_GB: float = 1.5
#: VRAM (GB) for the five-view generation model.
_FIVE_VIEW_MODEL_VRAM_GB: float = 5.0
#: Number of views produced by :class:`FiveViewNode`.
_FIVE_VIEW_COUNT: int = 5
#: Number of pixels in one megapixel.
_MEGAPIXEL_PIXELS: float = 1_000_000.0


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _ref_id(ref: Any) -> Optional[str]:
    """Return the asset id of ``ref`` for AssetRef, str, or id-bearing objects."""
    if ref is None:
        return None
    if isinstance(ref, AssetRef):
        return ref.asset_id
    if isinstance(ref, str):
        return ref
    if hasattr(ref, "id"):
        return ref.id
    return None


def _validate_dimensions(
    inputs: Dict[str, Any], node_type: str, errors: List[str]
) -> None:
    """Append dimension-range errors for ``width`` / ``height``."""
    for dim_name in ("width", "height"):
        dim = _coerce_int(inputs.get(dim_name))
        if dim is None:
            continue
        if dim < _CONSISTENCY_MIN_DIM or dim > _CONSISTENCY_MAX_DIM:
            errors.append(
                "Input {!r} for node {!r} must be in [{}, {}], got {}.".format(
                    dim_name,
                    node_type,
                    _CONSISTENCY_MIN_DIM,
                    _CONSISTENCY_MAX_DIM,
                    dim,
                )
            )


# ---------------------------------------------------------------------------
# CharacterApplyNode
# ---------------------------------------------------------------------------
@register_node("character_apply")
class CharacterApplyNode(BaseNode):
    """Character-application node (``character_apply``).

    Applies a character asset to a prompt and target resolution,
    producing a character-conditioned image (e.g. an IP-Adapter
    embedding rendered into a base image).

    Inputs:
        character: :class:`AssetRef` to a character asset (required).
        prompt: Text prompt describing the desired shot (required).
        width: Output width in pixels, in ``[64, 2048]``.
        height: Output height in pixels, in ``[64, 2048]``.

    Outputs:
        image: The character-conditioned image.
    """

    spec = NodeSpec(
        type="character_apply",
        name="Character Apply",
        description="Apply a character asset to a prompt to condition an image.",
        inputs={
            "character": "CHARACTER",
            "prompt": "PROMPT",
            "width": "INT",
            "height": "INT",
            "character_weight": "Optional[FLOAT]",
        },
        outputs={
            "image": "IMAGE",
        },
        tags=["consistency", "character", "conditioning"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate character-apply inputs.

        Extends the base checks with:

        * ``width`` / ``height`` in ``[64, 2048]``.
        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)
        _validate_dimensions(inputs, "character_apply", errors)

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'character_apply' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for character application.

        VRAM = consistency model + output megapixels; time scales with
        megapixels.
        """
        width = _coerce_int(inputs.get("width")) or 0
        height = _coerce_int(inputs.get("height")) or 0
        megapixels = (float(width) * float(height)) / _MEGAPIXEL_PIXELS

        vram_gb = _CONSISTENCY_MODEL_VRAM_GB + megapixels * 0.1
        ram_gb = 0.25 + megapixels * 0.05
        time_s = 0.5 + megapixels * 0.5
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Apply a character asset (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``character``, ``prompt``, ``width``, ``height``.

        Returns:
            A dict with ``image``.
        """
        character = inputs.get("character")
        prompt = str(inputs.get("prompt", ""))
        width = _coerce_int(inputs.get("width")) or 512
        height = _coerce_int(inputs.get("height")) or 512
        weight = inputs.get("character_weight")

        ctx.logger.debug(
            "character_apply run_id=%s character=%s %dx%d",
            ctx.run_id, _ref_id(character), width, height,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.character_apply",
                action="apply_character",
                resource_id=_ref_id(character),
                details={
                    "run_id": ctx.run_id,
                    "width": width,
                    "height": height,
                    "prompt": prompt[: 64],
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_character_image",
            "character": _ref_id(character),
            "width": width,
            "height": height,
            "prompt": prompt[: 64],
            "_character_weight": weight,
        }
        return {"image": image}


# ---------------------------------------------------------------------------
# OutfitApplyNode
# ---------------------------------------------------------------------------
@register_node("outfit_apply")
class OutfitApplyNode(BaseNode):
    """Outfit-application node (``outfit_apply``).

    Applies an outfit asset to an existing image (e.g. swapping the
    clothing of a character already present in the image).

    Inputs:
        image: The source image (required).
        outfit: :class:`AssetRef` to an outfit asset (required).

    Outputs:
        image: The image with the outfit applied.
    """

    spec = NodeSpec(
        type="outfit_apply",
        name="Outfit Apply",
        description="Apply an outfit asset to an existing image.",
        inputs={
            "image": "IMAGE",
            "outfit": "OUTFIT",
            "outfit_weight": "Optional[FLOAT]",
        },
        outputs={
            "image": "IMAGE",
        },
        tags=["consistency", "outfit", "conditioning"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate outfit-apply inputs (base type checks only)."""
        return super().validate_inputs(inputs)

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for outfit application."""
        vram_gb = _CONSISTENCY_MODEL_VRAM_GB
        ram_gb = 0.5
        time_s = 1.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Apply an outfit asset (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image``, ``outfit``.

        Returns:
            A dict with ``image``.
        """
        outfit = inputs.get("outfit")
        weight = inputs.get("outfit_weight")

        ctx.logger.debug(
            "outfit_apply run_id=%s outfit=%s",
            ctx.run_id, _ref_id(outfit),
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.outfit_apply",
                action="apply_outfit",
                resource_id=_ref_id(outfit),
                details={"run_id": ctx.run_id},
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_outfit_image",
            "outfit": _ref_id(outfit),
            "_outfit_weight": weight,
        }
        return {"image": image}


# ---------------------------------------------------------------------------
# SceneApplyNode
# ---------------------------------------------------------------------------
@register_node("scene_apply")
class SceneApplyNode(BaseNode):
    """Scene-application node (``scene_apply``).

    Applies a scene asset to an existing image (e.g. replacing the
    background / environment).

    Inputs:
        image: The source image (required).
        scene: :class:`AssetRef` to a scene asset (required).

    Outputs:
        image: The image with the scene applied.
    """

    spec = NodeSpec(
        type="scene_apply",
        name="Scene Apply",
        description="Apply a scene asset to an existing image.",
        inputs={
            "image": "IMAGE",
            "scene": "SCENE",
            "scene_weight": "Optional[FLOAT]",
        },
        outputs={
            "image": "IMAGE",
        },
        tags=["consistency", "scene", "conditioning"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate scene-apply inputs (base type checks only)."""
        return super().validate_inputs(inputs)

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for scene application."""
        vram_gb = _CONSISTENCY_MODEL_VRAM_GB
        ram_gb = 0.5
        time_s = 1.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Apply a scene asset (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image``, ``scene``.

        Returns:
            A dict with ``image``.
        """
        scene = inputs.get("scene")
        weight = inputs.get("scene_weight")

        ctx.logger.debug(
            "scene_apply run_id=%s scene=%s",
            ctx.run_id, _ref_id(scene),
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.scene_apply",
                action="apply_scene",
                resource_id=_ref_id(scene),
                details={"run_id": ctx.run_id},
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_scene_image",
            "scene": _ref_id(scene),
            "_scene_weight": weight,
        }
        return {"image": image}


# ---------------------------------------------------------------------------
# DepthConditionNode
# ---------------------------------------------------------------------------
@register_node("depth_condition")
class DepthConditionNode(BaseNode):
    """Depth-condition extraction node (``depth_condition``).

    Extracts a depth map from an image or scene description using a
    depth-estimation model.

    Inputs:
        image_or_scene: The source image or scene descriptor (required).
        method: Depth-estimation method -- ``"midas"`` or
            ``"depth_anything"``.

    Outputs:
        depth_map: The extracted depth map.
    """

    spec = NodeSpec(
        type="depth_condition",
        name="Depth Condition",
        description="Extract a depth map from an image or scene.",
        inputs={
            "image_or_scene": "IMAGE",
            "method": "TEXT",
            "depth_weight": "Optional[FLOAT]",
        },
        outputs={
            "depth_map": "DEPTH",
        },
        tags=["consistency", "depth", "conditioning"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate depth-condition inputs.

        Extends the base checks with:

        * ``method`` in ``{"midas", "depth_anything"}``.
        """
        errors = super().validate_inputs(inputs)

        method = inputs.get("method")
        if isinstance(method, str) and method not in _DEPTH_METHODS:
            errors.append(
                "Input 'method' for node 'depth_condition' must be one of "
                "{}, got {!r}.".format(list(_DEPTH_METHODS), method)
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for depth extraction."""
        vram_gb = _DEPTH_MODEL_VRAM_GB
        ram_gb = 0.5
        time_s = 0.5
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Extract a depth map (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image_or_scene``, ``method``.

        Returns:
            A dict with ``depth_map``.
        """
        method = str(inputs.get("method", "midas"))
        model = ctx.config.get("default_depth_model")
        weight = inputs.get("depth_weight")

        ctx.logger.debug(
            "depth_condition run_id=%s method=%s",
            ctx.run_id, method,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.depth_condition",
                action="extract_depth",
                resource_id=model,
                details={"run_id": ctx.run_id, "method": method},
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        depth_map = {
            "kind": "placeholder_depth_map",
            "method": method,
            "_depth_weight": weight,
        }
        return {"depth_map": depth_map}


# ---------------------------------------------------------------------------
# FiveViewNode
# ---------------------------------------------------------------------------
@register_node("character_five_view")
class FiveViewNode(BaseNode):
    """Character five-view generation node (``character_five_view``).

    Produces a 5-view reference sheet (front / back / left / right /
    3-quarter) for a character, used to lock character consistency
    across multiple shots.

    Inputs:
        reference_image: A reference image of the character (required).
        character_name: Name of the character (required).

    Outputs:
        five_views: A list of 5 images (front / back / left / right /
            3-quarter).
    """

    spec = NodeSpec(
        type="character_five_view",
        name="Character Five View",
        description="Generate a 5-view reference sheet for a character.",
        inputs={
            "reference_image": "IMAGE",
            "character_name": "TEXT",
        },
        outputs={
            "five_views": "LIST[IMAGE]",
        },
        tags=["consistency", "character", "multiview"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate five-view inputs.

        Extends the base checks with:

        * ``character_name`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        character_name = inputs.get("character_name")
        if isinstance(character_name, str) and not character_name.strip():
            errors.append(
                "Input 'character_name' for node 'character_five_view' must "
                "be a non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for five-view generation.

        Scales with the number of views (5 by default).
        """
        num_views = _FIVE_VIEW_COUNT
        vram_gb = _FIVE_VIEW_MODEL_VRAM_GB
        ram_gb = 0.5 + num_views * 0.1
        time_s = 1.0 + num_views * 1.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a five-view sheet (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``reference_image``, ``character_name``.

        Returns:
            A dict with ``five_views`` (a list of 5 placeholder images).
        """
        character_name = str(inputs.get("character_name", ""))
        model = ctx.config.get("default_five_view_model")

        ctx.logger.debug(
            "character_five_view run_id=%s character=%s",
            ctx.run_id, character_name,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.character_five_view",
                action="generate_five_view",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "character_name": character_name,
                    "num_views": _FIVE_VIEW_COUNT,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        view_labels = ("front", "back", "left", "right", "three_quarter")
        five_views = [
            {
                "kind": "placeholder_five_view",
                "view": label,
                "character_name": character_name,
            }
            for label in view_labels
        ]
        return {"five_views": five_views}
