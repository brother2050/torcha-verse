"""Image generation nodes for the TorchaVerse L4 capability layer.

This module decomposes the v0.1.0 ``image_engine.py`` "god class" (915
lines) into four single-responsibility nodes:

* :class:`ImageTxt2ImgNode` (``image_txt2img``) -- text-to-image
  generation with optional consistency assets (character / outfit /
  scene / depth) and LoRA adapters.
* :class:`ImageImg2ImgNode` (``image_img2img``) -- image-to-image
  transformation controlled by a ``strength`` parameter.
* :class:`ImageUpscaleNode` (``image_upscale``) -- super-resolution.
* :class:`ImageInpaintNode` (``image_inpaint``) -- masked inpainting.

All four nodes carry a real :meth:`validate_inputs` (spatial dimensions
are clamped to ``[64, 2048]``; ``strength`` to ``[0, 1]``; ``scale`` to a
positive integer) and a real :meth:`estimate_resources` (VRAM scales
with ``width * height * steps``).  Their :meth:`execute` bodies are
placeholder stubs returning deterministic mock data -- the interface is
complete and ready for the real diffusion backend to be wired in via
the :class:`ModuleBus`.

Media types (``image``, ``mask``, ``input_image``) are typed as
:data:`typing.Any` so that this module stays free of heavy imports
(``torch`` / ``PIL``); the concrete tensor / PIL representation is
decided by the backend at execution time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from assets.base import AssetRef

from .base import BaseNode, NodeContext, NodeSpec, register_node
from ._helpers import (
    _MEGAPIXEL_PIXELS,
    coerce_dim as _coerce_dim,
    ref_id as _ref_id,
)

__all__ = [
    "ImageTxt2ImgNode",
    "ImageImg2ImgNode",
    "ImageUpscaleNode",
    "ImageInpaintNode",
]


# ---------------------------------------------------------------------------
# Shared constants (image-specific estimation coefficients)
# ---------------------------------------------------------------------------
#: Inclusive lower bound for image width / height (pixels).
_IMAGE_MIN_DIM: int = 64
#: Inclusive upper bound for image width / height (pixels).
_IMAGE_MAX_DIM: int = 2048
#: VRAM (GB) per (megapixel * step) of diffusion compute.
_IMAGE_VRAM_PER_MPX_STEP_GB: float = 0.002
#: Host RAM (GB) per megapixel of image buffers.
_IMAGE_RAM_PER_MEGAPIXEL_GB: float = 0.05
#: Wall-clock seconds per (megapixel * step) of diffusion compute.
_IMAGE_TIME_PER_MPX_STEP_S: float = 0.10
#: Base VRAM (GB) for the diffusion model weights.
_IMAGE_MODEL_VRAM_GB: float = 4.0

#: 默认图像宽度/高度（像素），用于 execute() 中的回退值。
_DEFAULT_WIDTH: int = 512
#: 默认图像高度（像素），用于 execute() 中的回退值。
_DEFAULT_HEIGHT: int = 512
#: 默认推理步数，用于 execute() 中的回退值。
#: 与 config/inference_config.yaml 中 diffusion.default_steps 保持一致。
_DEFAULT_STEPS: int = 30
#: 默认 img2img 重绘强度，用于 execute() 中的回退值。
_DEFAULT_STRENGTH: float = 0.75
#: 默认放大倍数，用于 execute() 中的回退值。
_DEFAULT_SCALE: int = 2


def _validate_dimensions(
    inputs: Dict[str, Any], node_type: str, errors: List[str]
) -> None:
    """Append dimension-range errors for ``width`` / ``height``."""
    for dim_name in ("width", "height"):
        value = inputs.get(dim_name)
        dim = _coerce_dim(value)
        if dim is None:
            # Missing / wrong-type inputs are reported by the base check.
            continue
        if dim < _IMAGE_MIN_DIM or dim > _IMAGE_MAX_DIM:
            errors.append(
                "Input {!r} for node {!r} must be in [{}, {}], got {}.".format(
                    dim_name,
                    node_type,
                    _IMAGE_MIN_DIM,
                    _IMAGE_MAX_DIM,
                    dim,
                )
            )


# ---------------------------------------------------------------------------
# ImageTxt2ImgNode
# ---------------------------------------------------------------------------
@register_node("image_txt2img")
class ImageTxt2ImgNode(BaseNode):
    """Text-to-image generation node (``image_txt2img``).

    Generates an image from a text prompt, optionally conditioned on
    consistency assets (character / outfit / scene / depth) and one or
    more LoRA adapters.

    Inputs:
        prompt: Positive text prompt (required).
        negative_prompt: Negative text prompt (optional).
        width: Output width in pixels, in ``[64, 2048]``.
        height: Output height in pixels, in ``[64, 2048]``.
        steps: Number of diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Optional deterministic seed.
        character: Optional :class:`AssetRef` to a character asset.
        outfit: Optional :class:`AssetRef` to an outfit asset.
        scene: Optional :class:`AssetRef` to a scene asset.
        depth: Optional :class:`AssetRef` to a depth-map asset.
        loras: Optional list of :class:`AssetRef` LoRA adapters.

    Outputs:
        image: The generated image (PIL.Image or tensor).
        seed: The seed actually used (echoed back for reproducibility).
    """

    spec = NodeSpec(
        type="image_txt2img",
        name="Image Text-to-Image",
        description="Generate an image from a text prompt.",
        inputs={
            "prompt": "PROMPT",
            "negative_prompt": "Optional[PROMPT]",
            "width": "INT",
            "height": "INT",
            "steps": "INT",
            "guidance_scale": "FLOAT",
            "seed": "Optional[SEED]",
            "character": "Optional[CHARACTER]",
            "outfit": "Optional[OUTFIT]",
            "scene": "Optional[SCENE]",
            "depth": "Optional[DEPTH]",
            "loras": "Optional[LIST[LORA]]",
        },
        outputs={
            "image": "IMAGE",
            "seed": "SEED",
        },
        tags=["image", "generation", "diffusion", "txt2img"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate text-to-image inputs.

        Extends the base checks with:

        * ``width`` / ``height`` in ``[64, 2048]``.
        * ``steps`` a positive integer.
        * ``guidance_scale`` non-negative.
        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)
        _validate_dimensions(inputs, "image_txt2img", errors)

        steps = inputs.get("steps")
        if isinstance(steps, int) and steps <= 0:
            errors.append(
                "Input 'steps' for node 'image_txt2img' must be > 0, "
                "got {}.".format(steps)
            )

        guidance = inputs.get("guidance_scale")
        if isinstance(guidance, (int, float)) and float(guidance) < 0:
            errors.append(
                "Input 'guidance_scale' for node 'image_txt2img' must be "
                ">= 0, got {}.".format(guidance)
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'image_txt2img' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time from ``width * height * steps``.

        VRAM = model weights + (megapixels * steps * per-step coefficient);
        time scales with the same product; RAM scales with megapixels.
        """
        width = _coerce_dim(inputs.get("width")) or 0
        height = _coerce_dim(inputs.get("height")) or 0
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, (int, float)) and steps > 0 else 0

        megapixels = (float(width) * float(height)) / _MEGAPIXEL_PIXELS
        mpx_steps = megapixels * float(steps)

        vram_gb = _IMAGE_MODEL_VRAM_GB + mpx_steps * _IMAGE_VRAM_PER_MPX_STEP_GB
        ram_gb = megapixels * _IMAGE_RAM_PER_MEGAPIXEL_GB
        time_s = mpx_steps * _IMAGE_TIME_PER_MPX_STEP_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate an image from a prompt (placeholder implementation).

        .. note::
            Stub returning deterministic mock data; the real diffusion
            backend will be wired through ``ctx.bus.resolve("model.image",
            ...)``.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: See :attr:`spec.inputs`.

        Returns:
            A dict with ``image`` and ``seed``.
        """
        prompt = str(inputs.get("prompt", ""))
        _w = _coerce_dim(inputs.get("width"))
        width = _w if _w is not None else _DEFAULT_WIDTH
        _h = _coerce_dim(inputs.get("height"))
        height = _h if _h is not None else _DEFAULT_HEIGHT
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, int) and steps > 0 else _DEFAULT_STEPS
        seed = inputs.get("seed")
        seed = seed if isinstance(seed, int) else 0
        model = ctx.config.get("default_image_model")

        ctx.logger.debug(
            "image_txt2img run_id=%s %dx%d steps=%d seed=%d",
            ctx.run_id, width, height, steps, seed,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.image_txt2img",
                action="generate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                    "character": _ref_id(inputs.get("character")),
                    "outfit": _ref_id(inputs.get("outfit")),
                    "scene": _ref_id(inputs.get("scene")),
                    "depth": _ref_id(inputs.get("depth")),
                    "num_loras": _num_loras(inputs.get("loras")),
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_image",
            "width": width,
            "height": height,
            "prompt": prompt[: 64],
            "seed": seed,
        }
        return {"image": image, "seed": seed}


# ---------------------------------------------------------------------------
# ImageImg2ImgNode
# ---------------------------------------------------------------------------
@register_node("image_img2img")
class ImageImg2ImgNode(BaseNode):
    """Image-to-image transformation node (``image_img2img``).

    Transforms an input image guided by a text prompt; the ``strength``
    parameter controls how much of the original image is preserved
    (``0`` = identity, ``1`` = full re-generation).

    Inputs:
        input_image: The source image (required).
        prompt: Text prompt guiding the transformation (required).
        negative_prompt: Negative text prompt (optional).
        width: Output width in pixels, in ``[64, 2048]``.
        height: Output height in pixels, in ``[64, 2048]``.
        steps: Number of diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        strength: Transformation strength in ``[0, 1]``.
        seed: Optional deterministic seed.
        character: Optional :class:`AssetRef` to a character asset.
        outfit: Optional :class:`AssetRef` to an outfit asset.
        scene: Optional :class:`AssetRef` to a scene asset.
        depth: Optional :class:`AssetRef` to a depth-map asset.
        loras: Optional list of :class:`AssetRef` LoRA adapters.

    Outputs:
        image: The transformed image (PIL.Image or tensor).
        seed: The seed actually used.
    """

    spec = NodeSpec(
        type="image_img2img",
        name="Image Image-to-Image",
        description="Transform an input image guided by a text prompt.",
        inputs={
            "input_image": "IMAGE",
            "prompt": "PROMPT",
            "negative_prompt": "Optional[PROMPT]",
            "width": "INT",
            "height": "INT",
            "steps": "INT",
            "guidance_scale": "FLOAT",
            "strength": "FLOAT",
            "seed": "Optional[SEED]",
            "character": "Optional[CHARACTER]",
            "outfit": "Optional[OUTFIT]",
            "scene": "Optional[SCENE]",
            "depth": "Optional[DEPTH]",
            "loras": "Optional[LIST[LORA]]",
        },
        outputs={
            "image": "IMAGE",
            "seed": "SEED",
        },
        tags=["image", "generation", "diffusion", "img2img"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate image-to-image inputs.

        Extends the base checks with:

        * ``width`` / ``height`` in ``[64, 2048]``.
        * ``steps`` a positive integer.
        * ``guidance_scale`` non-negative.
        * ``strength`` in ``[0, 1]``.
        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)
        _validate_dimensions(inputs, "image_img2img", errors)

        steps = inputs.get("steps")
        if isinstance(steps, int) and steps <= 0:
            errors.append(
                "Input 'steps' for node 'image_img2img' must be > 0, "
                "got {}.".format(steps)
            )

        guidance = inputs.get("guidance_scale")
        if isinstance(guidance, (int, float)) and float(guidance) < 0:
            errors.append(
                "Input 'guidance_scale' for node 'image_img2img' must be "
                ">= 0, got {}.".format(guidance)
            )

        strength = inputs.get("strength")
        if isinstance(strength, (int, float)) and not (
            0.0 <= float(strength) <= 1.0
        ):
            errors.append(
                "Input 'strength' for node 'image_img2img' must be in "
                "[0, 1], got {}.".format(strength)
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'image_img2img' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources from ``width * height * steps``.

        Scaled by ``strength`` (a partial-strength run does less work).
        """
        width = _coerce_dim(inputs.get("width")) or 0
        height = _coerce_dim(inputs.get("height")) or 0
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, (int, float)) and steps > 0 else 0
        strength = inputs.get("strength")
        strength = (
            float(strength) if isinstance(strength, (int, float)) else 1.0
        )

        megapixels = (float(width) * float(height)) / _MEGAPIXEL_PIXELS
        effective_steps = float(steps) * max(0.0, min(1.0, strength))
        mpx_steps = megapixels * effective_steps

        vram_gb = _IMAGE_MODEL_VRAM_GB + mpx_steps * _IMAGE_VRAM_PER_MPX_STEP_GB
        ram_gb = megapixels * _IMAGE_RAM_PER_MEGAPIXEL_GB
        time_s = mpx_steps * _IMAGE_TIME_PER_MPX_STEP_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Transform an input image (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: See :attr:`spec.inputs`.

        Returns:
            A dict with ``image`` and ``seed``.
        """
        prompt = str(inputs.get("prompt", ""))
        _w = _coerce_dim(inputs.get("width"))
        width = _w if _w is not None else _DEFAULT_WIDTH
        _h = _coerce_dim(inputs.get("height"))
        height = _h if _h is not None else _DEFAULT_HEIGHT
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, int) and steps > 0 else _DEFAULT_STEPS
        strength = inputs.get("strength")
        strength = (
            float(strength) if isinstance(strength, (int, float)) else _DEFAULT_STRENGTH
        )
        seed = inputs.get("seed")
        seed = seed if isinstance(seed, int) else 0
        model = ctx.config.get("default_image_model")

        ctx.logger.debug(
            "image_img2img run_id=%s %dx%d steps=%d strength=%.2f seed=%d",
            ctx.run_id, width, height, steps, strength, seed,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.image_img2img",
                action="transform",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "strength": strength,
                    "seed": seed,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_image",
            "width": width,
            "height": height,
            "prompt": prompt[: 64],
            "strength": strength,
            "seed": seed,
        }
        return {"image": image, "seed": seed}


# ---------------------------------------------------------------------------
# ImageUpscaleNode
# ---------------------------------------------------------------------------
@register_node("image_upscale")
class ImageUpscaleNode(BaseNode):
    """Image super-resolution node (``image_upscale``).

    Upscales an input image by an integer ``scale`` factor using an
    optional super-resolution model.

    Inputs:
        image: The source image (required).
        scale: Integer upscale factor (e.g. ``2`` or ``4``).
        model: Optional registered upscale model name.

    Outputs:
        image: The upscaled image.
    """

    spec = NodeSpec(
        type="image_upscale",
        name="Image Upscale",
        description="Super-resolve an image by an integer scale factor.",
        inputs={
            "image": "IMAGE",
            "scale": "INT",
            "model": "Optional[TEXT]",
        },
        outputs={
            "image": "IMAGE",
        },
        tags=["image", "postprocess", "upscale", "super-resolution"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate upscale inputs.

        Extends the base checks with:

        * ``scale`` a positive integer (typically 2 / 4 / 8).
        """
        errors = super().validate_inputs(inputs)

        scale = inputs.get("scale")
        if isinstance(scale, int) and scale <= 0:
            errors.append(
                "Input 'scale' for node 'image_upscale' must be a positive "
                "integer, got {}.".format(scale)
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources from the output resolution (``scale ** 2``)."""
        scale = inputs.get("scale")
        scale = scale if isinstance(scale, int) and scale > 0 else 1

        # Output megapixels are unknown without the input size; use a
        # per-scale overhead that grows quadratically with the factor.
        vram_gb = 1.0 + (scale * scale) * 0.25
        ram_gb = 0.25 + (scale * scale) * 0.05
        time_s = 0.5 + (scale * scale) * 0.2

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Upscale an image (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image``, ``scale``, ``model``.

        Returns:
            A dict with ``image``.
        """
        scale = inputs.get("scale")
        scale = scale if isinstance(scale, int) and scale > 0 else _DEFAULT_SCALE
        model = inputs.get("model") or ctx.config.get("default_upscale_model")

        ctx.logger.debug(
            "image_upscale run_id=%s scale=%d model=%s",
            ctx.run_id, scale, model,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.image_upscale",
                action="upscale",
                resource_id=model,
                details={"run_id": ctx.run_id, "scale": scale},
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_upscaled_image",
            "scale": scale,
            "model": model,
        }
        return {"image": image}


# ---------------------------------------------------------------------------
# ImageInpaintNode
# ---------------------------------------------------------------------------
@register_node("image_inpaint")
class ImageInpaintNode(BaseNode):
    """Masked inpainting node (``image_inpaint``).

    Fills the region of ``image`` indicated by ``mask`` using the content
    described by ``prompt``.

    Inputs:
        image: The source image (required).
        mask: A mask identifying the region to inpaint (required).
        prompt: Text prompt describing the fill content (required).
        negative_prompt: Negative text prompt (optional).

    Outputs:
        image: The inpainted image.
    """

    spec = NodeSpec(
        type="image_inpaint",
        name="Image Inpaint",
        description="Fill a masked region of an image from a text prompt.",
        inputs={
            "image": "IMAGE",
            "mask": "IMAGE",
            "prompt": "PROMPT",
            "negative_prompt": "Optional[PROMPT]",
        },
        outputs={
            "image": "IMAGE",
        },
        tags=["image", "generation", "diffusion", "inpaint"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inpaint inputs.

        Extends the base checks with:

        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'image_inpaint' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for an inpaint run.

        Without explicit dimensions the estimate uses the model base
        overhead plus a fixed inpaint compute term.
        """
        vram_gb = _IMAGE_MODEL_VRAM_GB + 1.0
        ram_gb = 0.5
        time_s = 2.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Inpaint a masked region (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image``, ``mask``, ``prompt``, ``negative_prompt``.

        Returns:
            A dict with ``image``.
        """
        prompt = str(inputs.get("prompt", ""))
        model = ctx.config.get("default_image_model")

        ctx.logger.debug(
            "image_inpaint run_id=%s prompt=%r",
            ctx.run_id, prompt[: 32],
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.image_inpaint",
                action="inpaint",
                resource_id=model,
                details={"run_id": ctx.run_id, "prompt": prompt[: 64]},
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        image = {
            "kind": "placeholder_inpaint_image",
            "prompt": prompt[: 64],
        }
        return {"image": image}


# ---------------------------------------------------------------------------
# Small helpers shared by the image nodes
# ---------------------------------------------------------------------------
def _num_loras(loras: Any) -> int:
    """Return the number of LoRA adapters in ``loras`` (0 if absent)."""
    if isinstance(loras, list):
        return len(loras)
    return 0
