"""Image / audio / video capability methods for :class:`PipelineService`.

Like :mod:`serving.service._service_text`, the methods are
attached to the :class:`PipelineService` class at import time via
:func:`attach_media_methods` so the public surface (callers and
tests) is unchanged.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any, Dict, Optional

from infrastructure.defaults import DIFFUSION_GUIDANCE_SCALE, DIFFUSION_STEPS

__all__ = ["attach_media_methods"]


def attach_media_methods(cls: type) -> type:
    """Attach image / audio / video capability methods to ``cls``."""
    from serving.service._service import PipelineService  # noqa: F401  (for type)

    def image_txt2img(
        self: "PipelineService",
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = DIFFUSION_STEPS,
        guidance_scale: float = DIFFUSION_GUIDANCE_SCALE,
        seed: Optional[int] = None,
        model: str = "default",
    ) -> Dict[str, Any]:
        """Generate an image through the ``image_txt2img`` node."""
        inputs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        return self._run(
            "image_txt2img",
            "image_txt2img",
            "img",
            inputs,
            config={"default_image_model": model},
        )

    def image_img2img(
        self: "PipelineService",
        image: Any,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        width: int = 512,
        height: int = 512,
        steps: int = DIFFUSION_STEPS,
        guidance_scale: float = DIFFUSION_GUIDANCE_SCALE,
        seed: Optional[int] = None,
        model: str = "default",
    ) -> Dict[str, Any]:
        """Transform an image through the ``image_img2img`` node.

        A minimal :class:`AssetRef` is synthesised for the input image so
        the node's ``AssetRef``-typed input validates.  The node system
        currently produces placeholder output, so a full AssetStore
        round-trip is not required.
        """
        from assets.base import AssetRef
        from assets.types import AssetType

        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            content_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Cannot encode input image: {exc}", "error_type": "invalid_input"}

        ref = AssetRef(
            asset_id="cli-input",
            asset_type=AssetType.SCENE,
            revision="r1",
            content_hash=content_hash,
        )
        inputs: Dict[str, Any] = {
            "input_image": ref,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        return self._run(
            "image_img2img",
            "image_img2img",
            "img",
            inputs,
            config={"default_image_model": model},
        )

    def audio_tts(
        self: "PipelineService",
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        emotion: str = "neutral",
        model: str = "default",
    ) -> Dict[str, Any]:
        """Synthesise speech through the ``audio_tts`` node."""
        return self._run(
            "audio_tts",
            "audio_tts",
            "audio",
            {"text": text, "voice": voice, "speed": speed, "emotion": emotion},
            config={
                "default_tts_model": model,
                "default_tts_sample_rate": 22050,
            },
        )

    def video_txt2vid(
        self: "PipelineService",
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        fps: int = 8,
        steps: int = DIFFUSION_STEPS,
        guidance_scale: float = DIFFUSION_GUIDANCE_SCALE,
        seed: Optional[int] = None,
        model: str = "default",
    ) -> Dict[str, Any]:
        """Generate a video through the ``video_txt2vid`` node."""
        inputs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        return self._run(
            "video_txt2vid",
            "video_txt2vid",
            "video",
            inputs,
            config={"default_video_model": model},
        )

    cls.image_txt2img = image_txt2img
    cls.image_img2img = image_img2img
    cls.audio_tts = audio_tts
    cls.video_txt2vid = video_txt2vid
    return cls
