"""Video generation nodes for the TorchaVerse L4 capability layer.

This module decomposes the v0.1.0 ``video_engine.py`` "god class" (977
lines) into three single-responsibility nodes:

* :class:`VideoTxt2VidNode` (``video_txt2vid``) -- text-to-video
  generation parameterised by frame count, fps and spatial resolution.
* :class:`VideoInterpolateNode` (``video_interpolate``) -- frame-rate
  interpolation to a target fps.
* :class:`VideoStitchNode` (``video_stitch``) -- concatenation of
  multiple video clips with an optional transition.

All three nodes carry a real :meth:`validate_inputs` (frame / fps /
dimension ranges) and a real :meth:`estimate_resources` (VRAM scales
with ``width * height * num_frames * steps``).  Their :meth:`execute`
bodies are placeholder stubs returning deterministic mock data.

Media types (``video``, ``videos``) are typed as :data:`typing.Any` so
that this module stays free of heavy imports (``torch`` / ``av``); the
concrete tensor / file-path representation is decided by the backend at
execution time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from infrastructure.defaults import DIFFUSION_STEPS

from .base import BaseNode, NodeContext, NodeSpec, register_node
from ._helpers import (
    _MEGAPIXEL_PIXELS,
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
)

__all__ = [
    "VideoTxt2VidNode",
    "VideoInterpolateNode",
    "VideoStitchNode",
]


# ---------------------------------------------------------------------------
# Shared constants (video-specific estimation coefficients)
# ---------------------------------------------------------------------------
#: Inclusive lower bound for video width / height (pixels).
_VIDEO_MIN_DIM: int = 64
#: Inclusive upper bound for video width / height (pixels).
_VIDEO_MAX_DIM: int = 2048
#: Minimum supported frame count.
_VIDEO_MIN_FRAMES: int = 1
#: Maximum supported frame count (guard against absurd requests).
_VIDEO_MAX_FRAMES: int = 1024
#: Minimum supported frame rate (fps).
_VIDEO_MIN_FPS: int = 1
#: Maximum supported frame rate (fps).
_VIDEO_MAX_FPS: int = 120
#: VRAM (GB) per (megapixel * frame * step) of video diffusion compute.
_VIDEO_VRAM_PER_MPX_FRAME_STEP_GB: float = 0.0008
#: Host RAM (GB) per (megapixel * frame) of video buffers.
_VIDEO_RAM_PER_MPX_FRAME_GB: float = 0.01
#: Wall-clock seconds per (megapixel * frame * step) of video diffusion.
_VIDEO_TIME_PER_MPX_FRAME_STEP_S: float = 0.05
#: Base VRAM (GB) for the video diffusion model weights.
_VIDEO_MODEL_VRAM_GB: float = 6.0

#: 默认视频帧数，用于 execute() 中的回退值。
_DEFAULT_NUM_FRAMES: int = 16
#: 默认帧率（fps），用于 execute() 中的回退值。
_DEFAULT_FPS: int = 24
#: 默认视频宽度/高度（像素），用于 execute() 中的回退值。
_DEFAULT_WIDTH: int = 512
#: 默认视频高度（像素），用于 execute() 中的回退值。
_DEFAULT_HEIGHT: int = 512
#: 默认插帧目标帧率（fps），用于 execute() 中的回退值。
_DEFAULT_TARGET_FPS: int = 60


# ---------------------------------------------------------------------------
# VideoTxt2VidNode
# ---------------------------------------------------------------------------
@register_node("video_txt2vid")
class VideoTxt2VidNode(BaseNode):
    """Text-to-video generation node (``video_txt2vid``).

    Generates a video clip from a text prompt.

    Inputs:
        prompt: Positive text prompt (required).
        negative_prompt: Negative text prompt (optional).
        num_frames: Number of frames to generate.
        fps: Output frame rate.
        width: Output width in pixels, in ``[64, 2048]``.
        height: Output height in pixels, in ``[64, 2048]``.
        steps: Number of diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Optional deterministic seed.

    Outputs:
        video: The generated video (tensor or file path).
        seed: The seed actually used.
    """

    spec = NodeSpec(
        type="video_txt2vid",
        name="Video Text-to-Video",
        description="Generate a video clip from a text prompt.",
        inputs={
            "prompt": "PROMPT",
            "negative_prompt": "Optional[PROMPT]",
            "num_frames": "INT",
            "fps": "INT",
            "width": "INT",
            "height": "INT",
            "steps": "INT",
            "guidance_scale": "FLOAT",
            "seed": "Optional[SEED]",
        },
        outputs={
            "video": "VIDEO",
            "seed": "SEED",
        },
        tags=["video", "generation", "diffusion", "txt2vid"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate text-to-video inputs.

        Extends the base checks with:

        * ``width`` / ``height`` in ``[64, 2048]``.
        * ``num_frames`` in ``[1, 1024]``.
        * ``fps`` in ``[1, 120]``.
        * ``steps`` a positive integer.
        * ``guidance_scale`` non-negative.
        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        for dim_name in ("width", "height"):
            dim = _coerce_int(inputs.get(dim_name))
            if dim is None:
                continue
            if dim < _VIDEO_MIN_DIM or dim > _VIDEO_MAX_DIM:
                errors.append(
                    "Input {!r} for node 'video_txt2vid' must be in "
                    "[{}, {}], got {}.".format(
                        dim_name, _VIDEO_MIN_DIM, _VIDEO_MAX_DIM, dim
                    )
                )

        num_frames = _coerce_int(inputs.get("num_frames"))
        if num_frames is not None and not (
            _VIDEO_MIN_FRAMES <= num_frames <= _VIDEO_MAX_FRAMES
        ):
            errors.append(
                "Input 'num_frames' for node 'video_txt2vid' must be in "
                "[{}, {}], got {}.".format(
                    _VIDEO_MIN_FRAMES, _VIDEO_MAX_FRAMES, num_frames
                )
            )

        fps = _coerce_int(inputs.get("fps"))
        if fps is not None and not (
            _VIDEO_MIN_FPS <= fps <= _VIDEO_MAX_FPS
        ):
            errors.append(
                "Input 'fps' for node 'video_txt2vid' must be in "
                "[{}, {}], got {}.".format(
                    _VIDEO_MIN_FPS, _VIDEO_MAX_FPS, fps
                )
            )

        steps = inputs.get("steps")
        if isinstance(steps, int) and steps <= 0:
            errors.append(
                "Input 'steps' for node 'video_txt2vid' must be > 0, "
                "got {}.".format(steps)
            )

        guidance = inputs.get("guidance_scale")
        if isinstance(guidance, (int, float)) and float(guidance) < 0:
            errors.append(
                "Input 'guidance_scale' for node 'video_txt2vid' must be "
                ">= 0, got {}.".format(guidance)
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'video_txt2vid' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time from ``width * height * num_frames * steps``.

        VRAM = model weights + (megapixels * frames * steps * coefficient);
        time scales with the same product; RAM scales with megapixels *
        frames (the decoded buffer).
        """
        width = _coerce_int(inputs.get("width")) or 0
        height = _coerce_int(inputs.get("height")) or 0
        num_frames = _coerce_int(inputs.get("num_frames")) or 0
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, (int, float)) and steps > 0 else 0

        megapixels = (float(width) * float(height)) / _MEGAPIXEL_PIXELS
        mpx_frames_steps = megapixels * float(num_frames) * float(steps)

        vram_gb = (
            _VIDEO_MODEL_VRAM_GB
            + mpx_frames_steps * _VIDEO_VRAM_PER_MPX_FRAME_STEP_GB
        )
        ram_gb = megapixels * float(num_frames) * _VIDEO_RAM_PER_MPX_FRAME_GB
        time_s = mpx_frames_steps * _VIDEO_TIME_PER_MPX_FRAME_STEP_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a video from a prompt (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: See :attr:`spec.inputs`.

        Returns:
            A dict with ``video`` and ``seed``.
        """
        prompt = str(inputs.get("prompt", ""))
        num_frames = _coerce_int(inputs.get("num_frames")) or _DEFAULT_NUM_FRAMES
        fps = _coerce_int(inputs.get("fps")) or _DEFAULT_FPS
        _w = _coerce_int(inputs.get("width"))
        width = _w if _w is not None else _DEFAULT_WIDTH
        _h = _coerce_int(inputs.get("height"))
        height = _h if _h is not None else _DEFAULT_HEIGHT
        steps = inputs.get("steps")
        steps = steps if isinstance(steps, int) and steps > 0 else DIFFUSION_STEPS
        seed = inputs.get("seed")
        seed = seed if isinstance(seed, int) else 0
        model = ctx.config.get("default_video_model")

        ctx.logger.debug(
            "video_txt2vid run_id=%s %dx%d frames=%d fps=%d steps=%d seed=%d",
            ctx.run_id, width, height, num_frames, fps, steps, seed,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.video_txt2vid",
                action="generate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "num_frames": num_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                },
                severity="info",
            )

        from ._helpers import (
            call_motion_module_backend, call_video_backend,
        )

        result = call_video_backend(
            ctx.bus,
            model,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            seed=int(seed),
        )
        # F-9: motion module -- if the backend produced a tensor of
        # hidden states, pipe it through the real MotionModule to
        # inject cross-frame attention.  Backends that return a
        # metadata-only dict (echo) are left untouched.
        frames_obj = None
        if isinstance(result, dict):
            cand = result.get("frames")
            if cand is None:
                cand = result.get("video")
            if cand is not None and not isinstance(cand, bool):
                frames_obj = cand
        # Only call MotionModule when the object exposes a torch-like
        # ``dim()`` method (i.e. it is a tensor).
        has_dim = (
            frames_obj is not None
            and callable(getattr(frames_obj, "dim", None))
        )
        if has_dim:
            motion_scale = float(inputs.get("motion_scale", 1.0))
            motion = call_motion_module_backend(
                ctx.bus,
                ctx.config.get("default_motion_model"),
                hidden_states=frames_obj,
                num_frames=int(num_frames),
                motion_scale=motion_scale,
            )
            if isinstance(motion, dict) and motion.get("backend") != "placeholder":
                if isinstance(result, dict):
                    result["hidden_states"] = motion.get(
                        "hidden_states", frames_obj,
                    )
                    result["motion_scale"] = motion_scale
        return {"video": result, "seed": int(seed)}


# ---------------------------------------------------------------------------
# VideoInterpolateNode
# ---------------------------------------------------------------------------
@register_node("video_interpolate")
class VideoInterpolateNode(BaseNode):
    """Frame-rate interpolation node (``video_interpolate``).

    Increases the frame rate of an input video to ``target_fps`` using a
    frame-interpolation model.

    Inputs:
        video: The source video (required).
        target_fps: Desired output frame rate.

    Outputs:
        video: The interpolated video.
    """

    spec = NodeSpec(
        type="video_interpolate",
        name="Video Interpolate",
        description="Interpolate video frames to a target frame rate.",
        inputs={
            "video": "VIDEO",
            "target_fps": "INT",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["video", "postprocess", "interpolate", "frame"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate interpolation inputs.

        Extends the base checks with:

        * ``target_fps`` in ``[1, 120]``.
        """
        errors = super().validate_inputs(inputs)

        target_fps = _coerce_int(inputs.get("target_fps"))
        if target_fps is not None and not (
            _VIDEO_MIN_FPS <= target_fps <= _VIDEO_MAX_FPS
        ):
            errors.append(
                "Input 'target_fps' for node 'video_interpolate' must be in "
                "[{}, {}], got {}.".format(
                    _VIDEO_MIN_FPS, _VIDEO_MAX_FPS, target_fps
                )
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for interpolation.

        Without the source frame count the estimate uses a base overhead
        plus a per-target-fps term.
        """
        target_fps = _coerce_int(inputs.get("target_fps")) or 0
        vram_gb = 2.0 + (target_fps / 60.0) * 1.0
        ram_gb = 0.5 + (target_fps / 60.0) * 0.5
        time_s = 1.0 + target_fps * 0.1
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Interpolate video frames (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``video``, ``target_fps``.

        Returns:
            A dict with ``video``.
        """
        target_fps = _coerce_int(inputs.get("target_fps")) or _DEFAULT_TARGET_FPS
        model = ctx.config.get("default_interpolate_model")

        ctx.logger.debug(
            "video_interpolate run_id=%s target_fps=%d",
            ctx.run_id, target_fps,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.video_interpolate",
                action="interpolate",
                resource_id=model,
                details={"run_id": ctx.run_id, "target_fps": target_fps},
                severity="info",
            )

        from ._helpers import (
            call_frame_interpolation_backend, call_video_backend,
        )

        source_fps = _coerce_int(inputs.get("source_fps")) or 24
        # Frame interpolation: when the source video is a path or a
        # tensor, run the real FrameInterpolator forward pass.  For
        # generic backends that do not understand interpolation we
        # still call the video backend (which may opt to ignore the
        # ``target_fps`` keyword and return the original frames).
        result = call_frame_interpolation_backend(
            ctx.bus,
            model or "frame-interpolator",
            frames=inputs.get("video"),
            target_fps=int(target_fps),
            source_fps=int(source_fps),
        )
        if result.get("backend") == "placeholder":
            video_result = call_video_backend(
                ctx.bus,
                model or "frame-interpolator",
                prompt="interpolate",
                num_frames=0,
                fps=int(target_fps),
                input_video=inputs.get("video"),
            )
            if isinstance(video_result, dict):
                result.update(video_result)
        return {"video": result}


# ---------------------------------------------------------------------------
# Video understand node (added in v0.4.4)
# ---------------------------------------------------------------------------
@register_node("video_understand")
class VideoUnderstandNode(BaseNode):
    """Multimodal video-understanding node (``video_understand``).

    Takes a list of video frames (tensors / PIL images / paths)
    and an optional text question, and returns the multimodal
    model's textual response describing the video.  Internally
    uses the :class:`LocalTorchMultimodalProvider` via
    :func:`call_multimodal_backend`.

    Inputs:
        frames: A list of frame tensors / PIL images (required).
        question: Optional text question; defaults to
            ``"Describe the video in detail."``.
        max_new_tokens: Cap on the answer length (default 128).

    Outputs:
        text: The model's response.
    """

    spec: NodeSpec = NodeSpec(
        type="video_understand",
        name="Video Understand",
        description="Use a multimodal model to describe or answer questions about a video frame sequence.",
        inputs={
            "frames": "JSON",
            "question": "Optional[TEXT]",
            "max_new_tokens": "Optional[INT]",
        },
        outputs={"text": "TEXT", "raw": "JSON"},
        tags=["video", "understanding", "multimodal"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from ._helpers import call_multimodal_backend

        frames = inputs.get("frames")
        if not frames:
            raise ValueError("video_understand requires a non-empty `frames` list.")
        question = str(inputs.get("question", "Describe the video in detail."))
        max_new_tokens = int(inputs.get("max_new_tokens", 128) or 128)
        payload: Dict[str, Any] = {"text": question, "frames": list(frames)}
        result = call_multimodal_backend(
            ctx.bus, None, input=payload, max_new_tokens=max_new_tokens
        )
        text = str(result.get("text", "")).strip()
        return {"text": text, "raw": result}


# ---------------------------------------------------------------------------
# VideoStitchNode
# ---------------------------------------------------------------------------
@register_node("video_stitch")
class VideoStitchNode(BaseNode):
    """Video stitching node (``video_stitch``).

    Concatenates a list of video clips into a single clip, optionally
    applying a transition between consecutive clips.

    Inputs:
        videos: List of input video clips (required).
        transition: Optional transition name (e.g. ``"crossfade"``).

    Outputs:
        video: The stitched video.
    """

    spec = NodeSpec(
        type="video_stitch",
        name="Video Stitch",
        description="Concatenate multiple video clips with an optional transition.",
        inputs={
            "videos": "LIST[VIDEO]",
            "transition": "Optional[TEXT]",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["video", "postprocess", "stitch", "concat"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate stitching inputs.

        Extends the base checks with:

        * ``videos`` must be a non-empty list.
        """
        errors = super().validate_inputs(inputs)

        videos = inputs.get("videos")
        if isinstance(videos, list) and len(videos) == 0:
            errors.append(
                "Input 'videos' for node 'video_stitch' must contain at "
                "least one clip."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for stitching.

        Scales with the number of input clips.
        """
        videos = inputs.get("videos")
        num_clips = len(videos) if isinstance(videos, list) else 0
        vram_gb = 0.5 + num_clips * 0.25
        ram_gb = 0.25 + num_clips * 0.25
        time_s = 0.5 + num_clips * 0.5
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Stitch video clips (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``videos``, ``transition``.

        Returns:
            A dict with ``video``.
        """
        videos = inputs.get("videos")
        videos = videos if isinstance(videos, list) else []
        transition = inputs.get("transition")

        ctx.logger.debug(
            "video_stitch run_id=%s clips=%d transition=%s",
            ctx.run_id, len(videos), transition,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.video_stitch",
                action="stitch",
                resource_id=None,
                details={
                    "run_id": ctx.run_id,
                    "num_clips": len(videos),
                    "transition": transition,
                },
                severity="info",
            )

        from ._helpers import (
            call_video_backend, call_video_stitch_backend,
        )

        # F-13: real stitch via ffmpeg (when all videos are paths
        # and ffmpeg is on PATH) or a torch cross-fade fallback.
        transition_frames = int(inputs.get("transition_frames", 8))
        stitch = call_video_stitch_backend(
            ctx.bus, "video_stitch",
            videos=list(videos),
            transition=str(transition),
            transition_frames=transition_frames,
        )
        result = call_video_backend(
            ctx.bus,
            "video-stitcher",
            prompt="stitch",
            num_frames=0,
            fps=0,
            input_videos=list(videos),
            transition=transition,
        )
        if isinstance(stitch, dict) and stitch.get("backend") != "placeholder":
            if "frames" in stitch and stitch["frames"] is not None:
                result["stitched_frames"] = stitch["frames"]
            result["stitch_backend"] = stitch.get("backend")
            result["num_videos"] = stitch.get("num_videos", len(videos))
        return {"video": result}
