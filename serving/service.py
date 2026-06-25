"""Pipeline service and request/response helpers for the serving API.

This module hosts :class:`PipelineService` -- the service layer that
bridges the REST API to the Pipeline/Node system -- together with the
collection of helper functions used to build responses, estimate tokens
and (de)serialise media payloads.

It was extracted from the original monolithic ``api_server.py``.  The
``NodeContext`` import has been unified: ``nodes.base.NodeContext`` and
``pipeline.composer.NodeContext`` are the *same* class (the composer
re-exports it), so both the former ``NodeExecutionContext`` and
``PipelineContext`` aliases now resolve to a single ``NodeContext``.
"""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from infrastructure.cache_store import CacheStore
from infrastructure.config_center import ConfigCenter
from infrastructure.defaults import (
    DIFFUSION_GUIDANCE_SCALE,
    DIFFUSION_STEPS,
    SAMPLING_TEMPERATURE,
)
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from infrastructure.rate_limiter import RateLimiter
from security.input_sanitizer import InputSanitizer
from security.output_filter import OutputFilter

# Pipeline / Node system (v0.3.0 architecture) -- replaces the deleted
# ``engines`` package.  The service layer builds short single-node
# pipelines and runs them through the L5 composer, bridging each node
# type to its L4 executor.
#
# NOTE: ``pipeline.composer.NodeContext`` is a re-export of
# ``nodes.base.NodeContext`` (they are the *same* class).  We therefore
# import ``NodeContext`` from a single canonical location and use it for
# both the L4 node-execution context and the L5 composer context,
# removing the previous ``NodeExecutionContext`` / ``PipelineContext``
# alias duplication.
from nodes import NodeRegistry
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder

from serving.metrics import MetricsCollector
from serving.models import Choice, UnifiedResponse, Usage

try:  # FastAPI is declared in requirements.txt but guarded
    from fastapi.responses import JSONResponse
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastAPI is required for the serving API. "
        "Install it with: pip install fastapi uvicorn pydantic"
    ) from _exc

__all__ = [
    "PipelineService",
    "_generate_id",
    "_estimate_tokens",
    "_messages_to_prompt",
    "_make_response",
    "_error_response",
    "_image_to_b64",
    "_audio_to_b64",
    "_video_to_b64",
    "_media_payload",
    "_decode_b64_image",
    "_decode_b64_audio",
]


# ===========================================================================
# Pipeline service (bridges REST API to the Pipeline/Node system)
# ===========================================================================
class PipelineService:
    """Service layer that bridges the REST API to the Pipeline/Node system.

    Each capability method builds a short single-node :class:`Pipeline`
    via :class:`PipelineBuilder`, runs it through the L5 composer and
    returns the produced node outputs as a dictionary.  Node executors
    are resolved lazily: a per-node-type callable is registered on a
    fresh :class:`NodeContext` for every run, bridging the L5
    composer to the L4 node system.

    Methods return the node's output dictionary on success (e.g.
    ``{"text": ..., "usage": ...}``) or ``{"error": ..., "error_type":
    ...}`` when the pipeline cannot be built or run.

    Capabilities without a backing node (multimodal understanding, RAG
    query, agent execution) return a ``not_implemented`` error response
    so the REST contract stays intact while the node backends mature.
    """

    def __init__(self) -> None:
        self._cfg: ConfigCenter = ConfigCenter()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("PipelineService")
        self._metrics: MetricsCollector = MetricsCollector()
        self._registry: NodeRegistry = NodeRegistry()

        # Security gates (Gate 1 input sanitiser + Gate 3 output filter).
        self._sanitizer: InputSanitizer = InputSanitizer()
        self._filter: OutputFilter = OutputFilter()

        # Build a reusable executor map (node_type -> callable).  Each
        # executor creates a lightweight L4 NodeContext and dispatches to
        # the registered node, reading run-level config from the L5
        # composer context's metadata.
        self._executors: Dict[str, Any] = {}
        for spec in self._registry.list():
            self._executors[spec.type] = self._make_executor(spec.type)

        # Cache for idempotent generation results.
        cache_cfg = self._cfg.get("serving.cache", {})
        self._cache: CacheStore = CacheStore(
            max_size=cache_cfg.get("max_size", 256),
            ttl=cache_cfg.get("ttl", 300),
        )

        # Rate limiter.
        rate_cfg = self._cfg.get("serving.rate_limit", {})
        self._rate_limiter: RateLimiter = RateLimiter(
            rate=rate_cfg.get("rate", 100),
            burst=rate_cfg.get("burst", 200),
        )

        self._logger.info(
            "PipelineService initialised with %d node executors.",
            len(self._executors),
        )

    # ------------------------------------------------------------------
    # Executor bridge
    # ------------------------------------------------------------------
    def _make_executor(self, node_type: str) -> Any:
        """Create an L5 executor that dispatches to the L4 node ``node_type``.

        The returned callable has the signature ``(inputs, ctx) ->
        outputs`` expected by :class:`pipeline.composer.Pipeline`.  It
        reads the run-level config (model defaults) from the composer
        context's ``config["node_config"]`` bag.
        """
        registry = self._registry

        def _executor(inputs: Dict[str, Any], ctx: NodeContext) -> Dict[str, Any]:
            run_config: Dict[str, Any] = ctx.config.get("node_config", {})
            node_ctx = NodeContext(config=dict(run_config))
            node = registry.get(node_type)
            return node.execute(node_ctx, **inputs)

        return _executor

    def _run(
        self,
        name: str,
        node_type: str,
        node_id: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build, run and return the output of a single-node pipeline.

        Args:
            name: Pipeline name (for logging / serialisation).
            node_type: The L4 node type to execute.
            node_id: The node id used to look up the result.
            inputs: Static inputs forwarded to the node.
            config: Optional run-level config (model defaults).

        Returns:
            The node's output dictionary, or ``{"error": ...}`` on
            failure.
        """
        try:
            ctx = NodeContext(
                executors=self._executors,
                config={"node_config": config or {}},
            )
            pipeline = (
                PipelineBuilder(name)
                .node(node_type, id=node_id, **inputs)
                .build()
            )
            results = pipeline.run(ctx)
            return results.get(node_id, {})
        except Exception as exc:  # noqa: BLE001 - surface as API error
            self._logger.error("Pipeline '%s' failed: %s", name, exc)
            return {"error": str(exc), "error_type": "pipeline_error"}

    # ------------------------------------------------------------------
    # Text capabilities
    # ------------------------------------------------------------------
    def text_completion(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 256,
        temperature: float = SAMPLING_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Run a raw prompt completion through the ``text_completion`` node."""
        return self._run(
            "text_completion",
            "text_completion",
            "completion",
            {"prompt": prompt, "model": model, "max_tokens": max_tokens},
            config={"default_text_model": model},
        )

    def text_chat(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = SAMPLING_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Run a chat-style generation through the ``text_chat`` node."""
        return self._run(
            "text_chat",
            "text_chat",
            "chat",
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            config={"default_text_model": model},
        )

    # ------------------------------------------------------------------
    # Image capabilities
    # ------------------------------------------------------------------
    def image_txt2img(
        self,
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
        self,
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
        import hashlib

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

    # ------------------------------------------------------------------
    # Audio capabilities
    # ------------------------------------------------------------------
    def audio_tts(
        self,
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

    # ------------------------------------------------------------------
    # Video capabilities
    # ------------------------------------------------------------------
    def video_txt2vid(
        self,
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

    # ------------------------------------------------------------------
    # Capabilities not yet backed by a node
    # ------------------------------------------------------------------
    def multimodal_understand(self, **kwargs: Any) -> Dict[str, Any]:
        """Multimodal understanding is not yet available via the node system."""
        return {
            "error": (
                "Multimodal understanding is not yet available via the "
                "Pipeline/Node system."
            ),
            "error_type": "not_implemented",
        }

    def rag_query(self, **kwargs: Any) -> Dict[str, Any]:
        """RAG query is not yet available via the node system."""
        return {
            "error": "RAG query is not yet available via the Pipeline/Node system.",
            "error_type": "not_implemented",
        }

    def agent_run(self, **kwargs: Any) -> Dict[str, Any]:
        """Agent execution is not yet available via the node system."""
        return {
            "error": "Agent execution is not yet available via the Pipeline/Node system.",
            "error_type": "not_implemented",
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """Return the catalogue of registered node types as model metadata."""
        models: List[Dict[str, Any]] = []
        for spec in self._registry.list():
            models.append({
                "id": spec.type,
                "object": "node",
                "name": spec.name,
                "description": spec.description,
                "tags": list(spec.tags),
            })
        return models

    @property
    def metrics(self) -> MetricsCollector:
        """The metrics collector."""
        return self._metrics

    @property
    def cache(self) -> CacheStore:
        """The result cache."""
        return self._cache

    @property
    def rate_limiter(self) -> RateLimiter:
        """The rate limiter."""
        return self._rate_limiter

    @property
    def device_manager(self) -> DeviceManager:
        """The device manager."""
        return self._device_manager


# ===========================================================================
# Helper functions
# ===========================================================================
def _generate_id(prefix: str = "torcha") -> str:
    """Generate a unique response id."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``.

    Uses a simple heuristic of ~4 characters per token.
    """
    return max(1, len(text) // 4)


def _messages_to_prompt(messages: Any) -> str:
    """Flatten a list of chat messages into a single prompt string.

    The node system's ``text_chat`` node accepts a single ``prompt``
    rather than a structured message list, so multi-turn conversations
    are serialised as ``role: content`` lines separated by newlines.
    """
    parts: List[str] = []
    for msg in messages:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _make_response(
    model: str,
    text: str,
    object_type: str = "text_completion",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> UnifiedResponse:
    """Build a :class:`UnifiedResponse`."""
    if completion_tokens == 0:
        completion_tokens = _estimate_tokens(text)
    if prompt_tokens == 0:
        prompt_tokens = completion_tokens
    return UnifiedResponse(
        id=_generate_id(),
        object=object_type,
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, text=text, finish_reason="stop")],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _error_response(
    message: str,
    error_type: str = "internal_error",
    code: int = 500,
) -> JSONResponse:
    """Build a unified error :class:`JSONResponse`."""
    return JSONResponse(
        status_code=code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def _image_to_b64(image: Any) -> str:
    """Encode a PIL image to a base64 JPEG string."""
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _audio_to_b64(audio: Any) -> str:
    """Encode an audio object to a base64 WAV string.

    Accepts either a real audio object exposing ``numpy`` / ``waveform``
    / ``sample_rate`` attributes or a placeholder dict returned by the
    node system (in which case an empty string is returned).
    """
    import numpy as np
    import wave

    waveform = getattr(audio, "numpy", None)
    if waveform is None:
        waveform = getattr(audio, "waveform", None)
    if waveform is None:
        return ""
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform[0]  # take first channel
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    sample_rate = getattr(audio, "sample_rate", 22050)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_to_b64(video: Any) -> str:
    """Encode a video object to a base64 GIF string.

    Accepts either a real video object exposing ``frames`` / ``fps``
    attributes (a tensor or array of shape ``[T, C, H, W]`` or
    ``[T, H, W, C]``) or a placeholder dict returned by the node system
    (in which case an empty string is returned).
    """
    from PIL import Image as PILImage
    import numpy as np

    frames = getattr(video, "frames", None)
    if frames is None:
        return ""
    frames_np = np.asarray(frames)
    if frames_np.ndim == 5:
        frames_np = frames_np[0]
    # Normalise to [T, H, W, C] uint8.
    if frames_np.ndim == 4 and frames_np.shape[-1] not in (1, 3, 4):
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
    frames_np = (np.clip(frames_np, 0, 1) * 255).astype("uint8") \
        if frames_np.dtype.kind == "f" else frames_np.astype("uint8")

    pil_frames = [PILImage.fromarray(f) for f in frames_np]
    if not pil_frames:
        return ""
    fps = getattr(video, "fps", 8)
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _media_payload(media: Any, mime: str) -> str:
    """Build the choice text for a media output.

    Real media objects are base64-encoded into a ``data:<mime>;base64,...``
    URI; placeholder dicts returned by the node system are serialised as
    JSON so the response stays informative even without a real backend.
    """
    b64 = ""
    if mime.startswith("image"):
        b64 = _image_to_b64(media)
    elif mime.startswith("audio"):
        b64 = _audio_to_b64(media)
    elif mime.startswith("video") or mime == "image/gif":
        b64 = _video_to_b64(media)
    if b64:
        return f"data:{mime};base64,{b64}"
    # Placeholder dict or unsupported object -> JSON summary.
    try:
        return json.dumps(media, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(media)


def _decode_b64_image(b64_str: str) -> Any:
    """Decode a base64-encoded image string into a PIL image."""
    from PIL import Image as PILImage

    # Guard against decompression bombs: cap the maximum decoded pixel
    # count and reject oversized base64 payloads before decoding.
    PILImage.MAX_IMAGE_PIXELS = 50_000_000  # 50M pixels limit
    if len(b64_str) > 10 * 1024 * 1024:  # 10MB base64 limit
        raise ValueError("Image too large")

    raw = base64.b64decode(b64_str)
    return PILImage.open(io.BytesIO(raw))


def _decode_b64_audio(b64_str: str) -> Any:
    """Decode a base64-encoded WAV string into a waveform array.

    Returns a plain ``(waveform, sample_rate)`` tuple.  The node system
    operates on plain arrays, so this helper stays free of any
    engine-specific tensor types.
    """
    import numpy as np
    import wave

    raw = base64.b64decode(b64_str)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames_data = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio_np = np.frombuffer(frames_data, dtype=np.int16).astype("float32") / 32768.0
    elif sampwidth == 1:
        audio_np = np.frombuffer(frames_data, dtype=np.uint8).astype("float32") / 128.0 - 1.0
    else:
        audio_np = np.zeros(1024, dtype="float32")

    if n_channels > 1:
        audio_np = audio_np[::n_channels]

    return audio_np, framerate
