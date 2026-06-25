"""Unified inference serving API for TorchaVerse.

This module exposes the framework's generation, understanding, RAG, and
agent capabilities through a FastAPI-based RESTful API.  It follows an
OpenAI-compatible response shape (``{id, object, created, model, choices,
usage}``) so that existing client libraries can be reused.

Key design decisions:

* **Pipeline / Node service** -- generation capabilities are exposed
  through :class:`PipelineService`, which builds short single-node
  pipelines and runs them via the L5 composer, bridging each node type
  to its L4 executor.  This replaces the deleted ``engines`` package.
* **Server-Sent Events (SSE)** -- streaming endpoints yield ``text/event-
  stream`` responses with ``data: <json>\\n\\n`` frames.
* **Unified error format** -- every error is serialised into a
  ``{"error": {"message", "type", "code"}}`` envelope.
* **Prometheus metrics** -- request counters and latency histograms are
  exposed at ``GET /metrics`` in the Prometheus text exposition format.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from infrastructure.cache_store import CacheStore
from infrastructure.config_manager import ConfigManager
from infrastructure.defaults import (
    DIFFUSION_STEPS,
    DIFFUSION_GUIDANCE_SCALE,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    SAMPLING_REPETITION_PENALTY,
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
from nodes import NodeRegistry
from nodes.base import NodeContext as NodeExecutionContext
from pipeline.composer import NodeContext as PipelineContext, PipelineBuilder

try:  # FastAPI / Pydantic are declared in requirements.txt but guarded
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastAPI and Pydantic are required for the serving API. "
        "Install them with: pip install fastapi uvicorn pydantic"
    ) from _exc

__all__ = [
    "PipelineService",
    "MetricsCollector",
    "TextCompletionRequest",
    "ChatRequest",
    "ImageRequest",
    "AudioRequest",
    "VideoRequest",
    "MultimodalRequest",
    "RAGRequest",
    "AgentRequest",
    "UnifiedResponse",
    "ErrorResponse",
    "create_app",
    "main",
]


# ===========================================================================
# Pydantic request models
# ===========================================================================
class TextCompletionRequest(BaseModel):
    """Request body for ``POST /v1/text/completions``."""

    model: str = "default"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=SAMPLING_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(default=SAMPLING_TOP_K, ge=0)
    top_p: float = Field(default=SAMPLING_TOP_P, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=SAMPLING_REPETITION_PENALTY, ge=1.0, le=2.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    """Request body for ``POST /v1/text/chat``."""

    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=SAMPLING_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(default=SAMPLING_TOP_K, ge=0)
    top_p: float = Field(default=SAMPLING_TOP_P, ge=0.0, le=1.0)
    stream: bool = False


class ImageRequest(BaseModel):
    """Request body for ``POST /v1/images/generate``."""

    model: str = "default"
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=DIFFUSION_STEPS, ge=1, le=200)
    guidance_scale: float = Field(default=DIFFUSION_GUIDANCE_SCALE, ge=0.0, le=30.0)
    seed: Optional[int] = None
    response_format: str = "b64_json"


class AudioRequest(BaseModel):
    """Request body for ``POST /v1/audio/synthesize``."""

    model: str = "default"
    text: str
    speaker_id: int = 0
    emotion: str = "neutral"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = "b64_json"


class VideoRequest(BaseModel):
    """Request body for ``POST /v1/videos/generate``."""

    model: str = "default"
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    num_frames: int = Field(default=16, ge=1, le=128)
    fps: int = Field(default=8, ge=1, le=60)
    steps: int = Field(default=DIFFUSION_STEPS, ge=1, le=200)
    guidance_scale: float = Field(default=DIFFUSION_GUIDANCE_SCALE, ge=0.0, le=30.0)
    seed: Optional[int] = None
    response_format: str = "b64_json"


class MultimodalRequest(BaseModel):
    """Request body for ``POST /v1/multimodal/understand``."""

    model: str = "default"
    text: Optional[str] = None
    image: Optional[str] = None  # base64-encoded image
    audio: Optional[str] = None  # base64-encoded audio
    question: Optional[str] = None
    max_tokens: int = Field(default=256, ge=1, le=8192)


class RAGRequest(BaseModel):
    """Request body for ``POST /v1/rag/query``."""

    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = False


class AgentRequest(BaseModel):
    """Request body for ``POST /v1/agent/run``."""

    task: str
    agent_type: str = "react"
    flow: Optional[str] = None
    max_steps: int = Field(default=10, ge=1, le=50)
    stream: bool = False


# ===========================================================================
# Pydantic response models
# ===========================================================================
class Choice(BaseModel):
    """A single choice in a unified response."""

    index: int = 0
    text: str = ""
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UnifiedResponse(BaseModel):
    """The unified response envelope returned by all generation endpoints."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ErrorResponse(BaseModel):
    """Unified error response."""

    error: Dict[str, Any]


# ===========================================================================
# Metrics collector
# ===========================================================================
class MetricsCollector:
    """Collect and expose Prometheus-format metrics.

    Tracks request counts and latency per endpoint in a thread-safe
    manner.  The metrics are rendered in the Prometheus text exposition
    format for scraping by ``GET /metrics``.
    """

    def __init__(self) -> None:
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._latency_sum: Dict[str, float] = {}
        self._latency_count: Dict[str, int] = {}
        self._engine_loads: Dict[str, int] = {}
        self._start_time: float = time.time()

    def record_request(self, endpoint: str, latency: float, error: bool = False) -> None:
        """Record a completed request.

        Args:
            endpoint: The endpoint name.
            latency: Request latency in seconds.
            error: Whether the request resulted in an error.
        """
        self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1
        self._latency_sum[endpoint] = self._latency_sum.get(endpoint, 0.0) + latency
        self._latency_count[endpoint] = self._latency_count.get(endpoint, 0) + 1
        if error:
            self._error_counts[endpoint] = self._error_counts.get(endpoint, 0) + 1

    def record_engine_load(self, engine_type: str) -> None:
        """Record an engine load event."""
        self._engine_loads[engine_type] = self._engine_loads.get(engine_type, 0) + 1

    def render(self) -> str:
        """Render metrics in Prometheus text exposition format.

        Returns:
            A string suitable for a ``text/plain`` metrics response.
        """
        lines: List[str] = []
        uptime = time.time() - self._start_time

        # Uptime gauge.
        lines.append("# HELP torcha_uptime_seconds Server uptime in seconds.")
        lines.append("# TYPE torcha_uptime_seconds gauge")
        lines.append(f"torcha_uptime_seconds {uptime:.2f}")
        lines.append("")

        # Request counter.
        lines.append("# HELP torcha_requests_total Total number of requests.")
        lines.append("# TYPE torcha_requests_total counter")
        for ep, count in sorted(self._request_counts.items()):
            lines.append(f'torcha_requests_total{{endpoint="{ep}"}} {count}')
        lines.append("")

        # Error counter.
        lines.append("# HELP torcha_errors_total Total number of errors.")
        lines.append("# TYPE torcha_errors_total counter")
        for ep, count in sorted(self._error_counts.items()):
            lines.append(f'torcha_errors_total{{endpoint="{ep}"}} {count}')
        lines.append("")

        # Latency summary.
        lines.append("# HELP torcha_request_latency_seconds_avg Average request latency.")
        lines.append("# TYPE torcha_request_latency_seconds_avg gauge")
        for ep in sorted(self._latency_sum.keys()):
            total = self._latency_sum[ep]
            count = self._latency_count.get(ep, 1)
            avg = total / count if count else 0.0
            lines.append(
                f'torcha_request_latency_seconds_avg{{endpoint="{ep}"}} {avg:.6f}'
            )
        lines.append("")

        # Engine loads.
        lines.append("# HELP torcha_engine_loads_total Total engine load events.")
        lines.append("# TYPE torcha_engine_loads_total counter")
        for et, count in sorted(self._engine_loads.items()):
            lines.append(f'torcha_engine_loads_total{{engine="{et}"}} {count}')

        return "\n".join(lines) + "\n"


# ===========================================================================
# Pipeline service (bridges REST API to the Pipeline/Node system)
# ===========================================================================
class PipelineService:
    """Service layer that bridges the REST API to the Pipeline/Node system.

    Each capability method builds a short single-node :class:`Pipeline`
    via :class:`PipelineBuilder`, runs it through the L5 composer and
    returns the produced node outputs as a dictionary.  Node executors
    are resolved lazily: a per-node-type callable is registered on a
    fresh :class:`PipelineContext` for every run, bridging the L5
    composer to the L4 node system.

    Methods return the node's output dictionary on success (e.g.
    ``{"text": ..., "usage": ...}``) or ``{"error": ..., "error_type":
    ...}`` when the pipeline cannot be built or run.

    Capabilities without a backing node (multimodal understanding, RAG
    query, agent execution) return a ``not_implemented`` error response
    so the REST contract stays intact while the node backends mature.
    """

    def __init__(self) -> None:
        self._cfg: ConfigManager = ConfigManager()
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

        def _executor(inputs: Dict[str, Any], ctx: PipelineContext) -> Dict[str, Any]:
            run_config: Dict[str, Any] = ctx.config.get("node_config", {})
            node_ctx = NodeExecutionContext(config=dict(run_config))
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
            ctx = PipelineContext(
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


# ===========================================================================
# Application factory
# ===========================================================================
def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured :class:`FastAPI` instance with all routes
        registered.
    """
    app = FastAPI(
        title="TorchaVerse Inference API",
        description=(
            "Unified inference API for text, image, audio, video, "
            "multimodal, RAG, and agent capabilities."
        ),
        version="0.3.1",
    )

    # CORS middleware.  Origins are read from the TORCHA_CORS_ORIGINS
    # environment variable (comma-separated).  The default ``"*"`` is
    # permissive and intended for development only -- in production,
    # configure specific origins (e.g. ``https://app.example.com``).
    # ``allow_credentials`` is intentionally omitted: it is incompatible
    # with the wildcard ``allow_origins=["*"]`` and would be silently
    # dropped (or rejected) by the browser.
    cors_origins = os.environ.get("TORCHA_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = PipelineService()

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        service._logger.error("Unhandled exception: %s", exc, exc_info=True)
        # Never leak the raw exception text to the client in production;
        # return a generic message instead.
        return _error_response(
            "Internal Server Error", error_type="internal_error", code=500
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint."""
        device_info = DeviceManager().get_device_info()
        return {
            "status": "healthy",
            "version": "0.3.1",
            "device": device_info.get("device", "cpu"),
            "uptime": time.time() - service.metrics._start_time,
            "node_types": len(service.list_models()),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics() -> str:
        """Prometheus-format metrics endpoint."""
        return service.metrics.render()

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List all registered node types."""
        models = service.list_models()
        return {
            "object": "list",
            "data": models,
        }

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------
    @app.post("/v1/text/completions")
    async def text_completions(request: TextCompletionRequest) -> Any:
        """Generate text from a prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "text_completions"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                request.model = service._sanitizer.sanitize_text(request.model)
                if request.stop:
                    if isinstance(request.stop, str):
                        request.stop = service._sanitizer.sanitize_text(request.stop)
                    elif isinstance(request.stop, list):
                        request.stop = [
                            service._sanitizer.sanitize_text(s) for s in request.stop
                        ]
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(
                request.prompt
            )
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _text_completion_stream(service, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.text_completion(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            text = result.get("text", "")

            # Security Gate 3: filter the generated text output.
            try:
                filter_result = service._filter.filter_text(text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            prompt_tokens = _estimate_tokens(request.prompt)
            response = _make_response(
                model=request.model,
                text=text,
                object_type="text_completion",
                prompt_tokens=prompt_tokens,
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Text completion failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    def _text_completion_stream(
        svc: PipelineService,
        request: TextCompletionRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming text completion.

        The node system returns a complete generation in one shot, so the
        full text is emitted as a single chunk followed by the terminal
        ``[DONE]`` marker -- preserving the SSE contract.
        """
        start = time.time()
        try:
            result = svc.text_completion(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            text = result.get("text", "")

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "text": text, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            # Final frame.
            done_data = {
                "id": _generate_id(),
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "text": "", "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    @app.post("/v1/text/chat")
    async def text_chat(request: ChatRequest) -> Any:
        """Run a multi-turn chat conversation."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "text_chat"
        try:
            # Security Gate 1: sanitise every message's text content.
            try:
                for msg in request.messages:
                    msg.content = service._sanitizer.sanitize_text(msg.content)
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            prompt = _messages_to_prompt(request.messages)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(prompt)
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _chat_stream(service, prompt, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.text_chat(
                prompt=prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            text = result.get("text", "")

            # Security Gate 3: filter the generated text output.
            try:
                filter_result = service._filter.filter_text(text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            prompt_tokens = sum(_estimate_tokens(m.content) for m in request.messages)
            response = _make_response(
                model=request.model,
                text=text,
                object_type="chat.completion",
                prompt_tokens=prompt_tokens,
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Chat failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    def _chat_stream(
        svc: PipelineService,
        prompt: str,
        request: ChatRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming chat."""
        start = time.time()
        try:
            result = svc.text_chat(
                prompt=prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            text = result.get("text", "")

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            done_data = {
                "id": _generate_id(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    @app.post("/v1/images/generate")
    async def images_generate(request: ImageRequest) -> Any:
        """Generate an image from a text prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "images_generate"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                if request.negative_prompt:
                    request.negative_prompt = service._sanitizer.sanitize_text(
                        request.negative_prompt
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.image_txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            image = result.get("image", result)

            # Security Gate 3: filter the generated image output.
            try:
                filter_result = service._filter.filter_image(image)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            payload = _media_payload(image, "image/png")
            response = UnifiedResponse(
                id=_generate_id(),
                object="image",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.prompt) + 1,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Image generation failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Audio synthesis
    # ------------------------------------------------------------------
    @app.post("/v1/audio/synthesize")
    async def audio_synthesize(request: AudioRequest) -> Any:
        """Synthesize speech from text."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "audio_synthesize"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.text = service._sanitizer.sanitize_text(request.text)
                request.model = service._sanitizer.sanitize_text(request.model)
                request.emotion = service._sanitizer.sanitize_text(request.emotion)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.audio_tts(
                text=request.text,
                voice=request.speaker_id,
                speed=request.speed,
                emotion=request.emotion,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            # Security Gate 3: filter the output text content rather than
            # the input.  The audio node currently returns pure media data
            # (no text transcript), so there is nothing to filter.  If a
            # text transcript/caption is added to the output in the future,
            # it should be passed through service._filter.filter_text()
            # before release.
            output_text = str(result.get("text", ""))
            if output_text:
                try:
                    filter_result = service._filter.filter_text(output_text)
                    if not filter_result.passed:
                        return _error_response(
                            "Output filtered: " + filter_result.action,
                            error_type="output_filtered",
                            code=403,
                        )
                except Exception as filter_exc:  # noqa: BLE001
                    service._logger.warning(
                        "Output filter failed, allowing response: %s", filter_exc
                    )

            audio = result.get("audio", result)
            payload = _media_payload(audio, "audio/wav")
            response = UnifiedResponse(
                id=_generate_id(),
                object="audio",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.text),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.text) + 1,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Audio synthesis failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------
    @app.post("/v1/videos/generate")
    async def videos_generate(request: VideoRequest) -> Any:
        """Generate a video from a text prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "videos_generate"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                if request.negative_prompt:
                    request.negative_prompt = service._sanitizer.sanitize_text(
                        request.negative_prompt
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.video_txt2vid(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            video = result.get("video", result)

            # Security Gate 3: filter any text description that may accompany
            # the video.  Currently the video node returns pure media data
            # (no text caption), so there is nothing to filter.  If a text
            # description is added to the output in the future, it should be
            # passed through service._filter.filter_text() before release.
            output_text = str(result.get("text", ""))
            if output_text:
                try:
                    filter_result = service._filter.filter_text(output_text)
                    if not filter_result.passed:
                        return _error_response(
                            "Output filtered: " + filter_result.action,
                            error_type="output_filtered",
                            code=403,
                        )
                except Exception as filter_exc:  # noqa: BLE001
                    service._logger.warning(
                        "Output filter failed, allowing response: %s", filter_exc
                    )

            payload = _media_payload(video, "image/gif")
            response = UnifiedResponse(
                id=_generate_id(),
                object="video",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=request.num_frames,
                    total_tokens=_estimate_tokens(request.prompt) + request.num_frames,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Video generation failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Multimodal understanding
    # ------------------------------------------------------------------
    @app.post("/v1/multimodal/understand")
    async def multimodal_understand(request: MultimodalRequest) -> Any:
        """Understand multi-modal input and optionally answer a question.

        Multimodal understanding is not yet backed by a node; a
        ``not_implemented`` error is returned while the REST contract is
        preserved.
        """
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "multimodal_understand"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                if request.text:
                    request.text = service._sanitizer.sanitize_text(request.text)
                if request.question:
                    request.question = service._sanitizer.sanitize_text(
                        request.question
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.multimodal_understand()
            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            text = str(result.get("text", ""))

            # Security Gate 3: filter the text response.
            try:
                filter_result = service._filter.filter_text(text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            response = _make_response(
                model=request.model,
                text=text,
                object_type="multimodal.understanding",
                prompt_tokens=_estimate_tokens(
                    (request.text or "") + (request.question or "")
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Multimodal understanding failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------
    @app.post("/v1/rag/query")
    async def rag_query(request: RAGRequest) -> Any:
        """Answer a question using retrieval-augmented generation.

        RAG query is not yet backed by a node; a ``not_implemented``
        error is returned while the REST contract is preserved.
        """
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "rag_query"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.question = service._sanitizer.sanitize_text(request.question)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.rag_query()
            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            text = str(result.get("text", ""))

            # Security Gate 3: filter the text response.
            try:
                filter_result = service._filter.filter_text(text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            response = _make_response(
                model="rag",
                text=text,
                object_type="rag.answer",
                prompt_tokens=_estimate_tokens(request.question),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("RAG query failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    # ------------------------------------------------------------------
    # Agent run
    # ------------------------------------------------------------------
    @app.post("/v1/agent/run")
    async def agent_run(request: AgentRequest) -> Any:
        """Execute an agent on a task.

        Agent execution is not yet backed by a node; a
        ``not_implemented`` error is returned while the REST contract is
        preserved.
        """
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "agent_run"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.task = service._sanitizer.sanitize_text(request.task)
                request.agent_type = service._sanitizer.sanitize_text(request.agent_type)
                if request.flow:
                    request.flow = service._sanitizer.sanitize_text(request.flow)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(
                request.task
            )
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _agent_stream(service, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.agent_run()
            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            output_text = str(result.get("output", ""))

            # Security Gate 3: filter the final text response.
            try:
                filter_result = service._filter.filter_text(output_text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            response_dict = {
                "id": _generate_id(),
                "object": "agent.result",
                "created": int(time.time()),
                "model": "agent",
                "choices": [
                    {"index": 0, "text": output_text, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": _estimate_tokens(request.task),
                    "completion_tokens": 0,
                    "total_tokens": _estimate_tokens(request.task),
                },
                "steps": [],
                "metadata": {},
            }
            service.metrics.record_request(endpoint, time.time() - start)
            return response_dict

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Agent run failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    def _agent_stream(
        svc: PipelineService,
        request: AgentRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming agent execution."""
        start = time.time()
        try:
            result = svc.agent_run()
            if "error" in result:
                raise RuntimeError(result["error"])
            output_text = str(result.get("output", ""))

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(output_text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "agent.step",
                "created": int(time.time()),
                "model": "agent",
                "step": {"output": output_text},
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    return app


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """Entry point for the TorchaVerse API server.

    Parses ``--host`` and ``--port`` arguments and launches the
    Uvicorn ASGI server.
    """
    parser = argparse.ArgumentParser(
        description="TorchaVerse Inference API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    args = parser.parse_args()

    logger = get_logger("api_server")
    logger.info("Starting TorchaVerse API on %s:%d", args.host, args.port)

    try:
        import uvicorn

        uvicorn.run(
            "serving.api_server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install it with: pip install uvicorn"
        )
        raise


# Lazy app creation to avoid import side-effects (e.g. binding ports,
# loading config at import time).  Use ``get_app()`` to obtain the
# singleton, or reference ``serving.api_server:create_app`` with
# ``factory=True`` in uvicorn.
app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Return the singleton :class:`FastAPI` app, creating it on first call."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    main()
