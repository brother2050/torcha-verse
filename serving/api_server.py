"""Unified inference serving API for TorchaVerse.

This module exposes the framework's generation, understanding, RAG, and
agent capabilities through a FastAPI-based RESTful API.  It follows an
OpenAI-compatible response shape (``{id, object, created, model, choices,
usage}``) so that existing client libraries can be reused.

Key design decisions:

* **Lazy engine loading** -- engines are instantiated on first use and
  cached as singletons inside :class:`EngineManager`.  The API server
  therefore starts instantly even when no model weights are present.
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
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import torch

from engines.agent_engine import AgentEngine
from engines.audio_engine import AudioEngine, AudioTensor
from engines.image_engine import ImageEngine
from engines.multimodal_engine import MultiModalEngine
from engines.rag_engine import RAGEngine
from engines.text_engine import Message, TextEngine
from engines.video_engine import VideoEngine, VideoTensor
from infrastructure.cache_store import CacheStore
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from infrastructure.rate_limiter import RateLimiter

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
    "EngineManager",
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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ImageRequest(BaseModel):
    """Request body for ``POST /v1/images/generate``."""

    model: str = "default"
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
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
    steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
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
# Engine manager (singleton, lazy loading)
# ===========================================================================
class EngineManager:
    """Singleton manager that lazily instantiates and caches engines.

    Engines are created on first request and reused for subsequent
    calls.  Each engine type is keyed by model name where applicable so
    that multiple models can coexist.
    """

    _instance: Optional["EngineManager"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "EngineManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._cfg: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("EngineManager")
        self._metrics: MetricsCollector = MetricsCollector()

        self._text_engines: Dict[str, TextEngine] = {}
        self._image_engines: Dict[str, ImageEngine] = {}
        self._audio_engine: Optional[AudioEngine] = None
        self._video_engines: Dict[str, VideoEngine] = {}
        self._multimodal_engine: Optional[MultiModalEngine] = None
        self._rag_engine: Optional[RAGEngine] = None
        self._agent_engine: Optional[AgentEngine] = None

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

        self._logger.info("EngineManager initialised.")

    # ------------------------------------------------------------------
    # Text engine
    # ------------------------------------------------------------------
    def get_text_engine(self, model_name: str = "default") -> TextEngine:
        """Return a cached or newly created :class:`TextEngine`.

        Args:
            model_name: The model name to load.

        Returns:
            A :class:`TextEngine` instance.
        """
        if model_name not in self._text_engines:
            self._metrics.record_engine_load("text")
            self._logger.info("Loading TextEngine for model '%s'.", model_name)
            self._text_engines[model_name] = TextEngine(model_name)
        return self._text_engines[model_name]

    # ------------------------------------------------------------------
    # Image engine
    # ------------------------------------------------------------------
    def get_image_engine(self, model_name: str = "default") -> ImageEngine:
        """Return a cached or newly created :class:`ImageEngine`."""
        if model_name not in self._image_engines:
            self._metrics.record_engine_load("image")
            self._logger.info("Loading ImageEngine for model '%s'.", model_name)
            self._image_engines[model_name] = ImageEngine(model_name)
        return self._image_engines[model_name]

    # ------------------------------------------------------------------
    # Audio engine
    # ------------------------------------------------------------------
    def get_audio_engine(self) -> AudioEngine:
        """Return a cached or newly created :class:`AudioEngine`."""
        if self._audio_engine is None:
            self._metrics.record_engine_load("audio")
            self._logger.info("Loading AudioEngine.")
            self._audio_engine = AudioEngine()
        return self._audio_engine

    # ------------------------------------------------------------------
    # Video engine
    # ------------------------------------------------------------------
    def get_video_engine(self, model_name: str = "default") -> VideoEngine:
        """Return a cached or newly created :class:`VideoEngine`."""
        if model_name not in self._video_engines:
            self._metrics.record_engine_load("video")
            self._logger.info("Loading VideoEngine for model '%s'.", model_name)
            self._video_engines[model_name] = VideoEngine(model_name)
        return self._video_engines[model_name]

    # ------------------------------------------------------------------
    # Multimodal engine
    # ------------------------------------------------------------------
    def get_multimodal_engine(self) -> MultiModalEngine:
        """Return a cached or newly created :class:`MultiModalEngine`."""
        if self._multimodal_engine is None:
            self._metrics.record_engine_load("multimodal")
            self._logger.info("Loading MultiModalEngine.")
            self._multimodal_engine = MultiModalEngine()
        return self._multimodal_engine

    # ------------------------------------------------------------------
    # RAG engine
    # ------------------------------------------------------------------
    def get_rag_engine(self) -> RAGEngine:
        """Return a cached or newly created :class:`RAGEngine`."""
        if self._rag_engine is None:
            self._metrics.record_engine_load("rag")
            self._logger.info("Loading RAGEngine.")
            self._rag_engine = RAGEngine()
        return self._rag_engine

    # ------------------------------------------------------------------
    # Agent engine
    # ------------------------------------------------------------------
    def get_agent_engine(self) -> AgentEngine:
        """Return a cached or newly created :class:`AgentEngine`."""
        if self._agent_engine is None:
            self._metrics.record_engine_load("agent")
            self._logger.info("Loading AgentEngine.")
            self._agent_engine = AgentEngine()
        return self._agent_engine

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """Return a list of loaded models with metadata.

        Returns:
            A list of dictionaries, each describing a loaded engine.
        """
        models: List[Dict[str, Any]] = []
        for name, eng in self._text_engines.items():
            models.append({
                "id": name,
                "object": "text",
                "engine": "TextEngine",
                "device": str(eng._device),
            })
        for name, eng in self._image_engines.items():
            models.append({
                "id": name,
                "object": "image",
                "engine": "ImageEngine",
                "device": str(eng._device),
            })
        if self._audio_engine is not None:
            models.append({
                "id": "audio",
                "object": "audio",
                "engine": "AudioEngine",
            })
        for name, eng in self._video_engines.items():
            models.append({
                "id": name,
                "object": "video",
                "engine": "VideoEngine",
                "device": str(eng._device),
            })
        if self._multimodal_engine is not None:
            models.append({
                "id": "multimodal",
                "object": "multimodal",
                "engine": "MultiModalEngine",
            })
        if self._rag_engine is not None:
            models.append({
                "id": "rag",
                "object": "rag",
                "engine": "RAGEngine",
                "index_size": self._rag_engine.index_size,
            })
        if self._agent_engine is not None:
            models.append({
                "id": "agent",
                "object": "agent",
                "engine": "AgentEngine",
                "agents": len(self._agent_engine.agents),
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

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None
        cls._initialized = False


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


def _audio_to_b64(audio: AudioTensor) -> str:
    """Encode an :class:`AudioTensor` to a base64 WAV string."""
    import numpy as np
    import wave

    waveform = audio.numpy()
    if waveform.ndim == 2:
        waveform = waveform[0]  # take first channel
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_to_b64(video: VideoTensor) -> str:
    """Encode a :class:`VideoTensor` to a base64 GIF string.

    Uses an animated GIF as a portable, dependency-light container.
    """
    from PIL import Image as PILImage
    import numpy as np

    frames = video.frames
    if frames.dim() == 5:
        frames = frames[0]
    frames_np = (frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
        "uint8"
    )

    pil_frames = [PILImage.fromarray(f) for f in frames_np]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / video.fps),
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_b64_image(b64_str: str) -> Any:
    """Decode a base64-encoded image string into a PIL image."""
    from PIL import Image as PILImage

    raw = base64.b64decode(b64_str)
    return PILImage.open(io.BytesIO(raw))


def _decode_b64_audio(b64_str: str) -> AudioTensor:
    """Decode a base64-encoded WAV string into an :class:`AudioTensor`."""
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

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    return AudioTensor(waveform=waveform, sample_rate=framerate)


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
        version="0.1.0",
    )

    # CORS middleware.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = EngineManager()

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        manager._logger.error("Unhandled exception: %s", exc, exc_info=True)
        return _error_response(
            str(exc), error_type="internal_error", code=500
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
            "version": "0.1.0",
            "device": device_info.get("device", "cpu"),
            "uptime": time.time() - manager.metrics._start_time,
            "loaded_engines": len(manager.list_models()),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics() -> str:
        """Prometheus-format metrics endpoint."""
        return manager.metrics.render()

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List all loaded models/engines."""
        models = manager.list_models()
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
        start = time.time()
        endpoint = "text_completions"
        try:
            engine = manager.get_text_engine(request.model)

            if request.stream:
                return StreamingResponse(
                    _text_completion_stream(engine, request, manager, endpoint),
                    media_type="text/event-stream",
                )

            result = engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop,
            )

            prompt_tokens = _estimate_tokens(request.prompt)
            response = _make_response(
                model=request.model,
                text=result,
                object_type="text_completion",
                prompt_tokens=prompt_tokens,
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Text completion failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    def _text_completion_stream(
        engine: TextEngine,
        request: TextCompletionRequest,
        mgr: EngineManager,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming text completion."""
        start = time.time()
        try:
            stream = engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stream=True,
                stop=request.stop,
            )
            assert isinstance(stream, Iterator)

            for chunk in stream:
                data = {
                    "id": _generate_id(),
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {"index": 0, "text": chunk, "finish_reason": None}
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

            mgr.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            mgr.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    @app.post("/v1/text/chat")
    async def text_chat(request: ChatRequest) -> Any:
        """Run a multi-turn chat conversation."""
        start = time.time()
        endpoint = "text_chat"
        try:
            engine = manager.get_text_engine(request.model)
            messages = [
                Message(role=m.role, content=m.content) for m in request.messages
            ]

            if request.stream:
                return StreamingResponse(
                    _chat_stream(engine, messages, request, manager, endpoint),
                    media_type="text/event-stream",
                )

            reply = engine.chat(messages, max_tokens=request.max_tokens)
            text = reply.content if isinstance(reply, Message) else str(reply)

            prompt_tokens = sum(_estimate_tokens(m.content) for m in request.messages)
            response = _make_response(
                model=request.model,
                text=text,
                object_type="chat.completion",
                prompt_tokens=prompt_tokens,
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Chat failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    def _chat_stream(
        engine: TextEngine,
        messages: List[Message],
        request: ChatRequest,
        mgr: EngineManager,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming chat."""
        start = time.time()
        try:
            # Use generate with stream for token-by-token output.
            prompt = engine._build_chat_prompt(messages)
            stream = engine.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream=True,
            )
            assert isinstance(stream, Iterator)

            for chunk in stream:
                data = {
                    "id": _generate_id(),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk},
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

            mgr.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            mgr.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    @app.post("/v1/images/generate")
    async def images_generate(request: ImageRequest) -> Any:
        """Generate an image from a text prompt."""
        start = time.time()
        endpoint = "images_generate"
        try:
            engine = manager.get_image_engine(request.model)
            image = engine.txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )

            b64 = _image_to_b64(image)
            response = UnifiedResponse(
                id=_generate_id(),
                object="image",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        text=f"data:image/png;base64,{b64}" if b64 else "",
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.prompt) + 1,
                ),
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Image generation failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Audio synthesis
    # ------------------------------------------------------------------
    @app.post("/v1/audio/synthesize")
    async def audio_synthesize(request: AudioRequest) -> Any:
        """Synthesize speech from text."""
        start = time.time()
        endpoint = "audio_synthesize"
        try:
            engine = manager.get_audio_engine()
            audio = engine.synthesize(
                text=request.text,
                speaker_id=request.speaker_id,
                emotion=request.emotion,
                speed=request.speed,
            )

            b64 = _audio_to_b64(audio)
            response = UnifiedResponse(
                id=_generate_id(),
                object="audio",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        text=f"data:audio/wav;base64,{b64}" if b64 else "",
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.text),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.text) + 1,
                ),
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Audio synthesis failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------
    @app.post("/v1/videos/generate")
    async def videos_generate(request: VideoRequest) -> Any:
        """Generate a video from a text prompt."""
        start = time.time()
        endpoint = "videos_generate"
        try:
            engine = manager.get_video_engine(request.model)
            video = engine.txt2video(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )

            b64 = _video_to_b64(video)
            response = UnifiedResponse(
                id=_generate_id(),
                object="video",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        text=f"data:image/gif;base64,{b64}" if b64 else "",
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=request.num_frames,
                    total_tokens=_estimate_tokens(request.prompt) + request.num_frames,
                ),
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Video generation failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Multimodal understanding
    # ------------------------------------------------------------------
    @app.post("/v1/multimodal/understand")
    async def multimodal_understand(request: MultimodalRequest) -> Any:
        """Understand multi-modal input and optionally answer a question."""
        start = time.time()
        endpoint = "multimodal_understand"
        try:
            engine = manager.get_multimodal_engine()

            image = None
            if request.image:
                image = _decode_b64_image(request.image)

            audio = None
            if request.audio:
                audio = _decode_b64_audio(request.audio)

            result = engine.understand(
                image=image,
                audio=audio,
                text=request.text,
                question=request.question,
                max_tokens=request.max_tokens,
            )

            response = _make_response(
                model=request.model,
                text=result,
                object_type="multimodal.understanding",
                prompt_tokens=_estimate_tokens(
                    (request.text or "") + (request.question or "")
                ),
            )
            manager.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Multimodal understanding failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------
    @app.post("/v1/rag/query")
    async def rag_query(request: RAGRequest) -> Any:
        """Answer a question using retrieval-augmented generation."""
        start = time.time()
        endpoint = "rag_query"
        try:
            engine = manager.get_rag_engine()
            answer, sources = engine.query(
                question=request.question,
                top_k=request.top_k,
                rerank=request.rerank,
            )

            response = UnifiedResponse(
                id=_generate_id(),
                object="rag.answer",
                created=int(time.time()),
                model="rag",
                choices=[
                    Choice(index=0, text=answer.text, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.question),
                    completion_tokens=_estimate_tokens(answer.text),
                    total_tokens=_estimate_tokens(request.question)
                    + _estimate_tokens(answer.text),
                ),
            )
            # Attach sources as extra metadata.
            response_dict = response.model_dump()
            response_dict["sources"] = sources.to_dict()
            response_dict["confidence"] = answer.confidence
            manager.metrics.record_request(endpoint, time.time() - start)
            return response_dict

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("RAG query failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    # ------------------------------------------------------------------
    # Agent run
    # ------------------------------------------------------------------
    @app.post("/v1/agent/run")
    async def agent_run(request: AgentRequest) -> Any:
        """Execute an agent on a task."""
        start = time.time()
        endpoint = "agent_run"
        try:
            engine = manager.get_agent_engine()

            if request.stream:
                return StreamingResponse(
                    _agent_stream(engine, request, manager, endpoint),
                    media_type="text/event-stream",
                )

            if request.flow:
                # Create a multi-agent flow.
                engine.create_agent(
                    role="manager",
                    max_steps=request.max_steps,
                )
                engine.create_agent(
                    role="worker",
                    max_steps=request.max_steps,
                )
                flow = engine.create_flow(
                    agents=["manager", "worker"],
                    topology=request.flow,
                )
                result = engine.execute(flow, request.task)
            else:
                result = engine.run(request.task, max_steps=request.max_steps)

            response_dict = {
                "id": _generate_id(),
                "object": "agent.result",
                "created": int(time.time()),
                "model": "agent",
                "choices": [
                    {"index": 0, "text": result.output, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": _estimate_tokens(request.task),
                    "completion_tokens": _estimate_tokens(result.output),
                    "total_tokens": _estimate_tokens(request.task)
                    + _estimate_tokens(result.output),
                },
                "steps": [s.to_dict() for s in result.steps],
                "metadata": result.metadata,
            }
            manager.metrics.record_request(endpoint, time.time() - start)
            return response_dict

        except Exception as exc:
            manager.metrics.record_request(endpoint, time.time() - start, error=True)
            manager._logger.error("Agent run failed: %s", exc)
            return _error_response(str(exc), error_type="engine_error", code=500)

    def _agent_stream(
        engine: AgentEngine,
        request: AgentRequest,
        mgr: EngineManager,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming agent execution."""
        start = time.time()
        try:
            for step in engine.stream(request.task, max_steps=request.max_steps):
                data = {
                    "id": _generate_id(),
                    "object": "agent.step",
                    "created": int(time.time()),
                    "model": "agent",
                    "step": step.to_dict(),
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"
            mgr.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            mgr.metrics.record_request(endpoint, time.time() - start, error=True)
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
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0).",
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
            "torcha_verse.serving.api_server:create_app",
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


# Module-level app instance for ``uvicorn torcha_verse.serving.api_server:app``.
app: FastAPI = create_app()


if __name__ == "__main__":
    main()
