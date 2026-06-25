"""Pydantic request/response models for the serving API.

This module collects every :class:`pydantic.BaseModel` subclass that
describes the wire contract of the TorchaVerse inference API.  It was
extracted from the original monolithic ``api_server.py`` so that the
models can be reused (and tested) independently of the FastAPI app and
the pipeline service.

Public surface:

* Request models -- :class:`TextCompletionRequest`, :class:`ChatMessage`,
  :class:`ChatRequest`, :class:`ImageRequest`, :class:`AudioRequest`,
  :class:`VideoRequest`, :class:`MultimodalRequest`, :class:`RAGRequest`,
  :class:`AgentRequest`.
* Response models -- :class:`Choice`, :class:`Usage`,
  :class:`UnifiedResponse`, :class:`ErrorResponse`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from infrastructure.defaults import (
    DIFFUSION_GUIDANCE_SCALE,
    DIFFUSION_STEPS,
    SAMPLING_REPETITION_PENALTY,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
)

try:  # FastAPI / Pydantic are declared in requirements.txt but guarded
    from pydantic import BaseModel, Field
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Pydantic is required for the serving API. "
        "Install it with: pip install pydantic"
    ) from _exc

__all__ = [
    "TextCompletionRequest",
    "ChatMessage",
    "ChatRequest",
    "ImageRequest",
    "AudioRequest",
    "VideoRequest",
    "MultimodalRequest",
    "RAGRequest",
    "AgentRequest",
    "Choice",
    "Usage",
    "UnifiedResponse",
    "ErrorResponse",
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
