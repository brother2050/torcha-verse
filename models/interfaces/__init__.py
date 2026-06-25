"""Provider interfaces for TorchaVerse model adapters.

This sub-package exposes pluggable, protocol-shaped abstractions that
let the rest of the framework integrate with arbitrary backends
without coupling to a specific model class:

* :mod:`llm_provider` -- :class:`LLMProvider` for text generation /
  tool calling / embeddings (with three reference implementations:
  :class:`EchoProvider`, :class:`CallableProvider`, and
  :class:`ChatTemplateProvider`).
* :mod:`media_providers` -- :class:`ImageProvider` /
  :class:`AudioProvider` / :class:`VideoProvider` /
  :class:`MultimodalProvider` for the v0.4.x P0 multi-modal
  milestone, with matching :class:`Echo*Provider` reference
  implementations.
"""

from __future__ import annotations

from .llm_provider import (
    CallableProvider,
    ChatTemplateProvider,
    EchoProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMUsage,
)
from .media_providers import (
    AudioProvider,
    EchoAudioProvider,
    EchoImageProvider,
    EchoMultimodalProvider,
    EchoVideoProvider,
    ImageProvider,
    MultimodalProvider,
    VideoProvider,
)

__all__ = [
    # text
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMUsage",
    "EchoProvider",
    "CallableProvider",
    "ChatTemplateProvider",
    # multi-modal
    "ImageProvider",
    "AudioProvider",
    "VideoProvider",
    "MultimodalProvider",
    "EchoImageProvider",
    "EchoAudioProvider",
    "EchoVideoProvider",
    "EchoMultimodalProvider",
]
