"""Provider interfaces for TorchaVerse model adapters.

This sub-package exposes pluggable, protocol-shaped abstractions that
let the rest of the framework integrate with arbitrary backends
without coupling to a specific model class:

* :mod:`llm_provider` -- :class:`LLMProvider` for text generation /
  tool calling / embeddings (with three reference implementations:
  :class:`EchoProvider`, :class:`CallableProvider`, and
  :class:`ChatTemplateProvider`).
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

__all__ = [
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMUsage",
    "EchoProvider",
    "CallableProvider",
    "ChatTemplateProvider",
]
