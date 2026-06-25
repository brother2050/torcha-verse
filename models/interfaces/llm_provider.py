"""LLM provider interface for TorchaVerse.

This module defines the *single* contract that any text-generation
backend must satisfy to be plugged into the framework's agents,
tool-calling loops, and serving endpoints.  The previous codebase
required each consumer (``ReActAgent``, ``ToolCallAgent``,
``nodes/...``) to know about the concrete ``chat`` signature of one
specific model class, which made it impossible to swap backends
(HuggingFace, llama.cpp, a remote OpenAI-compatible API, a stub, etc.)
without patching the consumer.

The :class:`LLMProvider` :class:`typing.Protocol` fixes this by
exposing three operations:

* :meth:`LLMProvider.chat` -- one-shot generation, returning a
  :class:`LLMResponse` that mirrors the OpenAI Chat Completions shape
  (``text`` + ``tool_calls`` + ``usage``).
* :meth:`LLMProvider.stream` -- token-by-token streaming iterator.
* :meth:`LLMProvider.embed` -- dense text embedding for RAG.

Three reference implementations are provided:

* :class:`EchoProvider` -- a deterministic stub that returns the prompt
  verbatim.  Useful for offline tests and CI.
* :class:`ChatTemplateProvider` -- wraps any
  :class:`models.text.transformer.TransformerLM` (HuggingFace-style
  ``generate``) and exposes it through the LLMProvider contract.
* :class:`CallableProvider` -- wraps any Python callable with the
  ``(messages) -> str`` signature, so adapters to vLLM, llama.cpp,
  OpenAI HTTP, etc. can be expressed in a few lines.

Any class that implements the three methods structurally (a Protocol
class) qualifies -- no inheritance required.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    "LLMMessage",
    "LLMToolCall",
    "LLMUsage",
    "LLMResponse",
    "LLMProvider",
    "EchoProvider",
    "CallableProvider",
    "ChatTemplateProvider",
]


# ---------------------------------------------------------------------------
# Message / response dataclasses (OpenAI Chat Completions shape).
# ---------------------------------------------------------------------------
@dataclass
class LLMToolCall:
    """A single tool-call emitted by the model.

    Mirrors the OpenAI ``tool_calls[i]`` shape so adapters can pass
    values through unchanged.  ``arguments`` may be either a ``dict``
    (preferred) or a JSON string (the wire format used by OpenAI).
    """

    id: str = ""
    name: str = ""
    arguments: Any = field(default_factory=dict)


@dataclass
class LLMMessage:
    """One entry in a chat history.

    Roles follow OpenAI conventions: ``"system"``, ``"user"``,
    ``"assistant"``, ``"tool"``.  ``tool_calls`` is populated for
    assistant messages that requested tools; ``tool_call_id`` and
    ``name`` are populated for ``"tool"`` messages replying to a call.
    """

    role: str = "user"
    content: str = ""
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the OpenAI-flavoured JSON representation."""
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                        if isinstance(tc.arguments, str)
                        else _json_dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass
class LLMUsage:
    """Token accounting for a single :meth:`LLMProvider.chat` call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Single non-streaming LLM response.

    The shape mirrors the OpenAI Chat Completions non-streaming
    response, restricted to the fields TorchaVerse actually consumes.
    """

    text: str = ""
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str = "stop"
    response_id: str = field(default_factory=lambda: f"llm-{uuid.uuid4().hex[:16]}")
    created: float = field(default_factory=time.time)
    raw: Any = None  # the backend's native response, if any


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMProvider(Protocol):
    """Pluggable LLM backend.

    Any class that exposes :meth:`chat`, :meth:`stream`, and
    :meth:`embed` with the signatures below satisfies this protocol
    (no inheritance needed).

    Implementations should be cheap to instantiate, free of hidden
    global state, and safe to share between threads.
    """

    #: Free-form backend identifier (``"hf"``, ``"openai"``,
    #: ``"echo"``, ...).  Surfaced in :attr:`LLMResponse.model`.
    name: str

    def chat(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate one assistant response for ``messages``.

        Args:
            messages: Ordered chat history.
            tools: Optional list of tool schemas in OpenAI function
                format ``{"type": "function", "function": {...}}``.
            max_tokens: Hard cap on the number of generated tokens.
            temperature: Sampling temperature in ``[0, 2]``.
            stop: Optional list of stop strings.
            **kwargs: Backend-specific extensions.

        Returns:
            A single :class:`LLMResponse`.
        """
        ...

    def stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text token-by-token.

        Implementations that do not support streaming should yield the
        complete response as a single chunk.
        """
        ...

    def embed(self, text: str) -> List[float]:
        """Return a dense vector embedding for ``text``.

        Used by the RAG stack to project natural-language queries into
        the vector-store space.
        """
        ...


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------
class EchoProvider:
    """Deterministic stub provider.

    Returns the most recent user message verbatim (truncated /
    upper-cased, depending on the constructor flags).  Useful for
    tests, CI, and examples that need a working backend without
    loading model weights.
    """

    name: str = "echo"

    def __init__(self, uppercase: bool = False, repeat: int = 1) -> None:
        self._uppercase = uppercase
        self._repeat = max(1, int(repeat))

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return the last user message, optionally uppercased / repeated."""
        del max_tokens, temperature, stop, tools  # unused
        text = _last_user_text(messages)
        if self._uppercase:
            text = text.upper()
        text = (text + " ") * self._repeat
        text = text.rstrip()
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
                total_tokens=sum(len(m.content.split()) for m in messages) + len(text.split()),
            ),
            model=self.name,
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield the full response in two chunks (prefix + suffix)."""
        del max_tokens, temperature, stop, tools
        text = self.chat(messages).text
        if not text:
            return
        cut = max(1, len(text) // 2)
        yield text[:cut]
        yield text[cut:]

    def embed(self, text: str) -> List[float]:
        """Return a tiny deterministic hash-based embedding.

        The output is a 16-D float vector derived from a SHA-256 digest
        of ``text``.  It is **not** semantically meaningful; the method
        exists so RAG pipelines can be exercised end-to-end without a
        real embedding model.
        """
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


class CallableProvider:
    """Adapter that wraps a Python ``(messages) -> str`` callable.

    This is the smallest possible integration surface for vLLM,
    llama.cpp, OpenAI HTTP, or any other backend that already exposes
    a Python function.  Example::

        def my_chat(messages):
            ...

        provider = CallableProvider("openai-http", my_chat)

    The ``stream`` and ``embed`` methods fall back to the chat output
    (whole response) and a zero vector respectively, so the adapter is
    safe to use even when the underlying backend does not support
    streaming or embeddings natively.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Sequence[LLMMessage]], str],
    ) -> None:
        self.name: str = name
        self._fn: Callable[[Sequence[LLMMessage]], str] = fn

    def chat(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del max_tokens, temperature, stop, tools, kwargs
        text = self._fn(messages)
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
                total_tokens=sum(len(m.content.split()) for m in messages) + len(text.split()),
            ),
            model=self.name,
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        del max_tokens, temperature, stop, tools, kwargs
        text = self._fn(messages)
        if text:
            yield text

    def embed(self, text: str) -> List[float]:
        """Fallback embed: hash-based 16-D vector."""
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


class ChatTemplateProvider:
    """Wrap a :class:`models.text.transformer.TransformerLM` as an LLMProvider.

    The wrapper applies a minimal chat template (``role: content``)
    and a greedy / top-k decode using the model's ``generate`` method.
    This is the canonical "real model" backend used by the example
    notebooks and by the integration tests.
    """

    def __init__(self, model: Any, name: str = "transformer-lm") -> None:
        self._model = model
        self.name: str = name

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del tools  # the chat template below is tool-agnostic
        prompt = _messages_to_prompt(messages)
        gen_kwargs = dict(kwargs)
        gen_kwargs.setdefault("max_new_tokens", max_tokens)
        gen_kwargs.setdefault("temperature", temperature)
        if stop is not None:
            gen_kwargs.setdefault("stop", list(stop))
        text = self._model.generate(prompt, **gen_kwargs)
        if stop is not None:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
                total_tokens=len(prompt.split()) + len(text.split()),
            ),
            model=self.name,
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        del tools
        response = self.chat(messages, max_tokens=max_tokens, temperature=temperature, stop=stop, **kwargs)
        # Naive chunked streaming: one chunk per word.  Real tokenisers
        # can replace this with a proper generator.
        for token in response.text.split(" "):
            if token:
                yield token + " "

    def embed(self, text: str) -> List[float]:
        """Fallback: hash-based 16-D vector.

        A real implementation would call ``self._model.embed(text)``.
        Kept as a stub so the interface is satisfied out of the box.
        """
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _json_dumps(obj: Any) -> str:
    """Serialise ``obj`` as compact JSON, falling back to ``str(obj)``."""
    import json

    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(obj)


def _last_user_text(messages: Sequence[LLMMessage]) -> str:
    """Return the content of the last ``"user"`` message, or empty string."""
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def _messages_to_prompt(messages: Sequence[LLMMessage]) -> str:
    """Flatten a chat history into ``role: content`` lines.

    A simple, model-agnostic template; concrete deployments should
    pass their own template via the ``chat_template`` kwarg of the
    provider's ``chat`` method.
    """
    parts: List[str] = []
    for m in messages:
        role = m.role or "user"
        parts.append(f"{role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)
