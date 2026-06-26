"""Text capability methods for :class:`PipelineService` (v0.6.x).

Kept as their own module so the :class:`PipelineService` class
file (which hosts the dispatcher / constructor / RAG stack)
stays under the soft 500-line cap.

The :func:`attach_text_methods` helper attaches
:meth:`PipelineService.text_completion` and
:meth:`PipelineService.text_chat` to the class at import time
-- this is the v0.4.x "duck-punching" pattern, kept as the
backwards-compatible way of splitting a class across modules
without breaking ``monkeypatch.setattr(service, "text_chat",
...)`` or ``PipelineService.text_chat = my_override`` for tests
that override individual capability methods.
"""

from __future__ import annotations

from typing import Any, Dict

from infrastructure.defaults import SAMPLING_TEMPERATURE

__all__ = ["attach_text_methods"]


def attach_text_methods(cls: type) -> type:
    """Attach text capability methods to ``cls`` (the PipelineService class)."""
    from serving.service._service import PipelineService  # noqa: F401  (for type)

    def text_completion(
        self: "PipelineService",
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
        self: "PipelineService",
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

    cls.text_completion = text_completion
    cls.text_chat = text_chat
    return cls
