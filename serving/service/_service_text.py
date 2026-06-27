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


def _attach_text_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate ``result`` with text-quality diagnostics (v0.10.5).

    The underlying model returns a string that may contain
    U+FFFD replacement characters and ASCII control bytes
    when the project micro-transformer samples ids outside the
    byte vocabulary (random-weight artefact, see
    ``models/providers/_text_sanitiser.py``).  This helper
    re-runs the quality assessment against the *post-sanitise*
    ``text`` field and, when the quality is poor, attaches a
    ``quality_warning`` key that callers (CLI / FastAPI) can
    surface to the user without altering the raw model output.

    The raw model output is preserved as ``raw_text`` so
    debugging tooling (e.g. ``torcha debug text-last``) can
    still inspect it.
    """
    text = result.get("text")
    if not isinstance(text, str):
        return result
    try:
        from models.providers._text_sanitiser import garble_assessment
    except Exception:  # noqa: BLE001 - never break the response
        return result
    level, reason = garble_assessment(text)
    if level != "ok":
        result["quality_warning"] = reason
    result["garble_level"] = level
    return result


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
        result = self._run(
            "text_completion",
            "text_completion",
            "completion",
            {"prompt": prompt, "model": model, "max_tokens": max_tokens},
            config={"default_text_model": model},
        )
        return _attach_text_quality(result)

    def text_chat(
        self: "PipelineService",
        prompt: str,
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = SAMPLING_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Run a chat-style generation through the ``text_chat`` node."""
        result = self._run(
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
        return _attach_text_quality(result)

    cls.text_completion = text_completion
    cls.text_chat = text_chat
    return cls
