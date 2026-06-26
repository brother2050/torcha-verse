"""Id / token-estimate / prompt-serialisation helpers (v0.6.x).

These are pure functions used by the REST / SSE response builders
in :mod:`serving.service._response` and by the CLI's run-time
helpers in :mod:`serving.cli._runtime`.

Kept as their own module so the
:mod:`serving.service._service` file (which hosts the
:class:`PipelineService` class) stays focused on the dispatch
logic and the public surface does not change.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, List

__all__ = ["_generate_id", "_estimate_tokens", "_messages_to_prompt"]


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
