"""Response builders for the serving API (v0.6.x).

Two helpers, both used by the routers in
:mod:`serving.app._routers` and by the CLI in
:mod:`serving.cli`:

* :func:`_make_response` -- build a successful
  :class:`~serving.models.UnifiedResponse`.
* :func:`_error_response` -- build a uniform JSON error envelope
  (consistent error shape across every endpoint).

Kept as their own module so the response-shape code (which
imports the pydantic models) is isolated from the dispatch
logic in :mod:`serving.service._service`.
"""

from __future__ import annotations

import time
from typing import Any

try:  # FastAPI is declared in requirements.txt but guarded
    from fastapi.responses import JSONResponse
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastAPI is required for the serving API. "
        "Install it with: pip install fastapi uvicorn pydantic"
    ) from _exc

from serving.models import Choice, UnifiedResponse, Usage

from ._ids import _estimate_tokens, _generate_id

__all__ = ["_make_response", "_error_response"]


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
