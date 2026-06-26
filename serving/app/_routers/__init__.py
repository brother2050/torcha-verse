"""Router registry (v0.6.x).

Each router module exports a ``build_router(service)`` function
that returns a :class:`fastapi.APIRouter`.  The :func:`register`
function here imports them and attaches the routers to the app
under their existing ``/v1/...`` paths (no prefix changes -- the
public API contract is preserved).
"""

from __future__ import annotations

from fastapi import FastAPI

from serving.service import PipelineService

from ._agent import build_router as _build_agent_router
from ._media import build_router as _build_media_router
from ._multimodal import build_router as _build_multimodal_router
from ._rag import build_router as _build_rag_router
from ._text import build_router as _build_text_router

__all__ = ["register"]


def register(app: FastAPI, service: PipelineService) -> None:
    """Attach every v0.6.x router to ``app``.

    Order is not significant for routing (the routes have unique
    paths), but it is preserved here for readability: text/chat
    first, then media, then multimodal, then RAG, then agent.
    """
    app.include_router(_build_text_router(service))
    app.include_router(_build_media_router(service))
    app.include_router(_build_multimodal_router(service))
    app.include_router(_build_rag_router(service))
    app.include_router(_build_agent_router(service))
