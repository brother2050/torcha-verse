"""Backward-compatible re-exports for the serving layer.

The original monolithic ``api_server.py`` has been split into focused
modules with a single responsibility each:

* :mod:`serving.models`  -- Pydantic request/response models.
* :mod:`serving.metrics` -- :class:`MetricsCollector`.
* :mod:`serving.service` -- :class:`PipelineService` + helper functions.
* :mod:`serving.app`     -- :func:`create_app` FastAPI factory + routes.

This module remains as a thin compatibility shim so that existing imports
such as ``from serving.api_server import PipelineService`` and the
documented uvicorn entry point ``serving.api_server:create_app`` keep
working unchanged.
"""

from __future__ import annotations

from serving.app import create_app, get_app, main
from serving.metrics import MetricsCollector
from serving.models import *  # noqa: F401, F403
from serving.service import PipelineService

__all__ = [
    "PipelineService",
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
    "get_app",
    "main",
]
