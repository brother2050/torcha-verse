"""FastAPI application factory (v0.6.x).

Builds the :class:`FastAPI` application, attaches the CORS
middleware, registers the global exception handler, and
delegates the route definitions to :mod:`serving.app._routers`.

The factory is intentionally thin: every endpoint lives in
:mod:`serving.app._routers` so this module stays under the
soft 500-line cap and the endpoints are easy to navigate.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.device_manager import DeviceManager

from serving.service import PipelineService, _error_response

from . import _routers

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured :class:`FastAPI` instance with all routes
        registered.
    """
    app = FastAPI(
        title="TorchaVerse Inference API",
        description=(
            "Unified inference API for text, image, audio, video, "
            "multimodal, RAG, and agent capabilities."
        ),
        version="0.3.1",
    )

    # CORS middleware.  Origins are read from the TORCHA_CORS_ORIGINS
    # environment variable (comma-separated).  The default ``"*"`` is
    # permissive and intended for development only -- in production,
    # configure specific origins (e.g. ``https://app.example.com``).
    # ``allow_credentials`` is intentionally omitted: it is incompatible
    # with the wildcard ``allow_origins=["*"]`` and would be silently
    # dropped (or rejected) by the browser.
    cors_origins = os.environ.get("TORCHA_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = PipelineService()

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        service._logger.error("Unhandled exception: %s", exc, exc_info=True)
        # Never leak the raw exception text to the client in production;
        # return a generic message instead.
        return _error_response(
            "Internal Server Error", error_type="internal_error", code=500
        )

    # ------------------------------------------------------------------
    # Health / Metrics / List-models  (built directly on the app)
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint."""
        device_info = DeviceManager().get_device_info()
        return {
            "status": "healthy",
            "version": "0.3.1",
            "device": device_info.get("device", "cpu"),
            "uptime": time.time() - service.metrics._start_time,
            "node_types": len(service.list_models()),
        }

    @app.get("/metrics")
    async def metrics() -> str:
        """Prometheus-format metrics endpoint."""
        return service.metrics.render()

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List all registered node types."""
        models = service.list_models()
        return {
            "object": "list",
            "data": models,
        }

    # ------------------------------------------------------------------
    # Domain routers (text / media / multimodal / RAG / agent)
    # ------------------------------------------------------------------
    _routers.register(app, service)

    return app
