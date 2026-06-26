"""FastAPI application factory (v0.6.x).

Builds the :class:`FastAPI` application, attaches the CORS
middleware, registers the global exception handler, and
delegates the route definitions to :mod:`serving.app._routers`.

The factory is intentionally thin: every endpoint lives in
:mod:`serving.app._routers` so this module stays under the
soft 500-line cap and the endpoints are easy to navigate.

R-17 additions:
* ``RequestIDMiddleware`` -- injects ``X-Request-ID`` header
  and stores the id on ``request.state.request_id`` so loggers
  can include it via ``extra={"request_id": ...}``.
* Enhanced ``/health`` -- now returns ``request_id``,
  ``timestamp`` (ISO 8601), and ``config_dir``.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.device_manager import DeviceManager

from serving.service import PipelineService, _error_response

from . import _routers

__all__ = ["create_app"]


# ---------------------------------------------------------------------------
# Request-ID middleware (R-17)
# ---------------------------------------------------------------------------
class RequestIDMiddleware:
    """ASGI middleware that injects a request-id into every request.

    * If the client sends ``X-Request-ID``, it is preserved.
    * Otherwise a new UUID4 is generated.
    * The id is stored on ``request.state.request_id`` and a
      ``X-Request-ID`` header is added to the response.
    * The id is also injected into the logging context via
      ``extra={"request_id": ...}`` so that the
      :class:`~infrastructure.logger.JsonFormatter` picks it up.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self, scope: Any, receive: Any, send: Any
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate the request-id.
        headers = dict(scope.get("headers", []))
        rid = None
        for key, value in headers.items():
            if key == b"x-request-id":
                rid = value.decode("utf-8", errors="replace")
                break
        if not rid:
            rid = uuid.uuid4().hex

        # Stash on scope["state"] so ``request.state.request_id``
        # is available in route handlers.
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = rid

        # Inject a response header by wrapping ``send``.
        async def _send(message: Any) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-request-id", rid.encode("utf-8"))
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


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

    # R-17: request-id middleware (added before CORS so the
    # header is present on every response including errors).
    app.add_middleware(RequestIDMiddleware)

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
        request_id = getattr(
            getattr(request, "state", None), "request_id", None
        )
        service._logger.error(
            "Unhandled exception (request_id=%s): %s",
            request_id, exc, exc_info=True,
            extra={"request_id": request_id} if request_id else {},
        )
        # Never leak the raw exception text to the client in production;
        # return a generic message instead.
        resp = _error_response(
            "Internal Server Error", error_type="internal_error", code=500
        )
        if request_id:
            resp.headers["X-Request-ID"] = request_id
        return resp

    # ------------------------------------------------------------------
    # Health / Metrics / List-models  (built directly on the app)
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health(request: Request) -> Dict[str, Any]:
        """Health check endpoint (R-17 enhanced)."""
        device_info = DeviceManager().get_device_info()
        request_id = getattr(
            getattr(request, "state", None), "request_id", None
        )
        from infrastructure.config_center import ConfigCenter
        cc = ConfigCenter()
        return {
            "status": "healthy",
            "version": "0.3.1",
            "device": device_info.get("device", "cpu"),
            "uptime": time.time() - service.metrics._start_time,
            "node_types": len(service.list_models()),
            # R-17 additions
            "request_id": request_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "config_dir": str(cc.config_dir),
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
