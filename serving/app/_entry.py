"""Entry point: CLI launcher and lazy singleton (v0.6.x).

* :func:`main` -- CLI entry point launching the Uvicorn ASGI server.
* :func:`get_app` -- return the singleton :class:`FastAPI` app
  (lazily created on first call).
* :data:`app` -- the optional module-level singleton
  (``None`` until :func:`get_app` is called).

The CLI launcher hard-codes the import string
``"serving.app:create_app"`` -- the public contract is
"call ``create_app`` from the :mod:`serving.app` module",
which continues to work when :mod:`serving.app` is a sub-package
(the import resolves to the ``create_app`` symbol in
:mod:`serving.app.__init__`).
"""

from __future__ import annotations

import argparse
from typing import Optional

from fastapi import FastAPI

from infrastructure.logger import get_logger

from ._factory import create_app

__all__ = ["main", "get_app", "app"]


def main() -> None:
    """Entry point for the TorchaVerse API server.

    Parses ``--host`` and ``--port`` arguments and launches the
    Uvicorn ASGI server.
    """
    parser = argparse.ArgumentParser(
        description="TorchaVerse Inference API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    args = parser.parse_args()

    logger = get_logger("api_server")
    logger.info("Starting TorchaVerse API on %s:%d", args.host, args.port)

    try:
        import uvicorn

        uvicorn.run(
            "serving.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install it with: pip install uvicorn"
        )
        raise


# Lazy app creation to avoid import side-effects (e.g. binding ports,
# loading config at import time).  Use ``get_app()`` to obtain the
# singleton, or reference ``serving.app:create_app`` with
# ``factory=True`` in uvicorn.
app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Return the singleton :class:`FastAPI` app, creating it on first call."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    main()
