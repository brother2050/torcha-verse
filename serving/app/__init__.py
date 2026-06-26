"""FastAPI application factory for the TorchaVerse inference API (v0.6.x).

This sub-package exposes :func:`create_app` -- the public entry
point that builds and configures the :class:`FastAPI` application
together with all route handlers (text, image, audio, video,
multimodal, RAG, agent) and the Server-Sent Events streaming
generators.

It was extracted from the original monolithic ``app.py``.  The
request/response models live in :mod:`serving.models`, the
:class:`PipelineService` and helpers live in :mod:`serving.service`,
and the :class:`MetricsCollector` lives in :mod:`serving.metrics`.

Sub-modules
-----------

* :mod:`._factory` -- :func:`create_app`, the FastAPI factory
  (CORS, exception handler, health/metrics/list-models, and the
  one-line router registration).
* :mod:`._entry` -- :func:`main` (CLI launcher for ``uvicorn``),
  :func:`get_app` (lazy singleton), and the module-level
  :data:`app` reference.
* :mod:`._routers` -- one sub-module per domain:

  * :mod:`._routers._text` -- ``/v1/text/completions``,
    ``/v1/text/chat`` and their SSE generators.
  * :mod:`._routers._media` -- ``/v1/images/generate``,
    ``/v1/audio/synthesize``, ``/v1/videos/generate``.
  * :mod:`._routers._multimodal` --
    ``/v1/multimodal/understand``.
  * :mod:`._routers._rag` -- ``/v1/rag/query``.
  * :mod:`._routers._agent` -- ``/v1/agent/run`` and its SSE
    generator.

Public surface (preserved from v0.4.x / v0.5.x):

* :func:`create_app` -- build a configured :class:`FastAPI`
  instance.
* :func:`get_app` -- return the singleton app (lazily created).
* :func:`main` -- CLI entry point launching the Uvicorn ASGI
  server.
* :data:`app` -- the optional module-level singleton (``None``
  until :func:`get_app` is called).

The string import path ``"serving.app:create_app"`` (used by
``uvicorn --factory``) keeps working -- it now resolves to the
:func:`create_app` re-export in this sub-package.
"""

from __future__ import annotations

# Re-export the public API at the sub-package level so that
# ``from serving import app`` and ``from serving.app import
# create_app`` and ``from serving.app import get_app`` and
# ``from serving.app import main`` and ``from serving.app import
# app`` all work the same as they did in v0.4.x / v0.5.x.
from ._entry import app, get_app, main
from ._factory import create_app

__all__ = [
    "create_app",
    "get_app",
    "main",
    "app",
]
