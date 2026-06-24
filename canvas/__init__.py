"""L5 Canvas layer for the TorchaVerse v0.3.0 architecture.

This package provides the visual *canvas* system that sits above the L4
pipeline layer.  A canvas is a spatial, editable representation of a pipeline
DAG: nodes are placed at 2-D coordinates, connections are drawn between their
ports, and the whole thing can be serialised, versioned, shared and
auto-generated.  Canvas 是 L5 Pipeline 的可视化前端，与 Pipeline 同处 L5 层。

Layering (L1 -> L6):

* L1 ``infrastructure`` -- config, logging, devices, caching.
* L2 ``assets`` -- the asset model + store.
* L3 ``core`` -- module bus, model registry, schedulers, tokenizers.
* L4 ``nodes`` -- the node system (23 composable capability nodes).
* L5 ``pipeline`` / ``canvas`` (this package) -- DAG, composer, templates,
  prompt studio, visual canvas, versioning, sharing, AutoDirector v2,
  community registry.
* L6 ``consistency`` -- character, outfit, scene, depth, pipeline, scoring.

The canvas layer is deliberately *torch-free*: it never imports
:mod:`torch` directly.  It depends only on the L5 pipeline layer
(:class:`~pipeline.dag.DAG`, :class:`~pipeline.composer.Pipeline`,
:class:`~pipeline.templates.TemplateRegistry`) and the Python standard
library.  Node executors are referenced by their string ``type`` and
resolved lazily at execution time, exactly like the pipeline layer.

Submodules:

* :mod:`canvas.canvas` -- :class:`CanvasNode`, :class:`CanvasConnection`,
  :class:`CanvasState`, :class:`Canvas`.
* :mod:`canvas.versioning` -- :class:`CanvasVersion`,
  :class:`CanvasHistory`.
* :mod:`canvas.sharing` -- :class:`ShareLink`, :class:`ShareManager`.
* :mod:`canvas.autodirector` -- :class:`AutoDirector`.
* :mod:`canvas.registry` -- :class:`CommunityTemplate`,
  :class:`CommunityRegistry`.
"""

from __future__ import annotations

from .autodirector import AutoDirector
from .canvas import Canvas, CanvasConnection, CanvasNode, CanvasState
from .registry import CommunityRegistry, CommunityTemplate
from .sharing import ShareLink, ShareManager
from .versioning import CanvasHistory, CanvasVersion

__all__ = [
    # canvas core
    "CanvasNode",
    "CanvasConnection",
    "CanvasState",
    "Canvas",
    # versioning
    "CanvasVersion",
    "CanvasHistory",
    # sharing
    "ShareLink",
    "ShareManager",
    # autodirector
    "AutoDirector",
    # registry
    "CommunityTemplate",
    "CommunityRegistry",
]
