"""L5 Pipeline layer for the TorchaVerse framework.

This package is the *orchestration* layer of the v0.3.0 architecture.  It
sits above the L4 node system and the L3 core abstractions, turning a
structural graph of nodes into an executable, parallelisable pipeline.

Layering (L1 -> L5):

* L1 ``infrastructure`` -- config, logging, devices, caching.
* L2 ``assets`` -- the asset model + store.
* L3 ``core`` -- module bus, model registry, schedulers, tokenizers.
* L4 ``nodes`` -- the node system (created in parallel).
* L5 ``pipeline`` (this package) -- DAG, composer, templates, prompt studio.

The pipeline layer is deliberately *torch-free*: it never imports
:mod:`torch` directly.  Node executors are resolved lazily at runtime through
a :class:`~pipeline.composer.NodeContext`, so the orchestration logic can be
imported and exercised in any environment.

Submodules:

* :mod:`pipeline.dag` -- :class:`DAG`, :class:`DAGNode`, :class:`DAGEdge`.
* :mod:`pipeline.composer` -- :class:`Pipeline`, :class:`PipelineConfig`,
  :class:`PipelineBuilder`, :class:`NodeContext`.
* :mod:`pipeline.templates` -- :class:`PipelineTemplate`,
  :class:`TemplateRegistry`, :data:`BUILTIN_TEMPLATES`.
* :mod:`pipeline.prompt_studio` -- :class:`PromptTemplate`,
  :class:`PromptEnhancer`, :class:`SeedManager`, :class:`StylePreset`.
"""

from __future__ import annotations

from .composer import (
    NodeContext,
    Pipeline,
    PipelineBuilder,
    PipelineConfig,
)
from .dag import DAG, DAGEdge, DAGNode
from .prompt_studio import (
    BUILTIN_STYLE_PRESETS,
    PromptEnhancer,
    PromptTemplate,
    SeedManager,
    StylePreset,
)
from .templates import BUILTIN_TEMPLATES, PipelineTemplate, TemplateRegistry

__all__ = [
    # dag
    "DAG",
    "DAGNode",
    "DAGEdge",
    # composer
    "NodeContext",
    "Pipeline",
    "PipelineConfig",
    "PipelineBuilder",
    # templates
    "PipelineTemplate",
    "TemplateRegistry",
    "BUILTIN_TEMPLATES",
    # prompt_studio
    "PromptTemplate",
    "PromptEnhancer",
    "SeedManager",
    "StylePreset",
    "BUILTIN_STYLE_PRESETS",
]
