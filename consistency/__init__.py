"""Consistency framework for the TorchaVerse framework (v0.3.0).

This package implements the v0.3.0 "consistency framework" -- the set
of engines, profiles, scoring primitives and a top-level pipeline that
together ensure character / outfit / scene / depth identity is
preserved across shots and over time (video).

The consistency framework sits at L6 of the v0.3.0 architecture and
composes the L2 asset layer (characters, outfits, scenes, depth maps)
with the L4 node system (placeholder consistency nodes) into a
coherent generation surface.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- config, logging, devices.
* L2 ``assets`` -- the asset model + :class:`~assets.store.AssetStore`.
* L3 ``core`` -- module bus, model registry, schedulers.
* L4 ``nodes`` -- the node system (consistency-conditioning nodes).
* L5 ``pipeline`` / ``canvas`` -- DAG, composer, visual canvas.
* L6 ``consistency`` (this package) -- engines, pipeline, scoring.

Submodules:

* :mod:`consistency.profile` -- :class:`ConsistencyProfile`,
  :class:`ConsistencyManager`.
* :mod:`consistency.score` -- :class:`ConsistencyScore`,
  :class:`ScoreCalculator`.
* :mod:`consistency.character` -- :class:`CharacterEngine`.
* :mod:`consistency.outfit` -- :class:`OutfitEngine`.
* :mod:`consistency.scene` -- :class:`SceneEngine`.
* :mod:`consistency.pipeline` -- :class:`ConsistencyPipeline`.

Example::

    from consistency import (
        ConsistencyProfile,
        ConsistencyManager,
        ConsistencyPipeline,
        ConsistencyScore,
        ScoreCalculator,
        CharacterEngine,
        OutfitEngine,
        SceneEngine,
    )

    mgr = ConsistencyManager()
    profile = mgr.create_profile("default", character_weight=0.85)
    pipeline = ConsistencyPipeline(profile=profile)
    result = pipeline.generate("a portrait of the character")
    print(result["consistency_scores"])
"""

from __future__ import annotations

from .character import CharacterEngine
from .outfit import OutfitEngine
from .pipeline import ConsistencyPipeline
from .profile import ConsistencyManager, ConsistencyProfile
from .scene import SceneEngine
from .score import ConsistencyScore, ScoreCalculator

__all__ = [
    # profile
    "ConsistencyProfile",
    "ConsistencyManager",
    # score
    "ConsistencyScore",
    "ScoreCalculator",
    # engines
    "CharacterEngine",
    "OutfitEngine",
    "SceneEngine",
    # pipeline
    "ConsistencyPipeline",
]
