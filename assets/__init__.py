"""L2 Asset layer for the TorchaVerse framework.

This package unifies every versioned artefact in the framework -- model
weights, LoRA / ControlNet / IP-Adapter adapters, characters, outfits,
scenes, depth maps, subtitle tracks, prompt / pipeline / workflow
templates, voices and embeddings -- under a single :class:`Asset`
abstraction persisted by the tiered :class:`AssetStore`.

Layering (L1 -> L4):

* L1 ``infrastructure`` -- config, logging, devices, caching.
* L2 ``assets`` (this package) -- the asset model + store.
* L3 ``core`` -- model registry, schedulers, tokenizers.
* L4 ``engines`` -- text / image / audio / video / multimodal engines.

Assets are referenced across layers through immutable :class:`AssetRef`
handles, so that wiring between modules never silently drifts to a
different version.
"""

from __future__ import annotations

from .base import Asset, AssetRef, AssetRev
from .model_asset import (
    CharacterAsset,
    DepthAsset,
    ModelAsset,
    OutfitAsset,
    SceneAsset,
)
from .store import AssetStore, ColdStorageProtocol
from .types import AssetStatus, AssetType, LicenseRef

__all__ = [
    # types
    "AssetType",
    "AssetStatus",
    "LicenseRef",
    # base
    "Asset",
    "AssetRef",
    "AssetRev",
    # model_asset
    "ModelAsset",
    "CharacterAsset",
    "OutfitAsset",
    "SceneAsset",
    "DepthAsset",
    # store
    "AssetStore",
    "ColdStorageProtocol",
]
