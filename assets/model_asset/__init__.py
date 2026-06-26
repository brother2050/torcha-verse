"""Concrete asset types for TorchaVerse (L2, v0.6.x).

In v0.4.x the five concrete asset subclasses
(:class:`ModelAsset`, :class:`CharacterAsset`, :class:`OutfitAsset`,
:class:`SceneAsset`, :class:`DepthAsset`) lived in a single
``assets/model_asset.py`` (473 lines).  In v0.6.x we split them
into one file per subclass so each asset type is independently
reviewable:

* :mod:`._model`      -- :class:`ModelAsset`
* :mod:`._character`  -- :class:`CharacterAsset`
* :mod:`._outfit`     -- :class:`OutfitAsset`
* :mod:`._scene`      -- :class:`SceneAsset`
* :mod:`._depth`      -- :class:`DepthAsset`

Backward compatibility
----------------------
The public surface is unchanged: ``from assets.model_asset import
ModelAsset`` (and the other four) still works because the
``__init__`` here re-exports the classes from their per-subclass
modules.

Defining any of these subclasses also has the side-effect of
registering it in the internal asset registry (via
:meth:`Asset.__init_subclass__`), which :meth:`Asset.from_dict`
relies on to dispatch on the serialised ``asset_type`` field.
"""

from __future__ import annotations

from ._character import CharacterAsset
from ._depth import DepthAsset
from ._model import ModelAsset
from ._outfit import OutfitAsset
from ._scene import SceneAsset

__all__ = [
    "ModelAsset",
    "CharacterAsset",
    "OutfitAsset",
    "SceneAsset",
    "DepthAsset",
]
