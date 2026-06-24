"""Asset type definitions for the TorchaVerse Asset layer (L2).

This module defines the enumerations and lightweight value objects that
describe *what* an asset is and *how* it may be used.  Everything that is
persisted by :class:`assets.store.AssetStore` -- model weights, LoRA
adapters, ControlNets, IP-Adapters, characters, outfits, scenes, depth
maps, subtitle tracks, prompt / pipeline / workflow templates, voices and
embeddings -- is classified by :class:`AssetType` and tracked through its
lifecycle by :class:`AssetStatus`.

A :class:`LicenseRef` carries the SPDX-style licensing metadata that the
v0.3.0 architecture mandates on every :class:`assets.base.AssetRef`, so
that commercial-use gating and redistribution audits can be enforced at
load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

__all__ = ["AssetType", "AssetStatus", "LicenseRef"]


class AssetType(Enum):
    """Enumeration of every asset category known to the framework.

    The value of each member is a short, stable string identifier that is
    used as the SQLite index key and in serialised asset references, so it
    must never change once released.
    """

    MODEL = "model"
    LORA = "lora"
    CONTROLNET = "controlnet"
    IP_ADAPTER = "ip_adapter"
    CHARACTER = "character"
    OUTFIT = "outfit"
    SCENE = "scene"
    DEPTH = "depth"
    SUBTITLE_TRACK = "subtitle_track"
    PROMPT_TEMPLATE = "prompt_template"
    PIPELINE_TEMPLATE = "pipeline_template"
    WORKFLOW_TEMPLATE = "workflow_template"
    VOICE = "voice"
    EMBEDDING = "embedding"


class AssetStatus(Enum):
    """Lifecycle status of an asset.

    Assets are never physically removed from the store; deletion is a soft
    transition to :attr:`ARCHIVED` (or :attr:`DELETED` for a hard logical
    removal) so that historical references remain resolvable.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class LicenseRef:
    """Reference to a software / content license (SPDX-style).

    Every asset carries one :class:`LicenseRef`.  When a user configures a
    commercial use-case the framework consults :attr:`commercial_use` to
    decide whether the asset may be loaded, and redistribution exports
    attach the :attr:`spdx_id` / :attr:`name` / :attr:`url` to the
    produced artefact.

    Attributes:
        spdx_id: The SPDX license identifier (e.g. ``"Apache-2.0"``,
            ``"MIT"``).  Use ``"NOASSERTION"`` when the license is unknown.
        name: Human-readable license name.
        url: Canonical URL of the license text (may be empty).
        commercial_use: Whether the license permits commercial use.
    """

    spdx_id: str
    name: str
    url: str
    commercial_use: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this license reference to a plain dictionary.

        Returns:
            A JSON-serialisable dictionary with the four license fields.
        """
        return {
            "spdx_id": self.spdx_id,
            "name": self.name,
            "url": self.url,
            "commercial_use": self.commercial_use,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LicenseRef":
        """Reconstruct a :class:`LicenseRef` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`LicenseRef` instance.
        """
        return cls(
            spdx_id=d["spdx_id"],
            name=d["name"],
            url=d["url"],
            commercial_use=bool(d["commercial_use"]),
        )
