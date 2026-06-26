"""Outfit asset (``assets.model_asset.OutfitAsset``).

An :class:`OutfitAsset` is applied on top of a character to
change garments / style.  It references a style embedding
(IP-Adapter Style) and a LoRA that carries the garment weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import Asset, AssetRef, AssetRev
from ..types import AssetStatus, AssetType, LicenseRef

__all__ = ["OutfitAsset"]


class OutfitAsset(Asset):
    """An outfit asset (style + LoRA bundle).

    Args:
        id: Unique asset identifier.
        name: Human-readable display name.
        style_embedding_ref: Reference to the style embedding asset.
        lora_ref: Reference to the LoRA asset providing garment weights.
        description: Textual description used for prompt assembly.
        revisions: Initial revision history.
        status: Lifecycle status.
        license: License reference.
        tags: Free-form tags.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    asset_type = AssetType.OUTFIT

    def __init__(
        self,
        id: str,
        name: str,
        style_embedding_ref: Optional[AssetRef] = None,
        lora_ref: Optional[AssetRef] = None,
        description: str = "",
        revisions: Optional[List[AssetRev]] = None,
        status: AssetStatus = AssetStatus.DRAFT,
        license: Optional[LicenseRef] = None,
        tags: Optional[List[str]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            description=description,
            revisions=revisions,
            status=status,
            license=license,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )
        self.style_embedding_ref: Optional[AssetRef] = style_embedding_ref
        self.lora_ref: Optional[AssetRef] = lora_ref

    # ------------------------------------------------------------------
    def _extra_to_dict(self) -> Dict[str, Any]:
        return {
            "style_embedding_ref": (
                self.style_embedding_ref.to_dict()
                if self.style_embedding_ref
                else None
            ),
            "lora_ref": (
                self.lora_ref.to_dict() if self.lora_ref else None
            ),
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "OutfitAsset":
        base = Asset._base_fields_from_dict(d)
        return cls(
            **base,
            style_embedding_ref=(
                AssetRef.from_dict(d["style_embedding_ref"])
                if d.get("style_embedding_ref")
                else None
            ),
            lora_ref=(
                AssetRef.from_dict(d["lora_ref"])
                if d.get("lora_ref")
                else None
            ),
        )
