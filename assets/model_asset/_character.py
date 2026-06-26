"""Character asset (``assets.model_asset.CharacterAsset``).

A :class:`CharacterAsset` bundles the reference imagery, an
optional five-view sheet, a pose bank, an IP-Adapter embedding
reference, a consistency seed and the outfits it may wear.  These
fields together let the consistency pipeline reproduce the same
identity across angles, garments and scenes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import Asset, AssetRef, AssetRev
from ..types import AssetStatus, AssetType, LicenseRef

__all__ = ["CharacterAsset"]


class CharacterAsset(Asset):
    """A character asset for cross-shot consistency.

    Args:
        id: Unique asset identifier.
        name: Human-readable display name.
        reference_images: Paths / refs to multi-angle reference images
            (typically >= 4).
        five_view_sheet: Path / ref to the auto-generated five-view sheet.
        pose_bank: Paths / refs to commonly used poses.
        embedding_ref: Reference to the IP-Adapter face/style embedding.
        description: Textual description used for prompt assembly.
        consistency_seed: Cross-shot seed root for temporal consistency.
        outfit_refs: References to outfits this character may wear.
        revisions: Initial revision history.
        status: Lifecycle status.
        license: License reference.
        tags: Free-form tags.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    asset_type = AssetType.CHARACTER

    def __init__(
        self,
        id: str,
        name: str,
        reference_images: Optional[List[str]] = None,
        five_view_sheet: str = "",
        pose_bank: Optional[List[str]] = None,
        embedding_ref: Optional[AssetRef] = None,
        description: str = "",
        consistency_seed: int = 0,
        outfit_refs: Optional[List[AssetRef]] = None,
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
        self.reference_images: List[str] = list(reference_images) if reference_images else []
        self.five_view_sheet: str = five_view_sheet
        self.pose_bank: List[str] = list(pose_bank) if pose_bank else []
        self.embedding_ref: Optional[AssetRef] = embedding_ref
        self.consistency_seed: int = int(consistency_seed)
        self.outfit_refs: List[AssetRef] = list(outfit_refs) if outfit_refs else []

    # ------------------------------------------------------------------
    def _extra_to_dict(self) -> Dict[str, Any]:
        return {
            "reference_images": list(self.reference_images),
            "five_view_sheet": self.five_view_sheet,
            "pose_bank": list(self.pose_bank),
            "embedding_ref": (
                self.embedding_ref.to_dict() if self.embedding_ref else None
            ),
            "consistency_seed": self.consistency_seed,
            "outfit_refs": [r.to_dict() for r in self.outfit_refs],
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "CharacterAsset":
        base = Asset._base_fields_from_dict(d)
        return cls(
            **base,
            reference_images=d.get("reference_images", []),
            five_view_sheet=d.get("five_view_sheet", ""),
            pose_bank=d.get("pose_bank", []),
            embedding_ref=(
                AssetRef.from_dict(d["embedding_ref"])
                if d.get("embedding_ref")
                else None
            ),
            consistency_seed=d.get("consistency_seed", 0),
            outfit_refs=[
                AssetRef.from_dict(r) for r in d.get("outfit_refs", [])
            ],
        )
