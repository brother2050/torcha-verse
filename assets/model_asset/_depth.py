"""Depth-map asset (``assets.model_asset.DepthAsset``).

Depth maps are used as ControlNet conditioning.  They record
the source image they were derived from, the path to the depth
map itself and the estimation method that produced them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import Asset, AssetRev
from ..types import AssetStatus, AssetType, LicenseRef

__all__ = ["DepthAsset"]


class DepthAsset(Asset):
    """A depth-map asset.

    Args:
        id: Unique asset identifier.
        name: Human-readable display name.
        source_image: Path / ref to the image the depth map was derived
            from.
        depth_map_path: Path / ref to the produced depth map.
        method: Estimation method -- ``"midas"`` or ``"depth_anything"``.
        description: Free-form description.
        revisions: Initial revision history.
        status: Lifecycle status.
        license: License reference.
        tags: Free-form tags.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    asset_type = AssetType.DEPTH

    def __init__(
        self,
        id: str,
        name: str,
        source_image: str = "",
        depth_map_path: str = "",
        method: str = "midas",
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
        self.source_image: str = source_image
        self.depth_map_path: str = depth_map_path
        self.method: str = method

    # ------------------------------------------------------------------
    def _extra_to_dict(self) -> Dict[str, Any]:
        return {
            "source_image": self.source_image,
            "depth_map_path": self.depth_map_path,
            "method": self.method,
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "DepthAsset":
        base = Asset._base_fields_from_dict(d)
        return cls(
            **base,
            source_image=d.get("source_image", ""),
            depth_map_path=d.get("depth_map_path", ""),
            method=d.get("method", "midas"),
        )
