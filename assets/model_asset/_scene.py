"""Scene asset (``assets.model_asset.SceneAsset``).

A :class:`SceneAsset` fixes the background / environment of a
shot.  It references a scene LoRA, a ControlNet (e.g. Tile) and
a depth map that together constrain the diffusion process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import Asset, AssetRef, AssetRev
from ..types import AssetStatus, AssetType, LicenseRef

__all__ = ["SceneAsset"]


class SceneAsset(Asset):
    """A scene asset (LoRA + ControlNet + depth bundle).

    Args:
        id: Unique asset identifier.
        name: Human-readable display name.
        lora_ref: Reference to the scene LoRA asset.
        controlnet_ref: Reference to the ControlNet asset.
        description: Textual description used for prompt assembly.
        depth_ref: Reference to the depth-map asset for the scene.
        revisions: Initial revision history.
        status: Lifecycle status.
        license: License reference.
        tags: Free-form tags.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    asset_type = AssetType.SCENE

    def __init__(
        self,
        id: str,
        name: str,
        lora_ref: Optional[AssetRef] = None,
        controlnet_ref: Optional[AssetRef] = None,
        description: str = "",
        depth_ref: Optional[AssetRef] = None,
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
        self.lora_ref: Optional[AssetRef] = lora_ref
        self.controlnet_ref: Optional[AssetRef] = controlnet_ref
        self.depth_ref: Optional[AssetRef] = depth_ref

    # ------------------------------------------------------------------
    def _extra_to_dict(self) -> Dict[str, Any]:
        return {
            "lora_ref": (
                self.lora_ref.to_dict() if self.lora_ref else None
            ),
            "controlnet_ref": (
                self.controlnet_ref.to_dict()
                if self.controlnet_ref
                else None
            ),
            "depth_ref": (
                self.depth_ref.to_dict() if self.depth_ref else None
            ),
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "SceneAsset":
        base = Asset._base_fields_from_dict(d)
        return cls(
            **base,
            lora_ref=(
                AssetRef.from_dict(d["lora_ref"])
                if d.get("lora_ref")
                else None
            ),
            controlnet_ref=(
                AssetRef.from_dict(d["controlnet_ref"])
                if d.get("controlnet_ref")
                else None
            ),
            depth_ref=(
                AssetRef.from_dict(d["depth_ref"])
                if d.get("depth_ref")
                else None
            ),
        )
