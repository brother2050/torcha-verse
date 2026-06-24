"""Concrete asset types for TorchaVerse (L2).

This module defines the asset subclasses that describe the "things" the
framework generates and reuses: model weights, characters, outfits,
scenes and depth maps.  Each subclass extends :class:`assets.base.Asset`,
pins its :attr:`asset_type`, and adds domain-specific fields together
with the matching serialisation hooks (``_extra_to_dict`` / ``_from_dict``)
so that assets round-trip cleanly through the :class:`assets.store.AssetStore`.

Defining these subclasses also has the side-effect of registering them in
the internal asset registry (via :meth:`Asset.__init_subclass__`), which
:meth:`Asset.from_dict` relies on to dispatch on the serialised
``asset_type`` field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Asset, AssetRef, AssetRev
from .types import AssetStatus, AssetType, LicenseRef

__all__ = [
    "ModelAsset",
    "CharacterAsset",
    "OutfitAsset",
    "SceneAsset",
    "DepthAsset",
]


# ---------------------------------------------------------------------------
# ModelAsset
# ---------------------------------------------------------------------------
class ModelAsset(Asset):
    """A model-weights asset (base model, not an adapter).

    Args:
        id: Unique asset identifier.
        name: Human-readable display name.
        architecture: Architecture family, e.g. ``"decoder_only"``,
            ``"dit"``, ``"unet"``.
        format: Weight container format -- one of ``"safetensors"``,
            ``"pt"``, ``"gguf"``, ``"onnx"``.
        size_gb: Approximate size in gigabytes (for resource planning).
        source: Provenance string, e.g. ``"local"``, ``"huggingface"``.
        config: Arbitrary model configuration dictionary.
        description: Free-form description.
        revisions: Initial revision history.
        status: Lifecycle status.
        license: License reference.
        tags: Free-form tags.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    asset_type = AssetType.MODEL

    def __init__(
        self,
        id: str,
        name: str,
        architecture: str,
        format: str = "safetensors",
        size_gb: float = 0.0,
        source: str = "",
        config: Optional[Dict[str, Any]] = None,
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
        self.architecture: str = architecture
        self.format: str = format
        self.size_gb: float = float(size_gb)
        self.source: str = source
        self.config: Dict[str, Any] = dict(config) if config else {}

    # ------------------------------------------------------------------
    def _extra_to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "format": self.format,
            "size_gb": self.size_gb,
            "source": self.source,
            "config": dict(self.config),
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "ModelAsset":
        base = Asset._base_fields_from_dict(d)
        return cls(
            **base,
            architecture=d.get("architecture", ""),
            format=d.get("format", "safetensors"),
            size_gb=d.get("size_gb", 0.0),
            source=d.get("source", ""),
            config=d.get("config", {}),
        )


# ---------------------------------------------------------------------------
# CharacterAsset
# ---------------------------------------------------------------------------
class CharacterAsset(Asset):
    """A character asset for cross-shot consistency.

    A character bundles the reference imagery, an (optional) auto-generated
    five-view sheet, a pose bank, an IP-Adapter embedding reference, a
    consistency seed and the outfits it may wear.  These fields together
    let the consistency pipeline reproduce the same identity across
    angles, garments and scenes.

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


# ---------------------------------------------------------------------------
# OutfitAsset
# ---------------------------------------------------------------------------
class OutfitAsset(Asset):
    """An outfit asset (style + LoRA bundle).

    An outfit is applied on top of a character to change garments / style.
    It references a style embedding (IP-Adapter Style) and a LoRA that
    carries the garment weights.

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


# ---------------------------------------------------------------------------
# SceneAsset
# ---------------------------------------------------------------------------
class SceneAsset(Asset):
    """A scene asset (LoRA + ControlNet + depth bundle).

    A scene fixes the background / environment of a shot.  It references a
    scene LoRA, a ControlNet (e.g. Tile) and a depth map that together
    constrain the diffusion process.

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


# ---------------------------------------------------------------------------
# DepthAsset
# ---------------------------------------------------------------------------
class DepthAsset(Asset):
    """A depth-map asset.

    Depth maps are used as ControlNet conditioning.  They record the
    source image they were derived from, the path to the depth map itself
    and the estimation method that produced them.

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
