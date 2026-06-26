"""Model-weights asset (``assets.model_asset.ModelAsset``).

A :class:`ModelAsset` describes a base model -- an architecture
family, weight format, provenance and a config dictionary --
that the L4 nodes can resolve through the :class:`ModuleBus`.

Defining this subclass has the side effect of registering it in
the asset registry (via :meth:`Asset.__init_subclass__`), which
:meth:`Asset.from_dict` relies on to dispatch on the serialised
``asset_type`` field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import Asset, AssetRev
from ..types import AssetStatus, AssetType, LicenseRef

__all__ = ["ModelAsset"]


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
