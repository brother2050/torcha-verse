"""Asset base classes and cross-module references for TorchaVerse (L2).

This module defines the core abstractions of the asset layer:

* :class:`AssetRev` -- an immutable record describing a single revision of an
  asset (revision id, content hash, timestamp, size, metadata).
* :class:`AssetRef` -- a *frozen* (hashable) reference that points at a
  specific revision of a specific asset by id + content hash.  ``AssetRef``
  is the only currency used to wire assets together across modules; because
  it is immutable, a reference can never silently drift to a different
  version.
* :class:`Asset` -- the abstract base class for every concrete asset type.
  It owns the revision history, lifecycle status, license, tags and
  timestamps, and provides serialisation (``to_dict`` / ``from_dict``) plus
  equality / hashing based on ``id`` + current revision.

Concrete subclasses (model, character, outfit, scene, depth, ...) live in
:mod:`assets.model_asset` and register themselves with the internal
``_ASSET_REGISTRY`` so that :meth:`Asset.from_dict` can dispatch on the
serialised ``asset_type`` field.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .types import AssetStatus, AssetType, LicenseRef

__all__ = ["AssetRev", "AssetRef", "Asset"]


# ---------------------------------------------------------------------------
# Default license
# ---------------------------------------------------------------------------
def _default_license() -> LicenseRef:
    """Return a fresh "no assertion" license used when none is provided."""
    return LicenseRef(
        spdx_id="NOASSERTION",
        name="No Assertion",
        url="",
        commercial_use=False,
    )


# ---------------------------------------------------------------------------
# AssetRev
# ---------------------------------------------------------------------------
@dataclass
class AssetRev:
    """A single revision of an asset.

    Revisions are append-only: a new :class:`AssetRev` is created by
    :meth:`Asset.add_revision` every time new content is stored for an
    asset.  The :attr:`content_hash` (a sha256 hex digest) is the key into
    the content-addressed object store, so two revisions pointing at the
    same content share a single on-disk file.

    Attributes:
        revision: Human-readable revision identifier, e.g. ``"r1"``.
        content_hash: sha256 hex digest of the revision's content bytes.
        created_at: POSIX timestamp at which the revision was created.
        size_bytes: Size of the revision's content in bytes.
        metadata: Free-form metadata dictionary for this revision.
    """

    revision: str
    content_hash: str
    created_at: float
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this revision to a JSON-serialisable dictionary."""
        return {
            "revision": self.revision,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssetRev":
        """Reconstruct an :class:`AssetRev` from a serialised dictionary."""
        return cls(
            revision=d["revision"],
            content_hash=d["content_hash"],
            created_at=float(d["created_at"]),
            size_bytes=int(d["size_bytes"]),
            metadata=dict(d.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# AssetRef
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetRef:
    """Immutable, hashable reference to a specific asset revision.

    ``AssetRef`` is the canonical handle passed between modules, persisted
    in pipelines / workflows, and embedded inside other assets (e.g. a
    :class:`~assets.model_asset.CharacterAsset` references its embedding via
    an ``AssetRef``).  Because the dataclass is frozen, a reference is
    guaranteed to always point at the same ``revision`` + ``content_hash``;
    upgrading to a newer revision requires creating a *new* ``AssetRef``.

    Attributes:
        asset_id: The referenced asset's unique id.
        asset_type: The :class:`AssetType` of the referenced asset.
        revision: The specific revision identifier (e.g. ``"r3"``).
        content_hash: sha256 hex digest of the referenced content.
    """

    asset_id: str
    asset_type: AssetType
    revision: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this reference to a JSON-serialisable dictionary."""
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "revision": self.revision,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssetRef":
        """Reconstruct an :class:`AssetRef` from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new (frozen) :class:`AssetRef` instance.
        """
        return cls(
            asset_id=d["asset_id"],
            asset_type=AssetType(d["asset_type"]),
            revision=d["revision"],
            content_hash=d["content_hash"],
        )


# ---------------------------------------------------------------------------
# Asset registry (populated by subclasses via __init_subclass__)
# ---------------------------------------------------------------------------
_ASSET_REGISTRY: Dict[AssetType, Type["Asset"]] = {}


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------
class Asset(abc.ABC):
    """Abstract base class for every TorchaVerse asset.

    An :class:`Asset` is a versioned, licensed, taggable object identified
    by a unique ``id``.  It owns an append-only list of :class:`AssetRev`
    entries; the most recent one is exposed through the
    :attr:`current_revision` property.

    Concrete subclasses set the ``asset_type`` class attribute (which
    satisfies the abstract :attr:`asset_type` property) and add their own
    domain-specific fields.  They are expected to override
    :meth:`_extra_to_dict` and :meth:`_from_dict` so that the full asset --
    base fields plus subclass extras -- round-trips through
    :meth:`to_dict` / :meth:`from_dict`.

    Args:
        id: Unique asset identifier (e.g. ``"sakura"``).
        name: Human-readable display name.
        description: Free-form description (used by search).
        revisions: Initial revision history; defaults to empty.
        status: Lifecycle status; defaults to :attr:`AssetStatus.DRAFT`.
        license: License reference; defaults to a "NOASSERTION" license.
        tags: List of free-form tags; defaults to empty.
        created_at: Creation POSIX timestamp; defaults to now.
        updated_at: Last-update POSIX timestamp; defaults to ``created_at``.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str = "",
        revisions: Optional[List[AssetRev]] = None,
        status: AssetStatus = AssetStatus.DRAFT,
        license: Optional[LicenseRef] = None,
        tags: Optional[List[str]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
    ) -> None:
        if not id or not isinstance(id, str):
            raise ValueError("Asset 'id' must be a non-empty string.")
        if not name or not isinstance(name, str):
            raise ValueError("Asset 'name' must be a non-empty string.")

        self.id: str = id
        self.name: str = name
        self.description: str = description
        self.revisions: List[AssetRev] = list(revisions) if revisions else []
        self.status: AssetStatus = status
        self.license: LicenseRef = license if license is not None else _default_license()
        self.tags: List[str] = list(tags) if tags else []

        now: float = time.time()
        self.created_at: float = float(created_at) if created_at is not None else now
        self.updated_at: float = (
            float(updated_at) if updated_at is not None else self.created_at
        )

    # ------------------------------------------------------------------
    # Subclass registration
    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        at = getattr(cls, "asset_type", None)
        if isinstance(at, AssetType):
            _ASSET_REGISTRY[at] = cls

    # ------------------------------------------------------------------
    # Abstract / overridable
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def asset_type(self) -> AssetType:
        """The :class:`AssetType` of this asset.

        Subclasses satisfy this by assigning a class attribute, e.g.
        ``asset_type = AssetType.MODEL``.
        """
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_revision(self) -> Optional[AssetRev]:
        """The most recent :class:`AssetRev`, or ``None`` if there are none."""
        return self.revisions[-1] if self.revisions else None

    # ------------------------------------------------------------------
    # Revision management
    # ------------------------------------------------------------------
    def add_revision(
        self,
        content_hash: str,
        size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetRev:
        """Append a new revision to this asset.

        Revision identifiers are monotonically increasing strings of the
        form ``"r1"``, ``"r2"``, ... based on the current revision count.

        Args:
            content_hash: sha256 hex digest of the new content.
            size_bytes: Size of the new content in bytes.
            metadata: Optional metadata dictionary for the revision.

        Returns:
            The newly created :class:`AssetRev`.
        """
        revision = f"r{len(self.revisions) + 1}"
        rev = AssetRev(
            revision=revision,
            content_hash=content_hash,
            created_at=time.time(),
            size_bytes=int(size_bytes),
            metadata=dict(metadata) if metadata else {},
        )
        self.revisions.append(rev)
        self.updated_at = rev.created_at
        return rev

    def get_revision(self, rev: str) -> Optional[AssetRev]:
        """Return the :class:`AssetRev` whose identifier equals ``rev``.

        Args:
            rev: Revision identifier (e.g. ``"r2"``).

        Returns:
            The matching revision, or ``None`` if not found.
        """
        for candidate in self.revisions:
            if candidate.revision == rev:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise this asset to a JSON-serialisable dictionary.

        Includes all base fields plus whatever subclass extras are
        returned by :meth:`_extra_to_dict`.
        """
        d: Dict[str, Any] = {
            "id": self.id,
            "asset_type": self.asset_type.value,
            "name": self.name,
            "description": self.description,
            "revisions": [r.to_dict() for r in self.revisions],
            "status": self.status.value,
            "license": self.license.to_dict(),
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        d.update(self._extra_to_dict())
        return d

    def _extra_to_dict(self) -> Dict[str, Any]:
        """Return subclass-specific fields for serialisation.

        Subclasses override this to add their own fields to the serialised
        dictionary.  The base implementation returns an empty dict.
        """
        return {}

    @classmethod
    def _base_fields_from_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the common base fields from a serialised dictionary.

        Helper used by subclass ``_from_dict`` implementations.
        """
        return {
            "id": d["id"],
            "name": d["name"],
            "description": d.get("description", ""),
            "revisions": [AssetRev.from_dict(r) for r in d.get("revisions", [])],
            "status": AssetStatus(d.get("status", AssetStatus.DRAFT.value)),
            "license": (
                LicenseRef.from_dict(d["license"]) if d.get("license") else None
            ),
            "tags": list(d.get("tags", [])),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Asset":
        """Reconstruct an asset (of the correct subclass) from a dict.

        Dispatches on the ``asset_type`` field to the subclass registered
        for that type.  Known subclasses are imported lazily on first use
        so that the registry is populated even when this module is imported
        in isolation.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new asset instance of the appropriate concrete subclass.

        Raises:
            ValueError: If the asset type is unknown / unsupported.
        """
        asset_type = AssetType(d["asset_type"])
        subcls = _ASSET_REGISTRY.get(asset_type)
        if subcls is None:
            # Lazy-load the bundled subclasses to populate the registry.
            from . import model_asset as _model_asset  # noqa: F401
            subcls = _ASSET_REGISTRY.get(asset_type)
        if subcls is None:
            raise ValueError(
                f"Unsupported asset type '{asset_type.value}': no subclass "
                f"is registered for it."
            )
        return subcls._from_dict(d)

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Asset):
            return NotImplemented
        if self.id != other.id:
            return False
        sr = self.current_revision
        orr = other.current_revision
        if sr is None or orr is None:
            return sr is None and orr is None
        return sr.revision == orr.revision and sr.content_hash == orr.content_hash

    def __hash__(self) -> int:
        rev = self.current_revision
        revision = rev.revision if rev else ""
        content_hash = rev.content_hash if rev else ""
        return hash((self.id, revision, content_hash))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        rev = self.current_revision.revision if self.current_revision else "-"
        return (
            f"{self.__class__.__name__}(id={self.id!r}, "
            f"type={self.asset_type.value!r}, revision={rev!r}, "
            f"status={self.status.value!r})"
        )
