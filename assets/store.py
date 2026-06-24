"""The AssetStore: a tiered, content-addressed asset repository (L2).

The :class:`AssetStore` is the single persistence surface for every
versioned artefact in TorchaVerse -- model weights, LoRA / ControlNet /
IP-Adapter adapters, characters, outfits, scenes, depth maps, subtitle
tracks, templates, voices and embeddings.  It implements the three-tier
storage described in the v0.3.0 architecture:

* **Hot layer** -- an in-process LRU cache (``OrderedDict`` guarded by a
  re-entrant lock) holding recently used :class:`Asset` objects for
  zero-copy access.
* **Warm layer** -- a SQLite database (WAL mode) indexing asset metadata
  plus a content-addressed object store on the local filesystem.  Content
  files are named by their sha256 digest and sharded two hex characters
  deep, so identical content is stored exactly once.
* **Cold layer** -- a :class:`ColdStorageProtocol` hook reserved for
  future S3 / OSS / MinIO backends.  The protocol is defined but not yet
  wired into the hot/warm read/write path.

The store is intentionally *not* a singleton: multiple independent stores
(e.g. one per user / workspace) can coexist.  All public operations are
thread-safe thanks to a single ``threading.RLock`` serialising SQLite and
cache access, with SQLite opened in WAL mode and ``check_same_thread`` off.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from uuid import uuid4

from infrastructure.logger import get_logger

from .base import Asset, AssetRef, AssetRev
from .types import AssetStatus, AssetType

__all__ = ["AssetStore", "ColdStorageProtocol"]


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT    PRIMARY KEY,
    asset_type     TEXT    NOT NULL,
    name           TEXT    NOT NULL,
    description    TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL,
    tags_json      TEXT    NOT NULL DEFAULT '[]',
    metadata_json  TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    updated_at     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_type   ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
CREATE INDEX IF NOT EXISTS idx_assets_name   ON assets(name);
"""

#: Read buffer size used when hashing / copying content files.
_CHUNK_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Cold storage protocol (reserved for future S3 / OSS backends)
# ---------------------------------------------------------------------------
@runtime_checkable
class ColdStorageProtocol(Protocol):
    """Protocol for cold-tier (S3 / OSS / MinIO) content backends.

    The cold tier is *reserved* in v0.3.0: the :class:`AssetStore` accepts
    an optional object conforming to this protocol but does not yet route
    reads or writes through it.  Future implementations (e.g. an S3-backed
    cold store) need only satisfy this interface to be plugged in.

    Implementations are expected to be content-addressed: every method is
    keyed by the sha256 ``content_hash`` of the stored blob.
    """

    def fetch(self, content_hash: str, dst: Path) -> Path:
        """Download the blob for ``content_hash`` to ``dst`` and return it."""
        ...

    def store(self, content_hash: str, src: Path) -> None:
        """Upload the blob at ``src`` under ``content_hash``."""
        ...

    def exists(self, content_hash: str) -> bool:
        """Return ``True`` if the blob is present in the cold tier."""
        ...

    def delete(self, content_hash: str) -> bool:
        """Remove the blob; return ``True`` if something was deleted."""
        ...


# ---------------------------------------------------------------------------
# AssetStore
# ---------------------------------------------------------------------------
class AssetStore:
    """A tiered, content-addressed, thread-safe asset repository.

    The store combines an in-process LRU cache (hot), a SQLite metadata
    index plus a content-addressed filesystem object store (warm), and a
    reserved cold-tier hook (S3 / OSS).  Assets are referenced across the
    framework through immutable :class:`AssetRef` handles that pin a
    specific revision and content hash.

    Args:
        base_dir: Root directory for the warm tier.  Defaults to
            ``~/.local/share/torcha-verse/assets/``.  An ``objects/``
            sub-directory holds the content-addressed blobs and a
            ``metadata.db`` file holds the SQLite index.
        hot_size: Maximum number of assets kept in the hot LRU cache.
        warm_db_path: Optional explicit path for the SQLite database file.
            When ``None`` it defaults to ``<base_dir>/metadata.db``.
        cold_storage: Optional cold-tier backend conforming to
            :class:`ColdStorageProtocol`.  Reserved for future use.

    Example:
        >>> store = AssetStore(base_dir="/tmp/tv_assets")
        >>> ref = store.put(model_asset, "/path/to/weights.safetensors")
        >>> asset, path = store.get(ref)
        >>> store.close()
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        hot_size: int = 256,
        warm_db_path: Optional[Union[str, Path]] = None,
        cold_storage: Optional[ColdStorageProtocol] = None,
    ) -> None:
        if hot_size <= 0:
            raise ValueError(f"hot_size must be > 0, got {hot_size}.")

        if base_dir is None:
            base_dir = Path.home() / ".local" / "share" / "torcha-verse" / "assets"
        self._base_dir: Path = Path(base_dir).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._objects_dir: Path = self._base_dir / "objects"
        self._objects_dir.mkdir(parents=True, exist_ok=True)

        if warm_db_path is None:
            warm_db_path = self._base_dir / "metadata.db"
        self._warm_db_path: Path = Path(warm_db_path).expanduser().resolve()
        self._warm_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._hot_size: int = int(hot_size)
        self._hot: "OrderedDict[str, Asset]" = OrderedDict()
        self._lock: threading.RLock = threading.RLock()
        self._cold_storage: Optional[ColdStorageProtocol] = cold_storage
        self._logger = get_logger(self.__class__.__name__)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        """Root directory of the warm tier."""
        return self._base_dir

    @property
    def objects_dir(self) -> Path:
        """Directory holding content-addressed blobs."""
        return self._objects_dir

    @property
    def warm_db_path(self) -> Path:
        """Path to the SQLite metadata database."""
        return self._warm_db_path

    @property
    def hot_size(self) -> int:
        """Maximum number of entries in the hot LRU cache."""
        return self._hot_size

    @property
    def cold_storage(self) -> Optional[ColdStorageProtocol]:
        """The configured cold-tier backend (may be ``None``)."""
        return self._cold_storage

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """Open the SQLite connection in WAL mode and create the schema."""
        self._conn = sqlite3.connect(
            str(self._warm_db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_SCHEMA)
        self._logger.debug(
            "AssetStore opened at %s (db=%s).", self._base_dir, self._warm_db_path
        )

    def close(self) -> None:
        """Close the SQLite connection and clear the hot cache.

        After calling this method the store must not be used again.
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                finally:
                    self._conn.close()
                self._conn = None
            self._hot.clear()
            self._logger.debug("AssetStore closed.")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "AssetStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AssetStore(base_dir={str(self._base_dir)!r}, "
            f"hot_size={self._hot_size})"
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def put(
        self,
        asset: Asset,
        content_path: Union[str, Path],
    ) -> AssetRef:
        """Store an asset together with its content bytes.

        The content file at ``content_path`` is hashed (sha256), copied
        into the content-addressed object store (deduplicated when the
        hash already exists) and a new :class:`AssetRev` is appended to
        the asset's revision history.  If a revision with the same content
        hash already exists it is reused instead of creating a duplicate.

        Args:
            asset: The asset to store / update.
            content_path: Path to the file holding the asset's content.

        Returns:
            An :class:`AssetRef` pinning the (possibly new) revision.

        Raises:
            FileNotFoundError: If ``content_path`` does not exist.
        """
        content_path = Path(content_path)
        if not content_path.exists():
            raise FileNotFoundError(f"Content file not found: {content_path}")

        # Hashing is read-only on the source file; do it outside the lock
        # so that a slow hash of a large model does not block other ops.
        content_hash = self._hash_file(content_path)
        size_bytes = content_path.stat().st_size

        with self._lock:
            self._store_content(content_path, content_hash)

            # Preserve revision history from a previously stored version.
            stored = self._load_asset(asset.id)
            if stored is not None and stored.revisions:
                asset.revisions = list(stored.revisions)

            # Deduplicate: reuse an existing revision for the same content.
            existing_rev = next(
                (r for r in asset.revisions if r.content_hash == content_hash),
                None,
            )
            if existing_rev is not None:
                rev = existing_rev
            else:
                rev = asset.add_revision(content_hash, size_bytes)

            asset.updated_at = time.time()
            self._save_asset(asset)
            self._hot_put(asset)

            self._logger.debug(
                "Put asset %s rev %s (hash=%s, %d bytes).",
                asset.id, rev.revision, content_hash[:12], size_bytes,
            )
            return AssetRef(
                asset_id=asset.id,
                asset_type=asset.asset_type,
                revision=rev.revision,
                content_hash=rev.content_hash,
            )

    def get(self, ref: AssetRef) -> "tuple[Asset, Path]":
        """Retrieve an asset and the path to its content file.

        Args:
            ref: The :class:`AssetRef` identifying the asset + revision.

        Returns:
            A ``(asset, content_path)`` tuple.

        Raises:
            KeyError: If the asset or the referenced revision is missing.
            FileNotFoundError: If the content blob is missing on disk.
        """
        with self._lock:
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                raise KeyError(f"Asset not found: {ref.asset_id!r}")
            rev = asset.get_revision(ref.revision)
            if rev is None or rev.content_hash != ref.content_hash:
                raise KeyError(
                    f"Revision {ref.revision!r} (hash={ref.content_hash[:12]}) "
                    f"not found for asset {ref.asset_id!r}."
                )
            content_path = self._content_path(rev.content_hash)
            if not content_path.exists():
                raise FileNotFoundError(
                    f"Content blob missing for {ref}: {content_path}"
                )
            return asset, content_path

    def exists(self, ref: AssetRef) -> bool:
        """Return ``True`` if the referenced asset + revision exist."""
        with self._lock:
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                return False
            rev = asset.get_revision(ref.revision)
            return rev is not None and rev.content_hash == ref.content_hash

    def list(
        self,
        asset_type: Optional[AssetType] = None,
        tags: Optional[List[str]] = None,
        status: Optional[AssetStatus] = None,
    ) -> List[Asset]:
        """List assets, optionally filtered by type / tags / status.

        Args:
            asset_type: Only include assets of this type.
            tags: Only include assets whose tags contain *all* of these.
            status: Only include assets with this lifecycle status.

        Returns:
            A list of matching :class:`Asset` instances.
        """
        query = "SELECT metadata_json FROM assets WHERE 1=1"
        params: List[Any] = []
        if asset_type is not None:
            query += " AND asset_type = ?"
            params.append(asset_type.value)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY updated_at DESC;"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()  # type: ignore[union-attr]

        results: List[Asset] = []
        for (meta_json,) in rows:
            asset = Asset.from_dict(json.loads(meta_json))
            if tags is not None and not all(t in asset.tags for t in tags):
                continue
            results.append(asset)
        return results

    def delete(self, ref: AssetRef) -> bool:
        """Soft-delete an asset by marking it :attr:`AssetStatus.ARCHIVED`.

        The asset and its content remain on disk so that historical
        references stay resolvable.

        Args:
            ref: The asset reference to soft-delete.

        Returns:
            ``True`` if the asset was found and archived, ``False`` if it
            was not found.
        """
        with self._lock:
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                return False
            asset.status = AssetStatus.ARCHIVED
            asset.updated_at = time.time()
            self._save_asset(asset)
            self._hot_put(asset)
            self._logger.debug("Archived asset %s.", ref.asset_id)
            return True

    def search(self, query: str) -> List[Asset]:
        """Fuzzy-search assets by name, description and tags.

        The match is case-insensitive and matches any asset whose name,
        description or any tag *contains* the query substring.

        Args:
            query: Substring to search for.

        Returns:
            A list of matching :class:`Asset` instances.
        """
        if not query:
            return []
        pattern = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(  # type: ignore[union-attr]
                "SELECT metadata_json FROM assets "
                "WHERE name LIKE ? OR description LIKE ? OR tags_json LIKE ? "
                "ORDER BY updated_at DESC;",
                (pattern, pattern, pattern),
            ).fetchall()

        needle = query.lower()
        results: List[Asset] = []
        for (meta_json,) in rows:
            asset = Asset.from_dict(json.loads(meta_json))
            if (
                needle in asset.name.lower()
                or needle in (asset.description or "").lower()
                or any(needle in t.lower() for t in asset.tags)
            ):
                results.append(asset)
        return results

    def fork(self, ref: AssetRef, new_name: str) -> AssetRef:
        """Copy an asset (at the referenced revision) into a new asset.

        The new asset gets a fresh id, the provided ``new_name``, an empty
        revision history and :attr:`AssetStatus.ACTIVE` status.  Because
        content storage is content-addressed, no bytes are duplicated --
        the forked asset simply references the same blob.

        Args:
            ref: The asset + revision to fork from.
            new_name: Display name for the forked asset.

        Returns:
            An :class:`AssetRef` to the newly created asset.
        """
        asset, content_path = self.get(ref)
        data = asset.to_dict()
        new_id = f"{asset.id}-fork-{uuid4().hex[:8]}"
        data["id"] = new_id
        data["name"] = new_name
        data["revisions"] = []
        data["status"] = AssetStatus.ACTIVE.value
        now = time.time()
        data["created_at"] = now
        data["updated_at"] = now
        new_asset = Asset.from_dict(data)
        self._logger.debug(
            "Forking asset %s -> %s.", ref.asset_id, new_id
        )
        return self.put(new_asset, content_path)

    def verify(self, ref: AssetRef) -> bool:
        """Verify that the on-disk content matches the recorded hash.

        Recomputes the sha256 of the stored content blob and compares it
        with the hash recorded in the referenced revision.

        Args:
            ref: The asset + revision to verify.

        Returns:
            ``True`` if the content is present and its hash matches.
        """
        with self._lock:
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                return False
            rev = asset.get_revision(ref.revision)
            if rev is None or rev.content_hash != ref.content_hash:
                return False
            content_path = self._content_path(rev.content_hash)
            if not content_path.exists():
                return False
            actual = self._hash_file(content_path)
            return actual == rev.content_hash

    # ------------------------------------------------------------------
    # Hot layer (LRU cache)
    # ------------------------------------------------------------------
    def _hot_get(self, asset_id: str) -> Optional[Asset]:
        """Return a cached asset, marking it most-recently-used."""
        # Caller holds self._lock.
        asset = self._hot.get(asset_id)
        if asset is not None:
            self._hot.move_to_end(asset_id)
        return asset

    def _hot_put(self, asset: Asset) -> None:
        """Insert / update an asset in the hot cache, evicting LRU entries."""
        # Caller holds self._lock.
        if asset.id in self._hot:
            self._hot.move_to_end(asset.id)
        self._hot[asset.id] = asset
        while len(self._hot) > self._hot_size:
            self._hot.popitem(last=False)

    def _load_asset(self, asset_id: str) -> Optional[Asset]:
        """Load an asset from the hot cache, falling back to SQLite."""
        # Caller holds self._lock.
        asset = self._hot_get(asset_id)
        if asset is not None:
            return asset
        row = self._conn.execute(  # type: ignore[union-attr]
            "SELECT metadata_json FROM assets WHERE asset_id = ?;",
            (asset_id,),
        ).fetchone()
        if row is None:
            return None
        asset = Asset.from_dict(json.loads(row[0]))
        self._hot_put(asset)
        return asset

    def _save_asset(self, asset: Asset) -> None:
        """Upsert an asset's metadata into SQLite."""
        # Caller holds self._lock.
        data = asset.to_dict()
        self._conn.execute(  # type: ignore[union-attr]
            "INSERT OR REPLACE INTO assets "
            "(asset_id, asset_type, name, description, status, "
            " tags_json, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (
                asset.id,
                asset.asset_type.value,
                asset.name,
                asset.description,
                asset.status.value,
                json.dumps(asset.tags),
                json.dumps(data),
                asset.created_at,
                asset.updated_at,
            ),
        )

    # ------------------------------------------------------------------
    # Warm layer (content-addressed object store)
    # ------------------------------------------------------------------
    def _content_path(self, content_hash: str) -> Path:
        """Return the on-disk path for a content hash (sharded 2 deep)."""
        return self._objects_dir / content_hash[:2] / content_hash

    def _store_content(self, src: Path, content_hash: str) -> None:
        """Copy ``src`` into the object store under ``content_hash``.

        Deduplicated: if the target blob already exists nothing is done.
        """
        # Caller holds self._lock.
        dst = self._content_path(content_hash)
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fdst.write(chunk)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return the sha256 hex digest of the file at ``path``."""
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                h.update(chunk)
        return h.hexdigest()
