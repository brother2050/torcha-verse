"""The :class:`AssetStore` core (v0.6.x).

This module hosts the public :class:`AssetStore` class -- a
tiered, content-addressed, thread-safe asset repository.  The
class is intentionally compact: the heavy lifting is delegated
to focused helpers in this sub-package:

* :mod:`assets.store._schema` -- SQLite schema + chunk size
* :mod:`assets.store._db`     -- DB lifecycle (open, close, ensure_open)
* :mod:`assets.store._warm`   -- content-addressed object store (path, copy, hash)
* :mod:`assets.store._hot`    -- LRU hot cache
* :mod:`assets.store._cold`   -- cold-tier routing (push, promote, evict)
* :mod:`assets.store._sql`    -- list / search query builders
* :mod:`assets.store._protocol` -- :class:`ColdStorageProtocol`

Each public method follows the same shape: a short lock-acquire
section, the actual work, and a release.  Use-after-close is
guarded by :func:`assets.store._db.ensure_open` (S2-7).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import uuid4

from assets.base import Asset, AssetRef
from assets.types import AssetStatus, AssetType
from infrastructure.logger import get_logger

from ._cold import evict_to_cold, promote_from_cold, push_to_cold
from ._db import ensure_open, open_connection
from ._hot import HotCache, load_asset_from_row
from ._protocol import ColdStorageProtocol
from ._sql import list_query, search_query
from ._warm import cleanup_staging, content_path, copy_to_staging, hash_file

__all__ = ["AssetStore"]


class AssetStore:
    """A tiered, content-addressed, thread-safe asset repository.

    The store combines an in-process LRU cache (hot), a SQLite
    metadata index plus a content-addressed filesystem object
    store (warm), and a reserved cold-tier hook (S3 / OSS).  Assets
    are referenced across the framework through immutable
    :class:`AssetRef` handles that pin a specific revision and
    content hash.

    Args:
        base_dir: Root directory for the warm tier.  Defaults to
            ``~/.local/share/torcha-verse/assets/``.  An
            ``objects/`` sub-directory holds the content-addressed
            blobs and a ``metadata.db`` file holds the SQLite index.
        hot_size: Maximum number of assets kept in the hot LRU cache.
        warm_db_path: Optional explicit path for the SQLite database
            file.  When ``None`` it defaults to
            ``<base_dir>/metadata.db``.
        cold_storage: Optional cold-tier backend conforming to
            :class:`ColdStorageProtocol`.  When provided, every
            successful :meth:`put` is mirrored to the cold tier
            and :meth:`get` will transparently fall back to the
            cold tier (and re-promote the blob) when the warm
            content file is missing.
        mirror_to_cold: When ``True`` (default) writes are mirrored
            to the cold tier; set to ``False`` for warm-only
            deployments.

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
        mirror_to_cold: bool = True,
    ) -> None:
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

        self._hot: HotCache = HotCache(hot_size)
        self._lock: threading.RLock = threading.RLock()
        self._cold_storage: Optional[ColdStorageProtocol] = cold_storage
        self._mirror_to_cold: bool = bool(mirror_to_cold)
        self._logger = get_logger(self.__class__.__name__)
        self._closed: bool = False

        self._conn: Optional[sqlite3.Connection] = None
        self._conn = open_connection(self._warm_db_path)
        self._logger.debug(
            "AssetStore opened at %s (db=%s).", self._base_dir, self._warm_db_path
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def objects_dir(self) -> Path:
        return self._objects_dir

    @property
    def warm_db_path(self) -> Path:
        return self._warm_db_path

    @property
    def hot_size(self) -> int:
        return self._hot.capacity

    @property
    def cold_storage(self) -> Optional[ColdStorageProtocol]:
        return self._cold_storage

    @property
    def mirror_to_cold(self) -> bool:
        return self._mirror_to_cold

    def set_mirror_to_cold(self, enabled: bool) -> None:
        """Enable or disable cold-tier mirroring at runtime."""
        with self._lock:
            self._mirror_to_cold = bool(enabled)

    # ------------------------------------------------------------------
    # Cold-tier helpers (delegate to assets.store._cold)
    # ------------------------------------------------------------------
    def _push_to_cold(self, content_hash: str, content_path_arg: Path) -> None:
        push_to_cold(
            self._cold_storage, self._mirror_to_cold,
            content_hash, content_path_arg, self._logger,
        )

    def promote_from_cold(self, content_hash: str) -> Optional[Path]:
        """Download a blob from the cold tier and stage it in the warm tier.

        On success returns the warm-tier path; on failure (cold
        tier unconfigured, blob missing, network error) returns
        ``None`` and logs a warning.
        """
        return promote_from_cold(
            self._cold_storage, self._objects_dir, content_hash, self._logger,
        )

    def evict_to_cold(self, content_hash: str) -> bool:
        """Remove a blob from the warm tier; the cold tier keeps a copy."""
        return evict_to_cold(
            self._cold_storage, self._objects_dir, content_hash,
        )

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        ensure_open(self._closed)

    def close(self) -> None:
        """Close the SQLite connection and clear the hot cache."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
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
            f"hot_size={self._hot.capacity})"
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def put(
        self,
        asset: Asset,
        content_path_arg: Union[str, Path],
    ) -> AssetRef:
        """Store an asset together with its content bytes.

        The content file at ``content_path_arg`` is hashed
        (sha256), copied into the content-addressed object store
        (deduplicated when the hash already exists) and a new
        :class:`AssetRev` is appended to the asset's revision
        history.  If a revision with the same content hash
        already exists it is reused instead of creating a
        duplicate.
        """
        self._ensure_open()
        content_path_arg = Path(content_path_arg)
        if not content_path_arg.exists():
            raise FileNotFoundError(f"Content file not found: {content_path_arg}")

        size_bytes = content_path_arg.stat().st_size

        # Phase 1 (lock held): check status, generate staging path.
        with self._lock:
            self._ensure_open()
            stored = self._load_asset(asset.id)
            existing_revisions: list = (
                list(stored.revisions)
                if stored is not None and stored.revisions
                else []
            )
            staging_dir = self._objects_dir / ".staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            staging_path = staging_dir / f".{uuid4().hex}.tmp"

        # Phase 2 (lock free): perform file copy and hash.
        try:
            copy_to_staging(content_path_arg, staging_path)
            content_hash = hash_file(staging_path)
            size_match = any(
                r.size_bytes == size_bytes for r in existing_revisions
            )
        except Exception:
            cleanup_staging(staging_path)
            raise

        # Phase 3 (lock held): atomic rename + metadata update.
        with self._lock:
            self._ensure_open()
            try:
                final_path = self._content_path(content_hash)
                if not final_path.exists():
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging_path, final_path)
                else:
                    staging_path.unlink(missing_ok=True)

                # Explicit transaction (S2-6).
                self._conn.execute("BEGIN")  # type: ignore[union-attr]
                committed = False
                try:
                    stored = self._load_asset(asset.id)
                    if stored is not None and stored.revisions:
                        asset.revisions = list(stored.revisions)

                    existing_rev = None
                    if size_match:
                        existing_rev = next(
                            (r for r in asset.revisions
                             if r.content_hash == content_hash),
                            None,
                        )
                    if existing_rev is not None:
                        rev = existing_rev
                    else:
                        rev = asset.add_revision(content_hash, size_bytes)

                    asset.updated_at = time.time()
                    self._save_asset(asset)
                    self._conn.execute("COMMIT")  # type: ignore[union-attr]
                    committed = True
                finally:
                    if not committed:
                        self._conn.execute("ROLLBACK")  # type: ignore[union-attr]

                self._hot.put(asset)

                # Push to cold tier (best-effort).
                self._push_to_cold(rev.content_hash, final_path)

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
            except Exception:
                cleanup_staging(staging_path)
                raise

    def get(self, ref: AssetRef) -> "tuple[Asset, Path]":
        """Retrieve an asset and the path to its content file."""
        with self._lock:
            self._ensure_open()
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                raise KeyError(f"Asset not found: {ref.asset_id!r}")
            rev = asset.get_revision(ref.revision)
            if rev is None or rev.content_hash != ref.content_hash:
                raise KeyError(
                    f"Revision {ref.revision!r} (hash={ref.content_hash[:12]}) "
                    f"not found for asset {ref.asset_id!r}."
                )
            content_path_arg = self._content_path(rev.content_hash)
            if not content_path_arg.exists():
                # Warm tier is missing the blob; try the cold tier.
                promoted = self.promote_from_cold(rev.content_hash)
                if promoted is None:
                    raise FileNotFoundError(
                        f"Content blob missing for {ref}: {content_path_arg}"
                    )
                content_path_arg = promoted
            return asset, content_path_arg

    def exists(self, ref: AssetRef) -> bool:
        """Return ``True`` if the referenced asset + revision exist."""
        with self._lock:
            self._ensure_open()
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
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Asset]:
        """List assets, optionally filtered by type / tags / status."""
        sql, params = list_query(
            asset_type.value if asset_type is not None else None,
            status.value if status is not None else None,
            tags, limit, offset,
        )
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(sql, params).fetchall()  # type: ignore[union-attr]
        results: List[Asset] = []
        for (meta_json,) in rows:
            asset = Asset.from_dict(json.loads(meta_json))
            results.append(asset)
        return results

    def delete(self, ref: AssetRef) -> bool:
        """Soft-delete an asset by marking it :attr:`AssetStatus.ARCHIVED`."""
        with self._lock:
            self._ensure_open()
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                return False
            asset.status = AssetStatus.ARCHIVED
            asset.updated_at = time.time()
            self._save_asset(asset)
            self._hot.put(asset)
            self._logger.debug("Archived asset %s.", ref.asset_id)
            return True

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Asset]:
        """Fuzzy-search assets by name, description and tags."""
        if not query:
            return []
        sql, params = search_query(query, limit, offset)
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(  # type: ignore[union-attr]
                sql, params,
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
        """Copy an asset (at the referenced revision) into a new asset."""
        with self._lock:
            self._ensure_open()
        asset, content_path_arg = self.get(ref)
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
        return self.put(new_asset, content_path_arg)

    def verify(self, ref: AssetRef) -> bool:
        """Verify that the on-disk content matches the recorded hash."""
        with self._lock:
            self._ensure_open()
            asset = self._load_asset(ref.asset_id)
            if asset is None:
                return False
            rev = asset.get_revision(ref.revision)
            if rev is None or rev.content_hash != ref.content_hash:
                return False
            content_path_arg = self._content_path(rev.content_hash)
            if not content_path_arg.exists():
                return False
            actual = hash_file(content_path_arg)
            return actual == rev.content_hash

    # ------------------------------------------------------------------
    # Hot layer (LRU cache) + SQLite helpers
    # ------------------------------------------------------------------
    def _load_asset(self, asset_id: str) -> Optional[Asset]:
        # Caller holds self._lock.
        asset = self._hot.get(asset_id)
        if asset is not None:
            return asset
        row = self._conn.execute(  # type: ignore[union-attr]
            "SELECT metadata_json FROM assets WHERE asset_id = ?;",
            (asset_id,),
        ).fetchone()
        return load_asset_from_row(row, self._hot)

    def _save_asset(self, asset: Asset) -> None:
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
        return content_path(self._objects_dir, content_hash)
