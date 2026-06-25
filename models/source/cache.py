"""Local cache for the TorchaVerse model fetcher (v0.4.0).

When a model is fetched from an external source the weights and
metadata are stored under a per-user cache directory -- by default
``~/.cache/torcha-verse/<source>/<repo_id>/<revision>/`` -- so that
subsequent fetches can short-circuit the network.  This module owns
that directory layout, the on-disk manifest, the sha256 integrity
check, and the (de)serialisation of the :class:`CachedModel` record.

The cache is intentionally simple:

* one manifest file per model (``manifest.json``) holding the
  license id, the sha256 of every downloaded file, the fetch
  timestamp, and the source URL;
* one file per weight / tokenizer / config blob, written with
  ``O_CREAT | O_EXCL`` so a partial download is never mistaken for
  a complete one;
* a single re-entrant lock per cache directory (we use a process-
  level ``threading.RLock`` because the cache is process-local and
  does not need cross-process coordination).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- on-disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from infrastructure.logger import get_logger

__all__ = [
    "CacheLocation",
    "CachedFile",
    "CachedModel",
    "default_cache_root",
    "compute_content_fingerprint",
    "ModelCache",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Subdirectory of the user's home (or platform equivalent) where
#: cached models are stored.
_CACHE_DIRNAME = "torcha-verse"

#: Name of the manifest file written next to a cached model.
_MANIFEST_FILENAME = "manifest.json"

#: Read buffer size when hashing / copying file contents.
_CHUNK_SIZE = 1 << 20  # 1 MiB

#: Module-level logger.
_logger = get_logger("models.source.cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def default_cache_root() -> Path:
    """Return the platform-appropriate cache root for TorchaVerse.

    Uses ``$TORCHA_VERSE_CACHE`` if set (CI / sandboxed environments),
    otherwise falls back to ``~/.cache/torcha-verse`` on Linux and
    macOS or ``%LOCALAPPDATA%\\torcha-verse`` on Windows.
    """
    override = os.environ.get("TORCHA_VERSE_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base).expanduser().resolve() / _CACHE_DIRNAME


def compute_content_fingerprint(
    files: Sequence[Dict[str, Any]],
) -> str:
    """Return a stable content fingerprint for a list of file specs.

    The fingerprint is the sorted ``(name, sha256)`` joined by
    ``|`` and re-hashed (sha256).  Two manifests with the same
    name/sha256 pair set always produce the same fingerprint,
    *regardless* of the order in which the files were listed.
    This makes the fingerprint safe to use for cross-mirror
    deduplication -- if the same file set is available on two
    mirrors, they hash to the same value.

    Args:
        files: Iterable of dicts, each with at least ``name`` and
            ``sha256`` keys.  ``sha256`` is treated as empty string
            when missing (the caller is expected to compute it
            upstream if a strong fingerprint is needed).

    Returns:
        A 64-character hex-encoded SHA-256 digest.
    """
    pairs = []
    for f in files:
        name = str(f.get("name", ""))
        sha = str(f.get("sha256", ""))
        pairs.append((name, sha))
    pairs.sort()
    joined = "|".join("{}={}".format(n, s) for n, s in pairs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class CacheLocation:
    """The on-disk location of a cached model.

    Attributes:
        root: Cache root (``~/.cache/torcha-verse`` by default).
        source: The upstream source (``"huggingface"``, ``"civitai"``,
            ``"local"``).
        repo_id: The repository / model id (e.g. ``"Qwen/Qwen2.5-0.5B"``).
        revision: The source revision (``"main"``, ``"v1.0"``, ...).
            Empty string means "no revision".
    """

    root: Path
    source: str
    repo_id: str
    revision: str = ""

    def path(self) -> Path:
        """Return the on-disk directory for this cache entry."""
        if self.revision:
            return self.root / self.source / self.repo_id / self.revision
        return self.root / self.source / self.repo_id

    def manifest_path(self) -> Path:
        """Return the manifest file path."""
        return self.path() / _MANIFEST_FILENAME

    def as_dict(self) -> Dict[str, str]:
        """Return a JSON-friendly dictionary view."""
        return {
            "root": str(self.root),
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "path": str(self.path()),
        }


@dataclass
class CachedFile:
    """A single file in a cached model.

    Attributes:
        name: Filename (e.g. ``"model.safetensors"``).
        size: Size in bytes.
        sha256: Hex-encoded SHA-256 digest.
    """

    name: str
    size: int
    sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CachedModel:
    """A cached model record (mirrors the on-disk manifest).

    Attributes:
        source: Upstream source id.
        repo_id: The repository / model id.
        revision: Source revision.
        license_id: SPDX-style license id (validated at fetch time).
        url: The upstream URL the model was fetched from.
        fetched_at: Unix timestamp of the fetch.
        files: Per-file records.
    """

    source: str
    repo_id: str
    revision: str
    license_id: str
    url: str
    fetched_at: float
    files: List[CachedFile] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "url": self.url,
            "fetched_at": self.fetched_at,
            "files": [f.as_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CachedModel":
        """Reconstruct a :class:`CachedModel` from a serialised dict."""
        files_raw = d.get("files", [])
        files = [
            CachedFile(
                name=str(f.get("name", "")),
                size=int(f.get("size", 0)),
                sha256=str(f.get("sha256", "")),
            )
            for f in files_raw
        ]
        return cls(
            source=str(d.get("source", "")),
            repo_id=str(d.get("repo_id", "")),
            revision=str(d.get("revision", "")),
            license_id=str(d.get("license_id", "")),
            url=str(d.get("url", "")),
            fetched_at=float(d.get("fetched_at", 0.0)),
            files=files,
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @property
    def content_fingerprint(self) -> str:
        """Return a stable content fingerprint for this manifest.

        See :func:`compute_content_fingerprint` for the exact
        recipe.  Two manifests with the same ``(name, sha256)`` set
        always have the same fingerprint regardless of the order
        in which files were listed; this is what makes it safe to
        use for cross-mirror deduplication.
        """
        return compute_content_fingerprint(
            [{"name": f.name, "sha256": f.sha256} for f in self.files]
        )


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------
class ModelCache:
    """A cache for fetched model files.

    The cache is bound to a single root directory (default:
    :func:`default_cache_root`) and exposes:

    * :meth:`location_for` -- the :class:`CacheLocation` for a given
      source / repo / revision, without touching the disk.
    * :meth:`has` / :meth:`load_manifest` -- does the model exist in
      cache, and if so what is its manifest?
    * :meth:`write_files` -- atomically write a list of files into
      the cache and persist the manifest.
    * :meth:`verify` -- re-hash every cached file and confirm the
      digests match the manifest.
    * :meth:`clear` -- delete the on-disk directory for one model.
    """

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        self._root: Path = (
            Path(root).expanduser().resolve() if root is not None
            else default_cache_root()
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    @property
    def root(self) -> Path:
        """The cache root directory."""
        return self._root

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def location_for(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
    ) -> CacheLocation:
        """Return the on-disk location for a model (no disk access)."""
        if not source or not source.strip():
            raise ValueError("`source` must be non-empty")
        if not repo_id or not repo_id.strip():
            raise ValueError("`repo_id` must be non-empty")
        return CacheLocation(
            root=self._root,
            source=source.strip(),
            repo_id=repo_id.strip(),
            revision=revision.strip(),
        )

    def has(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
    ) -> bool:
        """Return ``True`` if a manifest exists for the given model."""
        loc = self.location_for(source, repo_id, revision)
        return loc.manifest_path().is_file()

    def find_by_fingerprint(
        self,
        source: str,
        fingerprint: str,
    ) -> Optional[CacheLocation]:
        """Locate a cached model by its content fingerprint.

        Walks every ``<root>/<source>/**/manifest.json`` entry and
        returns the first one whose manifest's
        :attr:`CachedModel.content_fingerprint` matches ``fingerprint``.
        Returns ``None`` when no match is found.

        The path is searched recursively (``rglob``) rather than
        fixed-depth because HF-style ``repo_id`` values may
        contain ``/`` (e.g. ``"Qwen/Qwen2.5"``) which Path treats
        as a directory separator; the on-disk layout ends up
        several levels deep but every manifest file is still at
        a known suffix.

        Use this for **cross-mirror / cross-revision deduplication**:
        if you already cached the same blob from a different mirror
        (or under a different revision tag), this is how you find it
        without re-downloading.

        Note: the search is O(N) over every cached manifest.  Callers
        that hit the cache heavily should add a memo layer on top.
        """
        if not fingerprint:
            return None
        base = self._root / source
        if not base.is_dir():
            return None
        with self._lock:
            for manifest_path in base.rglob(_MANIFEST_FILENAME):
                if not manifest_path.is_file():
                    continue
                try:
                    with manifest_path.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    m = CachedModel.from_dict(data)
                except (OSError, ValueError):
                    continue
                if m.content_fingerprint == fingerprint:
                    # Reconstruct the CacheLocation from the
                    # manifest path: <root>/<source>/<repo_id>/<revision>/manifest.json
                    rel = manifest_path.relative_to(self._root)
                    parts = rel.parts
                    # parts[0] is the source; the rest splits into
                    # repo_id (all-but-last) and revision (last-1).
                    if len(parts) < 3:
                        continue
                    revision = parts[-2]
                    repo_id = "/".join(parts[1:-2])
                    return CacheLocation(
                        root=self._root,
                        source=parts[0],
                        repo_id=repo_id,
                        revision=revision,
                    )
        return None

    def load_manifest(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
    ) -> CachedModel:
        """Load the manifest for a cached model.

        Raises:
            FileNotFoundError: If the manifest does not exist.
            ValueError: If the manifest JSON is malformed.
        """
        loc = self.location_for(source, repo_id, revision)
        with loc.manifest_path().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return CachedModel.from_dict(data)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_files(
        self,
        source: str,
        repo_id: str,
        revision: str,
        license_id: str,
        url: str,
        files: Sequence[Dict[str, Any]],
    ) -> CachedModel:
        """Atomically write a list of files into the cache.

        Each ``files[i]`` is a dict with keys ``name``, ``data``
        (bytes), and optional ``sha256`` (hex string; computed if
        absent).  The manifest is written *after* every file, so a
        partial write leaves no manifest behind and the cache lookup
        will miss -- the next fetch will retry from scratch.

        Args:
            source: Upstream source id.
            repo_id: The repository / model id.
            revision: Source revision.
            license_id: SPDX license id (already verified).
            url: The upstream URL the model was fetched from.
            files: Sequence of file dicts (see above).

        Returns:
            The :class:`CachedModel` manifest that was written.
        """
        if not files:
            raise ValueError("`files` must be non-empty")
        loc = self.location_for(source, repo_id, revision)
        target_dir = loc.path()
        with self._lock:
            target_dir.mkdir(parents=True, exist_ok=True)
            cached_files: List[CachedFile] = []
            for spec in files:
                name = str(spec.get("name", ""))
                data = spec.get("data")
                if not name or data is None:
                    raise ValueError(
                        "each file spec must have 'name' and 'data'"
                    )
                if isinstance(data, str):
                    data = data.encode("utf-8")
                if not isinstance(data, (bytes, bytearray)):
                    raise TypeError(
                        "`data` must be bytes (got {})".format(type(data))
                    )
                digest = str(spec.get("sha256", "")) or hashlib.sha256(
                    bytes(data)
                ).hexdigest()
                out_path = target_dir / name
                self._write_bytes(out_path, bytes(data))
                actual = hashlib.sha256(out_path.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(
                        "sha256 mismatch writing {}: expected {}, got {}".format(
                            out_path, digest, actual,
                        )
                    )
                cached_files.append(
                    CachedFile(
                        name=name,
                        size=out_path.stat().st_size,
                        sha256=digest,
                    )
                )
            manifest = CachedModel(
                source=source,
                repo_id=repo_id,
                revision=revision,
                license_id=license_id,
                url=url,
                fetched_at=time.time(),
                files=cached_files,
            )
            # Write manifest last so a partial fetch leaves no trace.
            self._write_bytes(
                loc.manifest_path(),
                manifest.to_json().encode("utf-8"),
            )
            self._logger.info(
                "Cached %d file(s) for %s/%s@%s at %s",
                len(cached_files), source, repo_id, revision, target_dir,
            )
        return manifest

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        """Atomically write ``data`` to ``path``."""
        # Write to a temp file in the same directory, fsync, rename.
        # This guarantees the destination is either the old content
        # or the new content -- never a partial file.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Verify / clear
    # ------------------------------------------------------------------
    def verify(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
    ) -> bool:
        """Re-hash every cached file and confirm the manifest digests.

        Returns ``True`` only when every recorded file exists and has
        the expected sha256.  Returns ``False`` (without raising) on
        any mismatch so callers can re-fetch transparently.
        """
        try:
            manifest = self.load_manifest(source, repo_id, revision)
        except (FileNotFoundError, ValueError):
            return False
        loc = self.location_for(source, repo_id, revision)
        for f in manifest.files:
            p = loc.path() / f.name
            if not p.is_file():
                return False
            if p.stat().st_size != f.size:
                return False
            actual = self._hash_file(p)
            if actual != f.sha256:
                return False
        return True

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def clear(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
    ) -> bool:
        """Delete the on-disk directory for one model.

        Returns ``True`` when something was deleted, ``False`` when
        the directory did not exist.
        """
        loc = self.location_for(source, repo_id, revision)
        target = loc.path()
        with self._lock:
            if not target.exists():
                return False
            # Walk and unlink -- avoids requiring shutil for portability.
            for entry in sorted(target.rglob("*"), reverse=True):
                if entry.is_file() or entry.is_symlink():
                    entry.unlink(missing_ok=True)
                elif entry.is_dir():
                    try:
                        entry.rmdir()
                    except OSError:
                        pass
            try:
                target.rmdir()
            except OSError:
                pass
            return True

    def __repr__(self) -> str:
        return "ModelCache(root={!r})".format(str(self._root))
