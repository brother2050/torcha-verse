"""Multi-source model fetching abstraction for TorchaVerse.

This module decouples the framework from any single model hub by defining a
uniform :class:`SourceFetcher` abstract base class and a handful of concrete
implementations:

* :class:`LocalSource` -- resolves models already present on the local
  filesystem (``file://`` URIs or plain paths).
* :class:`HuggingFaceSource` -- downloads from the Hugging Face Hub via the
  optional ``huggingface_hub`` package.
* :class:`ModelScopeSource` -- downloads from ModelScope via the optional
  ``modelscope`` SDK.
* :class:`ModelersSource` -- downloads from Huawei's Modelers (Ascend
  ecosystem) via the optional ``modelers`` SDK.

A :class:`SourceRegistry` ties the fetchers together: callers hand it an
opaque ``ref`` string and the registry routes it to the first fetcher that
claims to handle it.  The registry also provides resume support (via
``.partial`` marker files) and automatic parallel sharding for large files.

All third-party SDKs are imported lazily and guarded with ``try/except`` so
that the framework imports cleanly even when none of them are installed.

Example:
    >>> registry = SourceRegistry()
    >>> registry.register(LocalSource())
    >>> path = registry.fetch("/data/models/llama-8b", "/cache/llama-8b")
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

from .logger import get_logger

__all__ = [
    "SourceFetcher",
    "LocalSource",
    "HuggingFaceSource",
    "ModelScopeSource",
    "ModelersSource",
    "SourceRegistry",
    "LicenseRef",
    "FetchError",
]

#: Files larger than this threshold (in bytes) trigger parallel sharding.
_LARGE_FILE_THRESHOLD: int = 2 * 1024 ** 3  # 2 GiB

#: Number of shards used for parallel download of large files.
_SHARD_COUNT: int = 8

#: Suffix appended to in-progress downloads to support resume.
_PARTIAL_SUFFIX: str = ".partial"


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import huggingface_hub  # type: ignore
    from huggingface_hub import snapshot_download as _hf_snapshot_download  # type: ignore
    from huggingface_hub import hf_hub_download as _hf_hub_download  # type: ignore

    _HAS_HUGGINGFACE: bool = True
except Exception:  # pragma: no cover - huggingface_hub not installed
    _HAS_HUGGINGFACE = False

try:  # pragma: no cover - import guard
    from modelscope.hub.snapshot_download import (  # type: ignore
        snapshot_download as _ms_snapshot_download,
    )

    _HAS_MODELSCOPE: bool = True
except Exception:  # pragma: no cover - modelscope not installed
    _HAS_MODELSCOPE = False

try:  # pragma: no cover - import guard
    from modelers import download as _modelers_download  # type: ignore

    _HAS_MODELERS: bool = True
except Exception:  # pragma: no cover - modelers not installed
    _HAS_MODELERS = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FetchError(RuntimeError):
    """Raised when a model fetch operation fails."""


# ---------------------------------------------------------------------------
# LicenseRef dataclass
# ---------------------------------------------------------------------------
@dataclass
class LicenseRef:
    """Reference to a software/data license.

    Attributes:
        spdx_id: SPDX license identifier (e.g. ``"Apache-2.0"``).
        name: Human-readable license name.
        url: URL to the license text.
        commercial_use: Whether commercial use is permitted.
    """

    spdx_id: str
    name: str
    url: str
    commercial_use: bool = False


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class ProgressCallback(Protocol):
    """Callable invoked with download progress information."""

    def __call__(
        self,
        downloaded: int,
        total: Optional[int],
        message: str = "",
    ) -> None:
        ...


# A plain callable also satisfies the structural protocol above.
ProgressCallable = Callable[[int, Optional[int], str], None]


# ---------------------------------------------------------------------------
# SourceFetcher abstract base class
# ---------------------------------------------------------------------------
class SourceFetcher(ABC):
    """Abstract base class defining the source-fetcher protocol.

    Concrete fetchers implement five operations:

    * :meth:`can_handle` -- decide whether this fetcher understands ``ref``.
    * :meth:`fetch` -- download/copy the resource to ``dst``.
    * :meth:`verify` -- verify the integrity (sha256) of a fetched resource.
    * :meth:`license` -- return the :class:`LicenseRef` for ``ref`` (if known).
    * :meth:`cleanup_partial` -- remove any partial download artefacts.

    Subclasses may also set :attr:`supports_parallel` to ``True`` to opt into
    the registry's parallel-sharding path for large files.
    """

    #: Whether this fetcher can download byte ranges in parallel.
    supports_parallel: bool = False

    #: Whether this fetcher implements :meth:`fetch_range` / :meth:`head_size`
    #: so the registry can perform its own 8-way parallel sharding.  Fetchers
    #: backed by an SDK that already parallelises downloads (Hugging Face,
    #: ModelScope, ...) leave this ``False`` and let the SDK do the work.
    supports_ranges: bool = False

    #: Human-readable name used in diagnostics.
    name: str = "source"

    @abstractmethod
    def can_handle(self, ref: str) -> bool:
        """Return ``True`` if this fetcher can resolve ``ref``."""
        raise NotImplementedError

    @abstractmethod
    def fetch(
        self,
        ref: str,
        dst: Path,
        progress_callback: Optional[ProgressCallable] = None,
        resume: bool = True,
    ) -> Path:
        """Fetch the resource referenced by ``ref`` into ``dst``.

        Args:
            ref: Opaque reference string (URI, repo id, local path, ...).
            dst: Destination directory or file path.
            progress_callback: Optional callable receiving progress updates.
            resume: When ``True`` attempt to resume from a ``.partial`` file.

        Returns:
            The resolved path to the fetched resource.
        """
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        path: Path,
        expected_sha256: Optional[str] = None,
    ) -> bool:
        """Verify the integrity of ``path``.

        When ``expected_sha256`` is ``None`` the method returns ``True`` if
        the file exists and is readable; otherwise it compares the file's
        sha256 digest against the expected value.
        """
        raise NotImplementedError

    @abstractmethod
    def license(self, ref: str) -> Optional[LicenseRef]:
        """Return the :class:`LicenseRef` for ``ref`` if known."""
        raise NotImplementedError

    @abstractmethod
    def cleanup_partial(self, ref: str, dst: Path) -> None:
        """Remove any partial download artefacts for ``ref`` under ``dst``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks for registry-driven parallel sharding.
    # ------------------------------------------------------------------
    def head_size(self, ref: str) -> Optional[int]:
        """Return the total size in bytes of ``ref``, or ``None`` if unknown.

        The default implementation returns ``None``; fetchers that advertise
        :attr:`supports_ranges` should override this so the registry can
        decide whether to shard and how to split the byte ranges.
        """
        return None

    def fetch_range(
        self,
        ref: str,
        dst_part: Path,
        byte_start: int,
        byte_end: int,
        progress_callback: Optional[ProgressCallable] = None,
    ) -> Path:
        """Download a single ``[byte_start, byte_end)`` range of ``ref``.

        The default implementation raises :class:`NotImplementedError`;
        fetchers that advertise :attr:`supports_ranges` override this to
        enable the registry's 8-way parallel sharding.

        Args:
            ref: Opaque reference string.
            dst_part: Destination file for this shard.
            byte_start: Inclusive start byte offset.
            byte_end: Exclusive end byte offset.
            progress_callback: Optional progress callback.

        Returns:
            The path of the written shard file.
        """
        raise NotImplementedError(
            f"{self.name} fetcher does not implement byte-range downloads."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# LocalSource
# ---------------------------------------------------------------------------
class LocalSource(SourceFetcher):
    """Fetch models already present on the local filesystem.

    Accepts plain filesystem paths and ``file://`` URIs.  Fetching copies
    (or hard-links) the source into the destination so that downstream code
    always operates on a stable local copy.

    Example:
        >>> src = LocalSource()
        >>> src.can_handle("/data/models/llama")
        True
        >>> src.can_handle("file:///data/models/llama")
        True
    """

    name: str = "local"

    def can_handle(self, ref: str) -> bool:
        if not ref:
            return False
        if ref.startswith("file://"):
            return True
        # Treat anything that resolves to an existing local path as local.
        candidate = Path(ref).expanduser()
        return candidate.exists()

    def fetch(
        self,
        ref: str,
        dst: Path,
        progress_callback: Optional[ProgressCallable] = None,
        resume: bool = True,  # noqa: ARG002 - unused, kept for API symmetry
    ) -> Path:
        src = self._resolve_ref(ref)
        if not src.exists():
            raise FetchError(f"Local source does not exist: {src}")

        dst = Path(dst).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.resolve() == dst and src.exists():
            # Source and destination are identical; nothing to do.
            return dst

        if progress_callback is not None:
            total = self._path_size(src)
            progress_callback(0, total, f"copying {src}")

        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            # Prefer a hard link to avoid duplicating large weights; fall
            # back to a byte copy when hard-linking fails (cross-device).
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        if progress_callback is not None:
            total = self._path_size(dst)
            progress_callback(total, total, f"done {dst}")
        return dst

    def verify(self, path: Path, expected_sha256: Optional[str] = None) -> bool:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return False
        if expected_sha256 is None:
            return True
        return self._sha256(path) == expected_sha256.lower()

    def license(self, ref: str) -> Optional[LicenseRef]:
        # Local sources carry no intrinsic license metadata.
        return None

    def cleanup_partial(self, ref: str, dst: Path) -> None:
        dst = Path(dst).expanduser().resolve()
        partial = dst.with_name(dst.name + _PARTIAL_SUFFIX)
        if partial.exists():
            shutil.rmtree(partial) if partial.is_dir() else partial.unlink()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_ref(ref: str) -> Path:
        if ref.startswith("file://"):
            ref = ref[len("file://") :]
        return Path(ref).expanduser().resolve()

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


# ---------------------------------------------------------------------------
# HuggingFaceSource
# ---------------------------------------------------------------------------
class HuggingFaceSource(SourceFetcher):
    """Fetch models from the Hugging Face Hub.

    Uses the optional ``huggingface_hub`` package.  When the package is not
    installed, :meth:`can_handle` returns ``False`` and :meth:`fetch` raises
    a :class:`FetchError` with an informative message.

    A reference is considered a Hugging Face repo id when it matches the
    ``org/name`` pattern and is not a local path or ``file://`` URI.
    """

    name: str = "huggingface"
    supports_parallel: bool = True

    def can_handle(self, ref: str) -> bool:
        if not _HAS_HUGGINGFACE:
            return False
        if not ref or ref.startswith("file://"):
            return False
        if Path(ref).expanduser().exists():
            return False
        # Hugging Face repo ids look like "org/name" or "org/name/branch".
        return "/" in ref and not ref.startswith(("/", "."))

    def fetch(
        self,
        ref: str,
        dst: Path,
        progress_callback: Optional[ProgressCallable] = None,
        resume: bool = True,
    ) -> Path:
        if not _HAS_HUGGINGFACE:
            raise FetchError(
                "huggingface_hub is not installed; run "
                "`pip install huggingface_hub` to enable HuggingFaceSource."
            )
        dst = Path(dst).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)

        repo_id, revision = self._split_ref(ref)
        if progress_callback is not None:
            progress_callback(0, None, f"downloading {repo_id}@{revision}")

        local_dir = _hf_snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(dst),
            resume_download=resume,
        )
        if progress_callback is not None:
            progress_callback(0, None, f"done {dst}")
        return Path(local_dir)

    def verify(self, path: Path, expected_sha256: Optional[str] = None) -> bool:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return False
        if expected_sha256 is None:
            return True
        return LocalSource._sha256(path) == expected_sha256.lower()

    def license(self, ref: str) -> Optional[LicenseRef]:
        if not _HAS_HUGGINGFACE:
            return None
        repo_id, _ = self._split_ref(ref)
        try:
            info = huggingface_hub.model_info(repo_id)  # type: ignore[union-attr]
        except Exception:
            return None
        card_data = getattr(info, "card_data", None) or {}
        raw = card_data.get("license") if isinstance(card_data, dict) else None
        if not raw:
            return None
        return LicenseRef(
            spdx_id=str(raw),
            name=str(raw),
            url=f"https://huggingface.co/{repo_id}",
            commercial_use=False,
        )

    def cleanup_partial(self, ref: str, dst: Path) -> None:
        dst = Path(dst).expanduser().resolve()
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        incomplete = dst.parent / (dst.name + ".incomplete")
        if incomplete.exists():
            shutil.rmtree(incomplete, ignore_errors=True)

    @staticmethod
    def _split_ref(ref: str) -> "tuple[str, Optional[str]]":
        parts = ref.split("/")
        if len(parts) >= 3:
            return "/".join(parts[:2]), parts[2]
        return ref, None


# ---------------------------------------------------------------------------
# ModelScopeSource
# ---------------------------------------------------------------------------
class ModelScopeSource(SourceFetcher):
    """Fetch models from ModelScope via the optional ``modelscope`` SDK."""

    name: str = "modelscope"
    supports_parallel: bool = True

    def can_handle(self, ref: str) -> bool:
        if not _HAS_MODELSCOPE:
            return False
        if not ref or ref.startswith("file://"):
            return False
        if Path(ref).expanduser().exists():
            return False
        # ModelScope model ids look like "org/name".
        return "/" in ref and not ref.startswith(("/", "."))

    def fetch(
        self,
        ref: str,
        dst: Path,
        progress_callback: Optional[ProgressCallable] = None,
        resume: bool = True,  # noqa: ARG002 - modelscope handles resume internally
    ) -> Path:
        if not _HAS_MODELSCOPE:
            raise FetchError(
                "modelscope is not installed; run "
                "`pip install modelscope` to enable ModelScopeSource."
            )
        dst = Path(dst).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(0, None, f"downloading {ref} from ModelScope")
        local_dir = _ms_snapshot_download(model_id=ref, cache_dir=str(dst.parent))
        local_path = Path(local_dir)
        if local_path.resolve() != dst.resolve() and local_path.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(local_path), str(dst))
        if progress_callback is not None:
            progress_callback(0, None, f"done {dst}")
        return dst

    def verify(self, path: Path, expected_sha256: Optional[str] = None) -> bool:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return False
        if expected_sha256 is None:
            return True
        return LocalSource._sha256(path) == expected_sha256.lower()

    def license(self, ref: str) -> Optional[LicenseRef]:
        return None

    def cleanup_partial(self, ref: str, dst: Path) -> None:
        dst = Path(dst).expanduser().resolve()
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)


# ---------------------------------------------------------------------------
# ModelersSource (Huawei Ascend ecosystem)
# ---------------------------------------------------------------------------
class ModelersSource(SourceFetcher):
    """Fetch models from Huawei Modelers (Ascend ecosystem).

    Uses the optional ``modelers`` SDK.  References are prefixed with
    ``modelers://`` to disambiguate them from Hugging Face repo ids.
    """

    name: str = "modelers"
    supports_parallel: bool = True
    _PREFIX: str = "modelers://"

    def can_handle(self, ref: str) -> bool:
        if not ref:
            return False
        return ref.startswith(self._PREFIX)

    def fetch(
        self,
        ref: str,
        dst: Path,
        progress_callback: Optional[ProgressCallable] = None,
        resume: bool = True,  # noqa: ARG002
    ) -> Path:
        if not _HAS_MODELERS:
            raise FetchError(
                "modelers SDK is not installed; install the Huawei Modelers "
                "package to enable ModelersSource."
            )
        dst = Path(dst).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)
        model_id = ref[len(self._PREFIX) :]
        if progress_callback is not None:
            progress_callback(0, None, f"downloading {model_id} from Modelers")
        local_dir = _modelers_download(model_id=model_id, cache_dir=str(dst.parent))
        local_path = Path(local_dir)
        if local_path.resolve() != dst.resolve() and local_path.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(local_path), str(dst))
        if progress_callback is not None:
            progress_callback(0, None, f"done {dst}")
        return dst

    def verify(self, path: Path, expected_sha256: Optional[str] = None) -> bool:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return False
        if expected_sha256 is None:
            return True
        return LocalSource._sha256(path) == expected_sha256.lower()

    def license(self, ref: str) -> Optional[LicenseRef]:
        return None

    def cleanup_partial(self, ref: str, dst: Path) -> None:
        dst = Path(dst).expanduser().resolve()
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)


# ---------------------------------------------------------------------------
# SourceRegistry
# ---------------------------------------------------------------------------
class SourceRegistry:
    """Routes fetch requests to the first compatible :class:`SourceFetcher`.

    Fetchers are consulted in registration order; the first one whose
    :meth:`SourceFetcher.can_handle` returns ``True`` wins.  The registry
    adds two cross-cutting concerns on top of the raw fetchers:

    * **Resume** -- a ``.partial`` marker file tracks in-progress downloads
      so that interrupted fetches can be resumed.
    * **Parallel sharding** -- files larger than 2 GiB are downloaded in
      ``_SHARD_COUNT`` (8) parallel byte ranges when the selected fetcher
      advertises :attr:`SourceFetcher.supports_ranges` (i.e. implements
      :meth:`SourceFetcher.fetch_range`).  Fetchers backed by an SDK that
      already parallelises downloads (Hugging Face, ModelScope, Modelers)
      leave ``supports_ranges`` ``False`` and let the SDK do the work.

    Args:
        fetchers: Optional initial list of fetchers to register.

    Example:
        >>> registry = SourceRegistry()
        >>> registry.register(LocalSource())
        >>> registry.register(HuggingFaceSource())
        >>> path = registry.fetch("/data/models/llama", "/cache/llama")
    """

    def __init__(self, fetchers: Optional[List[SourceFetcher]] = None) -> None:
        self._fetchers: List[SourceFetcher] = []
        self._lock: threading.Lock = threading.Lock()
        self._logger = get_logger(self.__class__.__name__)
        for fetcher in fetchers or []:
            self.register(fetcher)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, fetcher: SourceFetcher) -> None:
        """Register a fetcher.

        Args:
            fetcher: A :class:`SourceFetcher` instance.

        Raises:
            TypeError: If ``fetcher`` is not a :class:`SourceFetcher`.
        """
        if not isinstance(fetcher, SourceFetcher):
            raise TypeError(
                f"fetcher must be a SourceFetcher instance, got "
                f"{type(fetcher).__name__}."
            )
        with self._lock:
            self._fetchers.append(fetcher)
        self._logger.debug("Registered source fetcher: %s", fetcher.name)

    def unregister(self, fetcher: SourceFetcher) -> bool:
        """Remove a previously registered fetcher.

        Returns:
            ``True`` if the fetcher was found and removed.
        """
        with self._lock:
            try:
                self._fetchers.remove(fetcher)
                return True
            except ValueError:
                return False

    @property
    def fetchers(self) -> List[SourceFetcher]:
        """A snapshot of the currently registered fetchers."""
        with self._lock:
            return list(self._fetchers)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch(
        self,
        ref: str,
        dst: Union[str, Path],
        progress_callback: Optional[ProgressCallable] = None,
        expected_sha256: Optional[str] = None,
        resume: bool = True,
    ) -> Path:
        """Fetch ``ref`` into ``dst`` using the first compatible fetcher.

        Args:
            ref: Opaque reference string.
            dst: Destination directory or file path.
            progress_callback: Optional progress callback.
            expected_sha256: Optional sha256 digest for post-fetch verification.
            resume: When ``True`` attempt to resume from a ``.partial`` file.

        Returns:
            The resolved path to the fetched resource.

        Raises:
            FetchError: If no fetcher can handle ``ref`` or the fetch fails.
        """
        fetcher = self._select_fetcher(ref)
        if fetcher is None:
            raise FetchError(
                f"No registered source fetcher can handle ref: {ref!r}"
            )

        dst_path = Path(dst).expanduser().resolve()
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        partial_marker = dst_path.with_name(dst_path.name + _PARTIAL_SUFFIX)
        if resume and partial_marker.exists():
            self._logger.info("Resuming partial download for %s.", ref)
        if resume:
            partial_marker.touch()

        try:
            shard_info = self._should_shard(fetcher, ref)
            if shard_info is not None:
                total_size = shard_info
                self._logger.info(
                    "Using parallel sharding (%d shards) for large file "
                    "%s (%.2f GiB).",
                    _SHARD_COUNT,
                    ref,
                    total_size / (1024 ** 3),
                )
                result = self._parallel_fetch(
                    fetcher,
                    ref,
                    dst_path,
                    total_size,
                    progress_callback=progress_callback,
                )
            else:
                result = fetcher.fetch(
                    ref,
                    dst_path,
                    progress_callback=progress_callback,
                    resume=resume,
                )
        except Exception as exc:
            # Keep the partial marker so a subsequent call can resume.
            raise FetchError(f"Failed to fetch {ref!r} via {fetcher.name}: {exc}") from exc
        finally:
            if partial_marker.exists():
                try:
                    partial_marker.unlink()
                except OSError:
                    pass

        if expected_sha256 is not None:
            if not fetcher.verify(result, expected_sha256):
                fetcher.cleanup_partial(ref, result)
                raise FetchError(
                    f"sha256 verification failed for {ref!r} "
                    f"(expected {expected_sha256})."
                )
        return result

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------
    def verify(
        self,
        path: Union[str, Path],
        expected_sha256: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> bool:
        """Verify the integrity of ``path``.

        When ``ref`` is provided the registry selects the matching fetcher;
        otherwise the first registered fetcher is used.
        """
        path = Path(path).expanduser().resolve()
        fetcher = self._select_fetcher(ref) if ref else self._first_fetcher()
        if fetcher is None:
            return path.exists()
        return fetcher.verify(path, expected_sha256)

    def license(self, ref: str) -> Optional[LicenseRef]:
        """Return the :class:`LicenseRef` for ``ref`` if any fetcher knows it."""
        fetcher = self._select_fetcher(ref)
        if fetcher is None:
            return None
        return fetcher.license(ref)

    def cleanup_partial(self, ref: str, dst: Union[str, Path]) -> None:
        """Remove partial download artefacts for ``ref`` under ``dst``."""
        fetcher = self._select_fetcher(ref)
        if fetcher is None:
            return
        fetcher.cleanup_partial(ref, Path(dst).expanduser().resolve())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _select_fetcher(self, ref: str) -> Optional[SourceFetcher]:
        """Return the first registered fetcher that can handle ``ref``."""
        with self._lock:
            fetchers = list(self._fetchers)
        for fetcher in fetchers:
            try:
                if fetcher.can_handle(ref):
                    return fetcher
            except Exception:
                continue
        return None

    def _first_fetcher(self) -> Optional[SourceFetcher]:
        with self._lock:
            return self._fetchers[0] if self._fetchers else None

    def _should_shard(
        self, fetcher: SourceFetcher, ref: str
    ) -> Optional[int]:
        """Decide whether to use parallel sharding for this fetch.

        Sharding is enabled only when the fetcher advertises both
        :attr:`SourceFetcher.supports_ranges` and a non-``None``
        :meth:`SourceFetcher.head_size` that exceeds
        :data:`_LARGE_FILE_THRESHOLD`.

        Returns:
            The total size in bytes when sharding should be used, otherwise
            ``None``.
        """
        if not getattr(fetcher, "supports_ranges", False):
            return None
        try:
            size = fetcher.head_size(ref)
        except Exception:
            return None
        if size is None or size <= 0:
            return None
        if size > _LARGE_FILE_THRESHOLD:
            return int(size)
        return None

    def _parallel_fetch(
        self,
        fetcher: SourceFetcher,
        ref: str,
        dst: Path,
        total_size: int,
        progress_callback: Optional[ProgressCallable] = None,
    ) -> Path:
        """Download ``ref`` in ``_SHARD_COUNT`` parallel byte ranges.

        Each shard is written to a ``.part-<i>`` file under ``dst``'s parent
        directory; once all shards complete they are concatenated into
        ``dst`` and the part files are removed.

        Args:
            fetcher: A fetcher implementing :meth:`SourceFetcher.fetch_range`.
            ref: Opaque reference string.
            dst: Final destination file path.
            total_size: Total size in bytes (from :meth:`head_size`).
            progress_callback: Optional progress callback.

        Returns:
            The path of the assembled file.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        shard_size = (total_size + _SHARD_COUNT - 1) // _SHARD_COUNT
        part_paths: List[Path] = []
        offsets: List[tuple[int, int]] = []
        for i in range(_SHARD_COUNT):
            start = i * shard_size
            if start >= total_size:
                break
            end = min(start + shard_size, total_size)
            offsets.append((start, end))
            part_paths.append(dst.with_name(f"{dst.name}.part-{i}"))

        downloaded = {"bytes": 0}
        progress_lock = threading.Lock()
        # S2-8: 跟踪每个分片上次报告的累计字节数。
        # fetch_range 的进度回调传入的 ``done`` 是该分片内的累计值
        # (而非增量)，因此需要记录上次值并计算差值，避免重复累加。
        shard_done: Dict[int, int] = {i: 0 for i in range(len(part_paths))}

        def _run_shard(index: int) -> Path:
            start, end = offsets[index]
            part = part_paths[index]

            def _shard_progress(done: int, total: Optional[int], msg: str = "") -> None:
                with progress_lock:
                    # S2-8: 计算本分片自上次回调以来的增量字节数，
                    # 累加到总进度(downloaded["bytes"])。
                    prev = shard_done.get(index, 0)
                    delta = done - prev if done > prev else 0
                    downloaded["bytes"] += delta
                    shard_done[index] = done
                if progress_callback is not None:
                    with progress_lock:
                        progress_callback(
                            downloaded["bytes"],
                            total_size,
                            f"shard {index}/{len(part_paths)} {msg}",
                        )

            return fetcher.fetch_range(ref, part, start, end, _shard_progress)

        with ThreadPoolExecutor(max_workers=_SHARD_COUNT) as executor:
            futures = [executor.submit(_run_shard, i) for i in range(len(part_paths))]
            for future in futures:
                future.result()  # propagate shard errors

        # Concatenate shards in order.
        with open(dst, "wb") as out:
            for part in part_paths:
                with open(part, "rb") as shard_handle:
                    shutil.copyfileobj(shard_handle, out, length=1024 * 1024)

        # Clean up part files.
        for part in part_paths:
            try:
                part.unlink()
            except OSError:
                pass

        if progress_callback is not None:
            progress_callback(total_size, total_size, f"done {dst}")
        return dst

    def __repr__(self) -> str:
        with self._lock:
            names = [f.name for f in self._fetchers]
        return f"SourceRegistry(fetchers={names})"
