"""Canvas sharing for the TorchaVerse v0.3.0 architecture (L5).

This module provides the mechanisms for sharing canvases between users:

* :class:`ShareLink` -- a shareable, optionally expiring and password-
  protected reference to a canvas state.
* :class:`ShareManager` -- a thread-safe manager that creates, resolves,
  revokes and lists share links, and can export / import self-contained
  canvas bundles (zip archives containing the canvas JSON plus a manifest
  of dependency asset references).

The sharing layer is *torch-free* and depends only on the canvas core
(:mod:`canvas.canvas`) and the Python standard library (``json``,
``zipfile``, ``hashlib``, ``time``, ``uuid``).

Public surface
--------------

* :class:`ShareLink` -- dataclass describing a single share link.
* :class:`ShareManager` -- the share-link and bundle manager.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .canvas import Canvas, CanvasState

__all__ = ["ShareLink", "ShareManager"]

# ---------------------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("canvas.sharing")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Default link expiry in hours (7 days).
_DEFAULT_EXPIRY_HOURS: float = 24.0 * 7.0
#: Number of seconds in one hour.
_SECONDS_PER_HOUR: float = 3600.0
#: Filename for the canvas JSON inside a bundle.
_BUNDLE_CANVAS_FILENAME: str = "canvas.json"
#: Filename for the manifest JSON inside a bundle.
_BUNDLE_MANIFEST_FILENAME: str = "manifest.json"
#: Bundle format version.
_BUNDLE_FORMAT_VERSION: str = "1.0.0"
#: Hash algorithm used for password verification.
_PASSWORD_HASH_ALGORITHM: str = "sha256"


def _hash_password(password: str) -> str:
    """Return a salted SHA-256 hex digest of ``password``.

    A random salt is prepended to the password before hashing so that
    identical passwords produce different hashes.

    Args:
        password: The plaintext password.

    Returns:
        A ``"salt:hash"`` string.
    """
    salt = os.urandom(16).hex()
    digest = hashlib.new(
        _PASSWORD_HASH_ALGORITHM,
        (salt + password).encode("utf-8"),
    ).hexdigest()
    return "{}:{}".format(salt, digest)


def _verify_password(password: str, stored: str) -> bool:
    """Verify ``password`` against a stored ``"salt:hash"`` string.

    Args:
        password: The plaintext password to check.
        stored: The stored ``"salt:hash"`` string.

    Returns:
        ``True`` if the password matches.
    """
    if ":" not in stored:
        return False
    salt, expected_hash = stored.split(":", 1)
    digest = hashlib.new(
        _PASSWORD_HASH_ALGORITHM,
        (salt + password).encode("utf-8"),
    ).hexdigest()
    # Use constant-time comparison to prevent timing side-channel attacks.
    import hmac as _hmac
    return _hmac.compare_digest(digest, expected_hash)


# ---------------------------------------------------------------------------
# ShareLink
# ---------------------------------------------------------------------------
@dataclass
class ShareLink:
    """A shareable reference to a canvas state.

    A share link captures a snapshot of a canvas at creation time.  It can
    optionally expire after a given number of hours, require a password
    for access, and be marked as view-only (preventing the recipient from
    editing the canvas).

    Attributes:
        link_id: Unique identifier for this link.
        canvas_state: The :class:`CanvasState` captured by this link.
        created_at: POSIX timestamp at which the link was created.
        expires_at: POSIX timestamp after which the link is invalid
            (``None`` for a link that never expires).
        password: Optional salted hash of the access password
            (``None`` for no password).
        view_only: When ``True``, the shared canvas should be treated as
            read-only.
    """

    link_id: str
    canvas_state: CanvasState
    created_at: float
    expires_at: Optional[float] = None
    password: Optional[str] = None
    view_only: bool = True

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return ``True`` if this link has expired.

        Args:
            now: The current POSIX timestamp.  When ``None``,
                :func:`time.time` is used.

        Returns:
            ``True`` if the link has an expiry and it has passed.
        """
        if self.expires_at is None:
            return False
        current = now if now is not None else time.time()
        return current >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this link to a JSON-serialisable dictionary."""
        return {
            "link_id": self.link_id,
            "canvas_state": self.canvas_state.to_dict(),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "password": self.password,
            "view_only": self.view_only,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShareLink":
        """Reconstruct a :class:`ShareLink` from a serialised dict.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`ShareLink` instance.
        """
        return cls(
            link_id=d["link_id"],
            canvas_state=CanvasState.from_dict(d.get("canvas_state") or {}),
            created_at=float(d.get("created_at", time.time())),
            expires_at=(
                float(d["expires_at"]) if d.get("expires_at") is not None else None
            ),
            password=d.get("password"),
            view_only=bool(d.get("view_only", True)),
        )

    def __repr__(self) -> str:
        return "ShareLink(id={!r}, view_only={}, expired={})".format(
            self.link_id, self.view_only, self.is_expired()
        )


# ---------------------------------------------------------------------------
# ShareManager
# ---------------------------------------------------------------------------
class ShareManager:
    """Thread-safe manager for canvas share links and bundles.

    A :class:`ShareManager` maintains an in-memory store of
    :class:`ShareLink` objects and provides operations to create, resolve,
    revoke and list them.  It also supports exporting a canvas as a
    self-contained zip bundle (canvas JSON + manifest) and importing it
    back.

    All operations are guarded by a re-entrant lock for thread safety.

    Example::

        sm = ShareManager()
        link = sm.create_link(canvas, expires_in_hours=24, password="secret")
        state = sm.resolve_link(link.link_id, password="secret")
    """

    def __init__(self) -> None:
        self._links: Dict[str, ShareLink] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Link management
    # ------------------------------------------------------------------
    def create_link(
        self,
        canvas: Canvas,
        expires_in_hours: Optional[float] = None,
        password: Optional[str] = None,
        view_only: bool = True,
    ) -> ShareLink:
        """Create a share link for a canvas.

        Args:
            canvas: The :class:`Canvas` to share.  A snapshot of its
                current state is captured.
            expires_in_hours: Optional expiry duration in hours.  When
                ``None`` the link never expires.
            password: Optional access password.  When provided, callers
                of :meth:`resolve_link` must supply the same password.
            view_only: When ``True``, the shared canvas is read-only.

        Returns:
            The newly created :class:`ShareLink`.
        """
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance.")
        with self._lock:
            state = CanvasState.from_dict(canvas.state.to_dict())
            created_at = time.time()
            expires_at: Optional[float] = None
            if expires_in_hours is not None:
                expires_at = created_at + (
                    float(expires_in_hours) * _SECONDS_PER_HOUR
                )
            link = ShareLink(
                link_id=str(uuid4()),
                canvas_state=state,
                created_at=created_at,
                expires_at=expires_at,
                password=_hash_password(password) if password else None,
                view_only=view_only,
            )
            self._links[link.link_id] = link
            _logger.debug("Created share link %r.", link.link_id)
            return link

    def resolve_link(
        self, link_id: str, password: Optional[str] = None
    ) -> Optional[CanvasState]:
        """Resolve a share link and return its canvas state.

        Args:
            link_id: The id of the share link to resolve.
            password: The access password (required only if the link was
                created with one).

        Returns:
            A copy of the link's :class:`CanvasState`, or ``None`` if the
            link does not exist, has expired, or the password is wrong.
        """
        with self._lock:
            link = self._links.get(link_id)
            if link is None:
                _logger.debug("Share link %r not found.", link_id)
                return None
            if link.is_expired():
                _logger.debug("Share link %r has expired.", link_id)
                return None
            if link.password is not None:
                if password is None or not _verify_password(
                    password, link.password
                ):
                    _logger.debug(
                        "Share link %r password mismatch.", link_id
                    )
                    return None
            return CanvasState.from_dict(link.canvas_state.to_dict())

    def revoke_link(self, link_id: str) -> bool:
        """Revoke (delete) a share link.

        Args:
            link_id: The id of the link to revoke.

        Returns:
            ``True`` if the link was found and revoked.
        """
        with self._lock:
            removed = self._links.pop(link_id, None)
            if removed is not None:
                _logger.debug("Revoked share link %r.", link_id)
                return True
            return False

    def list_links(self) -> List[ShareLink]:
        """Return a list of all active share links.

        Returns:
            A list of :class:`ShareLink` instances (insertion order).
        """
        with self._lock:
            return list(self._links.values())

    # ------------------------------------------------------------------
    # Bundle export / import
    # ------------------------------------------------------------------
    def export_bundle(
        self, canvas: Canvas, path: Union[str, Path]
    ) -> Path:
        """Export a canvas as a self-contained zip bundle.

        The bundle contains:

        * ``canvas.json`` -- the full canvas serialised as JSON.
        * ``manifest.json`` -- metadata (canvas name, node / connection
          counts, export timestamp, format version, and a list of asset
          references found in the canvas node inputs).

        Args:
            canvas: The :class:`Canvas` to export.
            path: Destination file path for the zip bundle.

        Returns:
            The resolved path that was written.
        """
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance.")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        canvas_json = canvas.to_json()
        asset_refs = self._collect_asset_refs(canvas)
        manifest: Dict[str, Any] = {
            "canvas_name": canvas.name,
            "node_count": len(canvas.list_nodes()),
            "connection_count": len(canvas.list_connections()),
            "exported_at": time.time(),
            "format_version": _BUNDLE_FORMAT_VERSION,
            "asset_references": asset_refs,
        }

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_BUNDLE_CANVAS_FILENAME, canvas_json)
            zf.writestr(
                _BUNDLE_MANIFEST_FILENAME,
                json.dumps(manifest, indent=2, ensure_ascii=False),
            )
        _logger.info("Exported canvas bundle to %s.", target)
        return target

    def import_bundle(self, path: Union[str, Path]) -> Canvas:
        """Import a canvas from a zip bundle.

        Args:
            path: Path to the zip bundle.

        Returns:
            A reconstructed :class:`Canvas`.

        Raises:
            FileNotFoundError: If the bundle does not exist.
            ValueError: If the bundle is malformed or missing the canvas
                JSON.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                "Bundle not found: {}".format(source)
            )
        with zipfile.ZipFile(source, "r") as zf:
            names = zf.namelist()
            if _BUNDLE_CANVAS_FILENAME not in names:
                raise ValueError(
                    "Bundle {!r} is missing {}.".format(
                        source, _BUNDLE_CANVAS_FILENAME
                    )
                )
            canvas_json = zf.read(_BUNDLE_CANVAS_FILENAME).decode("utf-8")
        canvas = Canvas.from_json(canvas_json)
        _logger.info("Imported canvas from bundle %s.", source)
        return canvas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_asset_refs(canvas: Canvas) -> List[Dict[str, str]]:
        """Scan canvas node inputs for asset references.

        This performs a best-effort scan for values that look like asset
        references (objects with an ``asset_id`` attribute, or dicts with
        an ``asset_id`` key).

        Args:
            canvas: The canvas to scan.

        Returns:
            A list of ``{"node_id", "input_key", "asset_id"}`` dicts.
        """
        refs: List[Dict[str, str]] = []
        for node in canvas.list_nodes():
            for key, value in node.inputs.items():
                asset_id = None
                if hasattr(value, "asset_id"):
                    asset_id = getattr(value, "asset_id", None)
                elif isinstance(value, dict):
                    asset_id = value.get("asset_id")
                if asset_id is not None:
                    refs.append(
                        {
                            "node_id": node.id,
                            "input_key": key,
                            "asset_id": str(asset_id),
                        }
                    )
        return refs

    def __repr__(self) -> str:
        with self._lock:
            return "ShareManager(links={})".format(len(self._links))
