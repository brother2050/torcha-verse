"""Cold-tier storage protocol for the v0.6.x asset store.

This module hosts the :class:`ColdStorageProtocol` definition, the
abstract contract every cold-tier backend (S3 / OSS / MinIO / R2 /
local directory / ...) must satisfy.  The protocol was historically
defined in :mod:`assets.store`; in v0.6.x we move it here so that
implementations (in :mod:`assets.cold_storage`) and the consumer
(:mod:`assets.store`) both depend on a single, focused module.

Backward compatibility
----------------------
The protocol is still re-exported from :mod:`assets.store` (via a
thin shim) so existing callers that wrote
``from assets.store import ColdStorageProtocol`` keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["ColdStorageProtocol"]


@runtime_checkable
class ColdStorageProtocol(Protocol):
    """Protocol for cold-tier (S3 / OSS / MinIO / R2) content backends.

    The cold tier is *reserved* in v0.3.0: the :class:`AssetStore`
    accepts an optional object conforming to this protocol but does
    not yet route every read or write through it.  Future
    implementations (e.g. an S3-backed cold store) need only satisfy
    this interface to be plugged in.

    Implementations are expected to be content-addressed: every method
    is keyed by the sha256 ``content_hash`` of the stored blob.
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
