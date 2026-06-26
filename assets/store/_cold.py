"""Cold-tier routing helpers for :class:`AssetStore` (v0.6.x).

These helpers isolate the "talk to the cold tier" logic from the
:class:`AssetStore` core so that the main class stays focused on
the hot / warm tier.  Three operations are exposed:

* :func:`push_to_cold` -- mirror a blob to the cold tier after a
  successful warm write.  Errors are logged but never raised.
* :func:`promote_from_cold` -- download a blob from the cold tier
  into the warm tier.  Returns the warm path on success, ``None``
  on failure.
* :func:`evict_to_cold` -- remove a blob from the warm tier (the
  cold tier is untouched).  Returns ``True`` if a blob was removed.

All three accept a logger and an :class:`objects_dir` rather than
coupling themselves to :class:`AssetStore` -- the v0.4.x test
suite already covers them via the public :class:`AssetStore`
methods.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

from ._warm import content_path, hash_file
from ._protocol import ColdStorageProtocol

__all__ = [
    "push_to_cold",
    "promote_from_cold",
    "evict_to_cold",
]


def _log_warning(logger, msg: str, *args: object) -> None:
    """Best-effort log; never raise from a logging call."""
    if logger is None:
        return
    try:
        logger.warning(msg, *args)
    except Exception:  # noqa: BLE001
        pass  # placeholder #92 (assets/store/_cold.py:45) -- logging warning() 异常时静默兜底,不该因 logger 失败让冷层路由挂掉


def _log_error(logger, msg: str, *args: object) -> None:
    if logger is None:
        return
    try:
        logger.error(msg, *args)
    except Exception:  # noqa: BLE001
        pass  # placeholder #93 (assets/store/_cold.py:54) -- logging error() 异常时静默兜底,同上


def push_to_cold(
    cold_storage: Optional[ColdStorageProtocol],
    enabled: bool,
    content_hash: str,
    content_path_arg: Path,
    logger=None,
) -> None:
    """Mirror ``content_path_arg`` to the cold tier (best-effort).

    Cold-tier errors are logged but never raised: a transient
    S3 / MinIO outage should not cause a successful warm write
    to fail.  The blob stays in the warm tier and the next
    :func:`promote_from_cold` can attempt to re-mirror on demand.
    """
    if cold_storage is None or not enabled:
        return
    try:
        cold_storage.store(content_hash, content_path_arg)
    except Exception as exc:  # noqa: BLE001
        _log_warning(
            logger,
            "Cold-tier store raised %s for hash=%s: %s",
            type(exc).__name__, content_hash[:12], exc,
        )


def promote_from_cold(
    cold_storage: Optional[ColdStorageProtocol],
    objects_dir: Path,
    content_hash: str,
    logger=None,
) -> Optional[Path]:
    """Download a blob from the cold tier and stage it in the warm tier.

    On success returns the warm-tier path; on failure (cold tier
    unconfigured, blob missing, network error) returns ``None``
    and logs a warning.
    """
    if cold_storage is None:
        return None
    try:
        staging_dir = objects_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / f".promote-{uuid4().hex}.tmp"
        cold_storage.fetch(content_hash, staging_path)
        # Re-verify the hash so a corrupted fetch never lands
        # in the warm tier.
        actual = hash_file(staging_path)
        if actual != content_hash:
            staging_path.unlink(missing_ok=True)
            _log_error(
                logger,
                "Cold fetch hash mismatch for %s: got %s, expected %s",
                content_hash[:12], actual[:12], content_hash[:12],
            )
            return None
        final_path = content_path(objects_dir, content_hash)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_path, final_path)
        return final_path
    except Exception as exc:  # noqa: BLE001
        _log_warning(
            logger,
            "Cold-tier promote raised %s for hash=%s: %s",
            type(exc).__name__, content_hash[:12], exc,
        )
        return None


def evict_to_cold(
    cold_storage: Optional[ColdStorageProtocol],
    objects_dir: Path,
    content_hash: str,
) -> bool:
    """Remove a blob from the warm tier; the cold tier keeps a copy.

    Useful to free disk space in the warm tier once the cold
    tier is configured.  The blob is **not** removed from the
    cold tier -- a subsequent :func:`promote_from_cold` can
    re-fetch it.

    Returns:
        ``True`` if a warm blob was removed, ``False`` if no
        such blob was present.
    """
    if cold_storage is None:
        return False
    path = content_path(objects_dir, content_hash)
    if not path.exists():
        return False
    path.unlink()
    try:
        if path.parent.exists() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    except OSError:
        pass  # placeholder #94 (assets/store/_cold.py:152) -- 删空 shard 目录失败兜底,与 v0.4.x assets/store.py:316 行为一致
    return True
