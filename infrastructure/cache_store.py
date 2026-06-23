"""Thread-safe LRU cache for TorchaVerse.

Provides :class:`CacheStore`, an in-memory least-recently-used cache
intended for high-frequency inference results.  Entries optionally expire
after a configurable time-to-live (TTL).  The store is fully thread-safe
and exposes hit-rate statistics.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

__all__ = ["CacheStore"]


class CacheStore:
    """Thread-safe LRU cache with optional TTL expiry.

    Args:
        max_size: Maximum number of entries.  When exceeded the
            least-recently-used entry is evicted.  Must be ``> 0``.
        ttl: Time-to-live in seconds for every entry.  ``None`` disables
            expiry (entries live until evicted by the LRU policy).

    Example:
        >>> cache = CacheStore(max_size=128, ttl=60)
        >>> cache.set("key", {"loss": 0.1})
        >>> cache.get("key")
        {'loss': 0.1}
        >>> cache.stats()
        {'size': 1, 'max_size': 128, 'hits': 1, 'misses': 0, 'hit_rate': 1.0}
    """

    def __init__(self, max_size: int = 1024, ttl: Optional[float] = None) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}.")
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be > 0 or None, got {ttl}.")

        self._max_size: int = int(max_size)
        self._ttl: Optional[float] = float(ttl) if ttl is not None else None
        self._data: "OrderedDict[Any, tuple]" = OrderedDict()
        self._lock = threading.RLock()

        # Statistics.
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expirations: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def max_size(self) -> int:
        """Maximum number of cached entries."""
        return self._max_size

    @property
    def ttl(self) -> Optional[float]:
        """Time-to-live in seconds (``None`` means no expiry)."""
        return self._ttl

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._data)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve a value by ``key``.

        Expired entries are removed lazily and treated as misses.  Accessing
        an entry marks it as most-recently-used.

        Args:
            key: Cache key.
            default: Value returned when the key is absent or expired.

        Returns:
            The cached value or ``default``.
        """
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return default

            value, expires_at = self._data[key]
            if self._is_expired(expires_at):
                # Lazy expiration.
                del self._data[key]
                self._expirations += 1
                self._misses += 1
                return default

            # Mark as most-recently-used.
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Store ``value`` under ``key``.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Optional per-entry TTL overriding the default.  ``None`` uses
                the store's default TTL.
        """
        with self._lock:
            expires_at = self._compute_expiry(ttl)

            if key in self._data:
                # Update in place and mark as MRU.
                self._data[key] = (value, expires_at)
                self._data.move_to_end(key)
                return

            # Evict the LRU entry when at capacity.
            while len(self._data) >= self._max_size:
                self._data.popitem(last=False)
                self._evictions += 1

            self._data[key] = (value, expires_at)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Return ``key`` if present, otherwise insert ``default`` and return it."""
        with self._lock:
            existing = self.get(key, _MISSING)
            if existing is not _MISSING:
                return existing
            self.set(key, default)
            return default

    def pop(self, key: Any, default: Any = None) -> Any:
        """Remove and return the value for ``key``."""
        with self._lock:
            if key not in self._data:
                return default
            value, _ = self._data.pop(key)
            return value

    def contains(self, key: Any) -> bool:
        """Return ``True`` if ``key`` exists and has not expired."""
        with self._lock:
            if key not in self._data:
                return False
            _, expires_at = self._data[key]
            if self._is_expired(expires_at):
                del self._data[key]
                self._expirations += 1
                return False
            return True

    __contains__ = contains

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def prune_expired(self) -> int:
        """Actively remove all expired entries.

        Returns:
            The number of entries removed.
        """
        if self._ttl is None:
            return 0
        removed = 0
        with self._lock:
            expired_keys = [
                key
                for key, (_, expires_at) in self._data.items()
                if self._is_expired(expires_at)
            ]
            for key in expired_keys:
                del self._data[key]
                removed += 1
            self._expirations += removed
        return removed

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return a dictionary of cache statistics.

        Includes current size, capacity, hit/miss counts, eviction and
        expiration counts, and the overall hit rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "hit_rate": round(hit_rate, 4),
            }

    def reset_stats(self) -> None:
        """Reset only the statistics counters (entries are preserved)."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_expiry(self, ttl: Optional[float]) -> Optional[float]:
        """Compute the absolute expiry timestamp for a new entry."""
        effective_ttl = self._ttl if ttl is None else ttl
        if effective_ttl is None:
            return None
        if effective_ttl <= 0:
            raise ValueError(f"ttl must be > 0 or None, got {effective_ttl}.")
        return time.monotonic() + float(effective_ttl)

    def _is_expired(self, expires_at: Optional[float]) -> bool:
        """Return ``True`` if ``expires_at`` has passed."""
        if expires_at is None:
            return False
        return time.monotonic() >= expires_at

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"CacheStore(max_size={self._max_size}, ttl={self._ttl}, "
            f"size={len(self._data)})"
        )


# Sentinel used to distinguish a missing key from a stored ``None`` value.
_MISSING: Any = object()
