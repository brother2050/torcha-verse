"""Hot-tier (in-process LRU cache) helpers (v0.6.x).

The hot tier is a simple LRU cache (an :class:`OrderedDict` guarded
by the store-wide re-entrant lock) that holds recently used
:class:`Asset` objects for zero-copy access.

This module hosts the *helpers* only -- the public API lives on
:class:`AssetStore` in :mod:`assets.store._store`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, Optional

from assets.base import Asset

__all__ = ["HotCache"]


class HotCache:
    """A bounded LRU cache of :class:`Asset` instances keyed by id.

    The cache is intentionally *not* thread-safe on its own; the
    caller is expected to hold the store-wide re-entrant lock for
    the duration of any read or write.  Splitting the cache out of
    the main :class:`AssetStore` class keeps the LRU bookkeeping
    testable in isolation.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"hot_size must be > 0, got {capacity}.")
        self._capacity: int = int(capacity)
        self._items: "OrderedDict[str, Asset]" = OrderedDict()

    @property
    def capacity(self) -> int:
        """Maximum number of entries in the cache."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def get(self, asset_id: str) -> Optional[Asset]:
        """Return the cached asset, marking it most-recently-used."""
        asset = self._items.get(asset_id)
        if asset is not None:
            self._items.move_to_end(asset_id)
        return asset

    def put(self, asset: Asset) -> None:
        """Insert / update an asset, evicting LRU entries as needed."""
        if asset.id in self._items:
            self._items.move_to_end(asset.id)
        self._items[asset.id] = asset
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop every cached entry (used by :meth:`AssetStore.close`)."""
        self._items.clear()


def load_asset_from_row(
    row: tuple,
    hot: HotCache,
) -> Optional[Asset]:
    """Re-hydrate an :class:`Asset` from a ``SELECT metadata_json`` row.

    Returns ``None`` when the row is empty.  The freshly-loaded
    asset is inserted into the hot cache before being returned.
    """
    if row is None:
        return None
    asset = Asset.from_dict(json.loads(row[0]))
    hot.put(asset)
    return asset
