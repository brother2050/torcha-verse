"""L2 AssetStore sub-package (v0.6.x).

The :class:`AssetStore` was historically a single 861-line file
(``assets/store.py``) that bundled together:

* the :class:`ColdStorageProtocol` contract
* the SQLite schema + lifecycle
* the content-addressed object store (warm tier)
* the in-process LRU cache (hot tier)
* the cold-tier routing (push / promote / evict)
* the SQL query builders for ``list`` / ``search``
* the public :class:`AssetStore` class itself

In v0.6.x we split it into focused sub-modules.  This ``__init__``
re-exports the public API -- :class:`AssetStore` and
:class:`ColdStorageProtocol` -- so callers that import from
``assets.store`` keep working.

Backward compatibility
----------------------
* ``from assets.store import AssetStore`` -- works.
* ``from assets.store import ColdStorageProtocol`` -- still works
  (re-exported from the moved protocol module).
* ``import assets.store`` -- works; ``assets.store.AssetStore``
  and ``assets.store.ColdStorageProtocol`` resolve to the same
  class objects the v0.4.x test suite expects.
"""

from __future__ import annotations

# Re-export the moved protocol from its new home.
from ._protocol import ColdStorageProtocol
from ._store import AssetStore

__all__ = ["AssetStore", "ColdStorageProtocol"]
