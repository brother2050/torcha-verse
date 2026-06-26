"""SQLite lifecycle helpers for :class:`AssetStore` (v0.6.x).

The store's metadata index is a single SQLite database file in WAL
mode with ``check_same_thread=False``.  This module hosts:

* :func:`open_connection` -- open a new SQLite connection in WAL
  mode and apply the standard pragmas.
* :func:`ensure_open` -- raise :class:`RuntimeError` when the
  store has been closed (the S2-7 use-after-close guard).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from ._schema import SCHEMA

__all__ = ["open_connection", "ensure_open"]


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection in WAL mode and create the schema."""
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        isolation_level=None,  # autocommit
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)
    return conn


def ensure_open(closed_flag: bool) -> None:
    """Raise :class:`RuntimeError` if the store has been closed (S2-7).

    All public methods call this before touching shared state so
    that use-after-close is detected eagerly rather than producing
    obscure SQLite errors.
    """
    if closed_flag:
        raise RuntimeError("AssetStore is closed")
