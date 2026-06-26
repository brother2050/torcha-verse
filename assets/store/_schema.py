"""SQLite schema for the warm-tier metadata index (v0.6.x).

Kept in its own module so the schema can be versioned / inspected
without dragging in the full :class:`AssetStore`.
"""

from __future__ import annotations

__all__ = ["SCHEMA"]


SCHEMA: str = """\
CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT    PRIMARY KEY,
    asset_type     TEXT    NOT NULL,
    name           TEXT    NOT NULL,
    description    TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL,
    tags_json      TEXT    NOT NULL DEFAULT '[]',
    metadata_json  TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    updated_at     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_type   ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
CREATE INDEX IF NOT EXISTS idx_assets_name   ON assets(name);
"""


#: Read buffer size used when hashing / copying content files.
CHUNK_SIZE: int = 1 << 20  # 1 MiB
