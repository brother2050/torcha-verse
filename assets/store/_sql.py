"""SQL query builders for :meth:`AssetStore.list` and
:meth:`AssetStore.search` (v0.6.x).

Centralising the SQL strings here keeps the public API surface
focused on the asset abstractions; the SQL itself is mechanical
``SELECT ... FROM assets WHERE ...`` glue.
"""

from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["list_query", "search_query"]


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so user-supplied substrings match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_query(
    asset_type: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> tuple:
    """Return ``(sql, params)`` for the ``list`` API.

    Filtering by tags is pushed down to SQL: each tag must appear
    in the ``tags_json`` column.
    """
    query = "SELECT metadata_json FROM assets WHERE 1=1"
    params: List[Any] = []
    if asset_type is not None:
        query += " AND asset_type = ?"
        params.append(asset_type)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if tags is not None and tags:
        for tag in tags:
            escaped_tag = _escape_like(tag)
            query += " AND tags_json LIKE ? ESCAPE '\\'"
            params.append(f'%"{escaped_tag}"%')
    query += " ORDER BY updated_at DESC"
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    query += ";"
    return query, params


def search_query(
    query_text: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> tuple:
    """Return ``(sql, params)`` for the ``search`` API.

    The match is case-insensitive on the SQLite side (LIKE) and
    is then re-applied on the in-memory :class:`Asset` objects in
    :meth:`AssetStore.search` to filter out false positives
    caused by the JSON serialisation.
    """
    escaped = _escape_like(query_text)
    pattern = f"%{escaped}%"
    sql = (
        "SELECT metadata_json FROM assets "
        "WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' "
        "OR tags_json LIKE ? ESCAPE '\\' "
        "ORDER BY updated_at DESC"
    )
    sql_params: List[Any] = [pattern, pattern, pattern]
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        sql_params.extend([limit, offset])
    sql += ";"
    return sql, sql_params
