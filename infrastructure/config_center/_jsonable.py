"""JSON-serialisation helpers for :class:`ConfigCenter` snapshots.

Split out so the main :mod:`infrastructure.config_center._center`
module can stay focused on load / query / snapshot semantics and
so the small recursive converter is independently testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["to_jsonable"]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable primitives.

    * :class:`pathlib.Path` objects are converted to strings.
    * Tuples become lists (Python ``json`` does not serialise tuples
      as JSON arrays, which surprises users who expect round-trip).
    * All other primitives are returned as-is.

    Args:
        obj: The value to convert.

    Returns:
        A JSON-serialisable equivalent of ``obj``.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
