"""File readers used by the v0.6.x :mod:`training.dataset` sub-package.

Three pure-IO helpers that load a tabular data file into a list
of ``{column: value}`` dicts so the dataset classes can work in
either JSONL / CSV / Parquet form without owning the
format-specific IO:

* :func:`read_jsonl_rows` -- one JSON object per line.
* :func:`read_csv_rows` -- :class:`csv.DictReader` with empty
  cells normalised to the empty string.
* :func:`read_parquet_rows` -- tries ``pyarrow`` first, then
  ``pandas``, with a clear error message when neither is
  installed.

The readers are deliberately decoupled from any
:class:`BaseDataset` so they can be reused by other framework
subsystems (e.g. the RAG store) without paying the cost of
importing the dataset classes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["read_jsonl_rows", "read_csv_rows", "read_parquet_rows"]


#: Path-like type alias used throughout the dataset sub-package.
PathLike = str  # convenience alias re-exported as ``str`` for callers
                 # that only need to type-annotate their file paths.


def read_jsonl_rows(file_path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSON-Lines file and return a list of dicts.

    Blank lines are silently skipped.  Malformed lines raise
    :class:`json.JSONDecodeError` with the offending line number.
    """
    out: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def read_csv_rows(file_path: PathLike) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of ``{column: value}`` dicts.

    Values are returned as plain strings (no automatic type
    coercion) so the dataset layer can decide what to do with
    them.  Empty cells become the empty string, never ``None``,
    to keep downstream ``row.get("text", "")`` lookups robust.
    """
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_parquet_rows(file_path: PathLike) -> List[Dict[str, Any]]:
    """Read a Parquet file and return a list of ``{column: value}`` dicts.

    Parquet support is opt-in: this helper tries ``pyarrow``
    first, then ``pandas``, and finally raises a clear error
    when neither optional dependency is available.  The dataset
    layer is therefore usable in zero-dependency environments
    *as long as* the operator avoids Parquet files.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    # pyarrow is the canonical, lightweight reader.
    try:
        import pyarrow.parquet as _pq  # type: ignore[import-not-found]

        table = _pq.read_table(p)
        columns = table.column_names
        out: List[Dict[str, Any]] = []
        for i in range(table.num_rows):
            row: Dict[str, Any] = {}
            for col in columns:
                value = table.column(col)[i].as_py()
                row[str(col)] = "" if value is None else value
            out.append(row)
        return out
    except ImportError:
        pass
    # pandas is the second-choice reader -- many production
    # environments already have it installed for data analysis.
    try:
        import pandas as _pd  # type: ignore[import-not-found]

        df = _pd.read_parquet(p)
        # ``df.to_dict("records")`` returns a list of plain dicts
        # with Python native types -- exactly what the dataset
        # layer expects.
        records: List[Dict[str, Any]] = df.to_dict("records")
        return [
            {str(k): ("" if v is None else v) for k, v in row.items()}
            for row in records
        ]
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires either `pyarrow` or `pandas`. "
            "Install one of them (e.g. `pip install pyarrow`) and retry."
        ) from exc
