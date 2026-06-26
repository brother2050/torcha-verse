"""Report formatters for the hardcoding scanner (v0.6.x).

Two output formats are supported:

* :func:`format_text` -- a human-readable text report (the default)
  with one violation per line.
* :func:`format_json` -- a JSON array of :class:`Violation` dicts.

Plus :func:`export_critical` which writes the critical
violations out as a YAML whitelist stub, so an operator can use
it as a starting point for a new ``hardcoded_whitelist.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ._constants import SEVERITY_CRITICAL
from ._types import Violation

__all__ = ["format_text", "format_json", "export_critical"]


def format_text(violations: List[Violation]) -> str:
    """Format ``violations`` as a human-readable text report."""
    if not violations:
        return "no hardcoding violations found"
    lines: List[str] = []
    for v in violations:
        lines.append(
            f"{v.severity.upper():8s} {v.file}:{v.line}:{v.col} "
            f"[{v.type}] {v.content}"
        )
    return "\n".join(lines)


def format_json(violations: List[Violation]) -> str:
    """Format ``violations`` as a JSON array."""
    return json.dumps([v.as_dict() for v in violations], ensure_ascii=False, indent=2)


def export_critical(
    violations: List[Violation],
    export_path: Path,
) -> int:
    """Export ``violations`` to ``export_path`` as a YAML whitelist stub.

    Only :data:`SEVERITY_CRITICAL` violations are exported;
    entries are deduplicated by ``(file, line, type)``.

    Returns:
        The number of entries written.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to export; install it with: pip install pyyaml"
        ) from exc
    seen: set = set()
    entries: List[dict] = []
    for v in violations:
        if v.severity != SEVERITY_CRITICAL:
            continue
        key = (v.file, v.line, v.type)
        if key in seen:
            continue
        seen.add(key)
        entries.append({
            "file": v.file,
            "type": v.type,
            "line": v.line,
            "content_contains": v.content.strip("'").strip('"'),
            "reason": "auto-exported by check_hardcoding.py",
        })
    with open(export_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"exemptions": entries}, fh, sort_keys=False, allow_unicode=True)
    return len(entries)
