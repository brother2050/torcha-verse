"""Whitelist + severity filtering for the hardcoding scanner (v0.6.x).

The :func:`load_whitelist` helper reads a YAML file of
:class:`Exemption` entries; :func:`filter_by_severity` applies
those exemptions to a list of :class:`Violation` objects and
filters the result to a given minimum severity.

The YAML schema is::

    exemptions:
      - file: "core/*.py"
        type: string_literal        # or "*" / "all"
        line: 42                    # optional, exact line match
        content_contains: "secret"  # optional, substring match
        severity: info              # optional, downgrade
        protocol_format: true       # optional, mark as info
        reason: "intentional constant"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from ._constants import SEVERITY_ORDER
from ._types import Exemption, Violation

__all__ = ["load_whitelist", "filter_by_severity"]


def load_whitelist(path: Path) -> List[Exemption]:
    """Load a YAML whitelist from ``path``.

    Returns an empty list (and prints a warning) when ``path`` does
    not exist -- the v0.4.x behaviour was to ``sys.exit(2)`` here,
    but a missing whitelist is "no exemptions" not a hard error
    in v0.6.x.  Invalid severities still ``sys.exit(2)``.
    """
    if not Path(path).exists():
        sys.stderr.write(
            "error: whitelist not found: {}\n".format(path)
        )
        sys.exit(2)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the whitelist; "
            "install it with: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw_entries = data.get("exemptions", data if isinstance(data, list) else [])
    out: List[Exemption] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        file_glob = entry.get("file", "")
        if not file_glob:
            continue
        rules_raw = entry.get("rules")
        rules: Optional[set] = None
        if rules_raw is not None:
            if isinstance(rules_raw, str):
                rules = {rules_raw}
            else:
                try:
                    rules = set(rules_raw)
                except TypeError:
                    rules = None
        sev = entry.get("severity")
        if sev is not None and sev not in SEVERITY_ORDER:
            sys.stderr.write(
                "error: invalid severity {!r} in whitelist {!r} (entry: {})\n".format(
                    sev, path, entry,
                )
            )
            sys.exit(2)
        out.append(Exemption(
            file=file_glob,
            type=entry.get("type", "*"),
            line=entry.get("line"),
            content_contains=entry.get("content_contains"),
            severity=sev,
            protocol_format=bool(entry.get("protocol_format", False)),
            rules=rules,
            reason=entry.get("reason", ""),
        ))
    return out


def filter_by_severity(
    violations: List[Violation],
    min_severity: str,
    exemptions: Optional[List[Exemption]] = None,
) -> List[Violation]:
    """Apply exemptions + filter to ``violations``.

    Exemptions are applied first (a matching entry can downgrade a
    violation's severity or remove it entirely).  The remaining
    violations are then filtered to those at or above
    ``min_severity`` (in :data:`SEVERITY_ORDER`).
    """
    if min_severity not in SEVERITY_ORDER:
        raise ValueError(f"Unknown severity: {min_severity!r}")
    min_idx = SEVERITY_ORDER.index(min_severity)
    exemptions = exemptions or []
    out: List[Violation] = []
    for v in violations:
        # Apply all matching exemptions in order (later wins).
        for ex in exemptions:
            ex.apply(v)
        if SEVERITY_ORDER.index(v.severity) <= min_idx:
            out.append(v)
    return out
