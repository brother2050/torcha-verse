#!/usr/bin/env python3
"""CI linter: ensure every ``pass`` / ``NotImplementedError`` is registered.

This script is the command-line front-end for
:mod:`infrastructure.placeholder_registry`.  It scans the entire project
tree (excluding tests, caches, and virtualenvs) for placeholder
candidates, compares them against
:doc:`/docs/placeholder_registry`, and exits non-zero when it finds
unregistered hits.

Usage::

    python scripts/check_placeholders.py                  # full project
    python scripts/check_placeholders.py path/to/file.py  # single file
    python scripts/check_placeholders.py --list           # show all registered
    python scripts/check_placeholders.py --stats          # show counts

Exit codes:

* ``0`` -- clean (or ``--list`` / ``--stats`` only).
* ``1`` -- unregistered placeholders found; details printed to stdout.
* ``2`` -- internal error (registry missing, malformed row, ...).

The script is dependency-free (no third-party imports) so it can be
invoked from any environment, including minimal CI containers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Make ``infrastructure`` importable when running from the project root.
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from infrastructure.placeholder_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    PlaceholderCategory,
    PlaceholderEntry,
    SCAN_IGNORE_DIRS,
    ScanHit,
    find_unregistered,
    load_registry,
    registry_index,
    scan_source,
)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _format_hit(h: ScanHit) -> str:
    return "{}:{}: [{}] {}".format(h.file, h.line, h.kind, h.text)


def _print_hits(hits: List[ScanHit]) -> None:
    for h in hits:
        print(_format_hit(h))


def _print_entries(entries: List[PlaceholderEntry]) -> None:
    # Group by category for readability.
    by_cat: dict[PlaceholderCategory, List[PlaceholderEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)
    for cat in PlaceholderCategory:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print("\n[{}]  ({} entries)".format(cat.value, len(items)))
        for e in items:
            desc = e.description
            if len(desc) > 70:
                desc = desc[:67] + "..."
            print("  {}:{}  {}".format(e.file, e.line, desc))


def _print_stats(
    entries: List[PlaceholderEntry],
    hits: List[ScanHit],
    unregistered: List[ScanHit],
) -> None:
    print("Registry entries: {}".format(len(entries)))
    by_cat: dict[PlaceholderCategory, int] = {}
    for e in entries:
        by_cat[e.category] = by_cat.get(e.category, 0) + 1
    for cat in PlaceholderCategory:
        n = by_cat.get(cat, 0)
        if n:
            print("  - {}: {}".format(cat.value, n))
    print("Scanner hits (file:line): {}".format(len(hits)))
    print("Unregistered hits:        {}".format(len(unregistered)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that every 'pass' / NotImplementedError is "
            "documented in docs/placeholder_registry.md."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(_PROJECT_ROOT),
        help="File or directory to scan (default: project root).",
    )
    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY_PATH,
        help=(
            "Path to the registry markdown.  Default: "
            + DEFAULT_REGISTRY_PATH
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(_PROJECT_ROOT),
        help="Project root for relative paths.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all registered entries and exit.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the 'unregistered' section when no hits found.",
    )
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    target = Path(args.target).resolve()

    # 1. Load registry.
    try:
        registry = load_registry(
            args.registry, project_root=project_root,
        )
    except FileNotFoundError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print("error: failed to parse registry: {}".format(exc), file=sys.stderr)
        return 2

    # 2. List-only or stats-only mode (no scan).
    if args.list:
        _print_entries(registry)
        return 0

    # 3. Scan.
    hits = scan_source(target, project_root=project_root)
    unregistered = find_unregistered(hits, registry)

    if args.stats:
        _print_stats(registry, hits, unregistered)
        return 0

    # 4. Default: print full report.  Exit non-zero if any unregistered.
    if not hits:
        if not args.quiet:
            print(
                "No placeholders found in {}.".format(target),
            )
        return 0

    if not unregistered:
        print(
            "OK: scanned {} placeholder(s), all registered.".format(len(hits))
        )
        return 0

    print(
        "FAIL: {} unregistered placeholder(s) out of {} scanned:".format(
            len(unregistered), len(hits),
        ),
        file=sys.stderr,
    )
    _print_hits(unregistered)
    print(
        "\nAdd the missing entries to {} and re-run.".format(
            args.registry,
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
