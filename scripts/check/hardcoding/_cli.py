"""CLI for the hardcoding scanner (v0.6.x).

Argument parsing + exit-code logic lives here.  The CLI is
deliberately thin: the actual scan / whitelist / formatting work
is delegated to the other modules in
:mod:`scripts.check.hardcoding`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# When the script is run directly via ``python scripts/check/hardcoding
# /_cli.py`` (e.g. by a CI shell step or a subprocess test), the package
# root is *not* on ``sys.path`` -- inject it.  Library callers that
# already have the project on ``sys.path`` pay a one-time O(1) cost.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.check.hardcoding_rules import Rule, get_rule, list_rule_names

from scripts.check.hardcoding import (
    SEVERITY_ORDER,
    export_critical,
    format_json,
    format_text,
    scan_directory,
    scan_file,
    filter_by_severity,
    load_whitelist,
)

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``scripts/check_hardcoding.py`` command.

    Returns:
        ``0`` on success, ``1`` when violations are found, ``2`` on
        usage / configuration error.
    """
    parser = argparse.ArgumentParser(description="TorchaVerse hardcoding scanner (D1).")
    parser.add_argument(
        "--path", default=".", help="File or directory to scan (default: cwd).",
    )
    parser.add_argument(
        "--whitelist", type=Path, default=None,
        help="YAML file with whitelist entries.",
    )
    parser.add_argument(
        "--severity", choices=SEVERITY_ORDER, default="critical",
        help="Minimum severity to report (default: critical).",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Export violations to this YAML file as a whitelist stub.",
    )
    parser.add_argument(
        "--list-rules", action="store_true",
        help="List all known rules and exit.",
    )
    parser.add_argument(
        "--only-rule", default=None,
        help="Run only the named rule (e.g. ``string_literal``).",
    )
    args = parser.parse_args(argv)

    if args.list_rules:
        sys.stdout.write("\n".join(list_rule_names()) + "\n")
        return 0

    scan_path = Path(args.path).resolve()
    if not scan_path.exists():
        sys.stderr.write(f"path not found: {scan_path}\n")
        return 2

    rules: Optional[List[Rule]] = None
    if args.only_rule:
        rule = get_rule(args.only_rule)
        if rule is None:
            sys.stderr.write(f"unknown rule: {args.only_rule}\n")
            return 2
        rules = [rule]

    violations = (
        [v for v in scan_file(scan_path, scan_path.name, rules=rules)]
        if scan_path.is_file()
        else scan_directory(scan_path, rules=rules)
    )

    exemptions = load_whitelist(args.whitelist) if args.whitelist else []
    filtered = filter_by_severity(violations, args.severity, exemptions)

    if args.export:
        n = export_critical(filtered, args.export)
        sys.stderr.write(
            "Wrote {} critical entries to {}\n".format(n, args.export)
        )

    if args.format == "json":
        sys.stdout.write(format_json(filtered) + "\n")
    else:
        sys.stdout.write(format_text(filtered) + "\n")

    return 1 if filtered else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
