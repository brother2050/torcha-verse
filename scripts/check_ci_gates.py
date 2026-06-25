"""Aggregate CI gate entry point (D1 stage three, v0.4.x).

This script is the single command the CI pipeline runs to
enforce all "non-negotiable" engineering gates declared in
``pyproject.toml``'s ``[tool.torcha-verse.ci-gates]`` table:

* hardcoding (D1) -- exits non-zero when ``critical`` violations
  remain after the whitelist is applied;
* placeholders (D3) -- exits non-zero when an unregistered
  ``pass`` / ``NotImplementedError`` is found;
* degrade_logging (D3 stage three) -- exits non-zero when a
  silent-degrade ``except ...: pass`` block is found that does
  not contain ``logger.warning`` / ``safe_call`` /
  ``record_degrade`` / explicit ``raise``.  Default is **off**
  until the 38 known sites are fixed (see
  ``docs/placeholder_registry.md`` section 2.4);
* future gates (TODO) can register themselves by adding a
  ``[tool.torcha-verse.ci-gates.<name>]`` table that contains
  an ``enabled = true`` key.

The point of this script is to make the CI pipeline a *single
command* (``python scripts/check_ci_gates.py``) so future gates
can be added without touching the CI YAML.

Usage::

    python scripts/check_ci_gates.py            # run all enabled gates
    python scripts/check_ci_gates.py --gate hardcoding   # only the named gate
    python scripts/check_ci_gates.py --list                # list gates and exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ci_config import _find_pyproject, _parse_mini_toml
except ImportError:
    from scripts.ci_config import _find_pyproject, _parse_mini_toml

__all__ = ["main", "GATE_REGISTRY"]


# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------
class GateResult:
    """Outcome of a single gate run."""

    def __init__(self, name: str, exit_code: int, summary: str) -> None:
        self.name = name
        self.exit_code = exit_code
        self.summary = summary

    def as_line(self) -> str:
        marker = "PASS" if self.exit_code == 0 else "FAIL"
        return "[{marker}] {name:<16} {summary}".format(
            marker=marker, name=self.name, summary=self.summary,
        )


def _run_hardcoding_gate() -> GateResult:
    """Run the D1 hardcoding critical gate.

    Reuses :mod:`scripts.check_hardcoding` with ``--ci`` so the
    rules / whitelist / path all come from
    ``[tool.torcha-verse.hardcoding]`` in ``pyproject.toml``.
    """
    from check_hardcoding import main as hc_main
    exit_code = hc_main(["--ci"])
    if exit_code == 0:
        return GateResult("hardcoding", 0, "0 critical violations")
    if exit_code == 1:
        return GateResult(
            "hardcoding", 1,
            "critical violations remain (run "
            "scripts/check_hardcoding.py --severity critical "
            "for the full list)",
        )
    return GateResult("hardcoding", exit_code, "scanner usage error")


def _run_placeholder_gate() -> GateResult:
    """Run the D3 placeholder-registry gate."""
    from check_placeholders import main as ph_main
    exit_code = ph_main([])
    if exit_code == 0:
        return GateResult("placeholders", 0, "0 unregistered placeholders")
    if exit_code == 1:
        return GateResult(
            "placeholders", 1,
            "unregistered placeholders found (run "
            "scripts/check_placeholders.py for the full list)",
        )
    return GateResult("placeholders", exit_code, "scanner usage error")


def _run_degrade_logging_gate() -> GateResult:
    """Run the D3 stage three degrade-logging gate.

    The gate invokes :mod:`scripts.check_degrade_logging` against
    the project root.  The scanner exits ``0`` when no silent
    degrade is found, ``1`` when at least one site fails the
    forensic-trace rule.  We re-wrap that into a :class:`GateResult`
    so the unified runner can aggregate.
    """
    from check_degrade_logging import main as dl_main
    exit_code = dl_main([])
    if exit_code == 0:
        return GateResult("degrade_logging", 0, "no silent-degrade blocks")
    if exit_code == 1:
        return GateResult(
            "degrade_logging", 1,
            "silent-degrade blocks remain (run "
            "scripts/check_degrade_logging.py for the full list)",
        )
    return GateResult("degrade_logging", exit_code, "scanner usage error")


GATE_REGISTRY: Dict[str, Any] = {
    "hardcoding": {
        "runner": _run_hardcoding_gate,
        "default_enabled": True,
        "description": "D1 hardcoding critical violations",
    },
    "placeholders": {
        "runner": _run_placeholder_gate,
        "default_enabled": True,
        "description": "D3 unregistered pass/NotImplementedError",
    },
    "degrade_logging": {
        "runner": _run_degrade_logging_gate,
        "default_enabled": False,
        "description": (
            "D3 stage three silent-degrade blocks must contain "
            "logger.warning / safe_call / record_degrade / raise"
        ),
    },
}


# ---------------------------------------------------------------------------
# pyproject.toml loader
# ---------------------------------------------------------------------------
def _read_gate_config() -> Dict[str, bool]:
    """Read ``[tool.torcha-verse.ci-gates.<name>]`` from ``pyproject.toml``.

    Each gate has a section like::

        [tool.torcha-verse.ci-gates.hardcoding]
        enabled = false

    Returns a ``{gate_name: enabled_bool}`` mapping; missing gates
    fall back to :data:`GATE_REGISTRY`'s ``default_enabled``.
    """
    pyproject = _find_pyproject(Path.cwd())
    if pyproject is None:
        return {name: spec["default_enabled"] for name, spec in GATE_REGISTRY.items()}

    text = pyproject.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, bool] = {}
    for gate_name in GATE_REGISTRY:
        raw = _parse_mini_toml(
            text, "tool.torcha-verse.ci-gates.{}".format(gate_name),
        )
        if "enabled" in raw:
            low = raw["enabled"].strip().lower()
            out[gate_name] = low in ("true", "1", "yes")
        else:
            out[gate_name] = GATE_REGISTRY[gate_name]["default_enabled"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_ci_gates",
        description=(
            "Run all enabled CI gates declared in "
            "[tool.torcha-verse.ci-gates.*] of pyproject.toml."
        ),
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=None,
        choices=list(GATE_REGISTRY.keys()),
        help=(
            "Run only the named gate (can be passed multiple "
            "times).  Defaults to every enabled gate."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available gates and their enabled status, then exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the requested gates and return the aggregate exit code.

    Returns:
        ``0`` if every requested gate passed, ``1`` if any gate
        failed, ``2`` on usage error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        config = _read_gate_config()
        sys.stdout.write("CI gates (D1 stage three, v0.4.x):\n")
        for name, spec in GATE_REGISTRY.items():
            status = "enabled" if config.get(name, spec["default_enabled"]) else "disabled"
            sys.stdout.write(
                "  - {name:<16} [{status:<9}]  {description}\n".format(
                    name=name, status=status, description=spec["description"],
                )
            )
        return 0

    config = _read_gate_config()
    selected = args.gate or [
        name for name, enabled in config.items() if enabled
    ]
    if not selected:
        sys.stdout.write("No gates enabled -- nothing to do.\n")
        return 0

    sys.stdout.write(
        "Running {} gate(s): {}\n".format(
            len(selected), ", ".join(selected),
        )
    )
    results: List[GateResult] = []
    for name in selected:
        if name not in GATE_REGISTRY:
            sys.stderr.write("Error: unknown gate {!r}\n".format(name))
            return 2
        result = GATE_REGISTRY[name]["runner"]()
        results.append(result)
        sys.stdout.write(result.as_line() + "\n")

    if any(r.exit_code != 0 for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
