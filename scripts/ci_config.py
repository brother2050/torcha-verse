"""CI configuration loader for hardcoding (D1 stage three, v0.4.x).

The hardcoding scanner is invoked by CI with ``--ci``.  In that
mode it expects to read its settings from
``[tool.torcha-verse.hardcoding]`` in ``pyproject.toml`` so the
CI pipeline has a single source of truth and operators do not
duplicate the same ``--path`` / ``--whitelist`` flags at every
caller site.

The schema is intentionally tiny: only the fields the scanner
actually reads are exposed.  Adding a new field is a 4-line
change; the scanner will simply ignore it.

Schema (see ``pyproject.toml`` ``[tool.torcha-verse.hardcoding]``)::

    [tool.torcha-verse.hardcoding]
    path = "."
    whitelist = "config/hardcoded_whitelist.yaml"
    ci_fail_on = "critical"            # one of: critical / warn / info
    enabled = true                      # set to false to skip the gate

This module is intentionally stdlib-only and a 50-line script:
it should not grow a dependency on ``tomllib`` (which is
stdlib since 3.11 but missing on older interpreters) or
``tomli`` (which the project explicitly avoids).  We parse the
small ``[tool.torcha-verse.hardcoding]`` block by hand, since
the schema is fixed and a real TOML parser would be overkill.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["load_hardcoding_ci_settings", "DEFAULT_CI_SETTINGS"]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CI_SETTINGS: Dict[str, Any] = {
    "path": ".",
    "whitelist": "config/hardcoded_whitelist.yaml",
    "ci_fail_on": "critical",
    "enabled": True,
}

_VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "warn", "info"})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
#: Pattern for ``key = value`` where ``value`` is a quoted string
#: or a bare word.  We only parse the values we need (strings +
#: booleans); this is not a general TOML parser.
_KV_PATTERN = re.compile(
    r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("([^"]*)"|([A-Za-z0-9_./\-]+))""",
)


def _parse_mini_toml(text: str, section: str) -> Dict[str, str]:
    """Extract a single ``[section]`` table from ``text``.

    Stops at the next ``[section]`` header or EOF.  Returns a
    flat ``{key: raw_string_value}`` mapping.  Boolean literals
    are not promoted (callers should call :func:`_coerce_value`).
    """
    inside = False
    out: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        # ``[table]`` header line.
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            inside = current == section
            continue
        if not inside:
            continue
        match = _KV_PATTERN.match(stripped)
        if not match:
            continue
        key = match.group(1)
        # Prefer the quoted form; fall back to the bare word.
        value = match.group(3) if match.group(3) is not None else match.group(4)
        out[key] = value
    return out


def _coerce_value(raw: str, name: str) -> Any:
    """Coerce ``raw`` to the right Python type for ``name``.

    Supported:

    * ``enabled`` (bool): ``"true"`` / ``"false"`` (case-insensitive).
    * ``ci_fail_on`` (str): one of ``critical``/``warn``/``info``.
    * ``path`` / ``whitelist`` (str): passed through.

    Anything else raises ``SystemExit(2)`` (the scanner's
    "usage error" exit code).
    """
    if name == "enabled":
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise SystemExit(
            "Error: [tool.torcha-verse.hardcoding].enabled must be "
            "true or false (got {!r})".format(raw)
        )
    if name == "ci_fail_on":
        if raw not in _VALID_SEVERITIES:
            raise SystemExit(
                "Error: [tool.torcha-verse.hardcoding].ci_fail_on "
                "must be one of {} (got {!r})".format(
                    sorted(_VALID_SEVERITIES), raw,
                )
            )
        return raw
    return raw


def load_hardcoding_ci_settings(
    pyproject_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load ``[tool.torcha-verse.hardcoding]`` from ``pyproject.toml``.

    Args:
        pyproject_path: Path to ``pyproject.toml``.  When ``None``,
            the function searches the current working directory and
            walks up the directory tree until it finds one (standard
            ``pyproject.toml`` discovery).

    Returns:
        A dict with the four schema keys; missing keys fall back to
        :data:`DEFAULT_CI_SETTINGS`.  The returned dict is a fresh
        copy -- callers may mutate it without affecting subsequent
        calls.
    """
    if pyproject_path is None:
        pyproject_path = _find_pyproject(Path.cwd())
    if pyproject_path is None or not pyproject_path.exists():
        # No pyproject.toml found -- fall back to the defaults
        # (this is the historical behaviour pre-D1 stage three).
        return dict(DEFAULT_CI_SETTINGS)

    text = pyproject_path.read_text(encoding="utf-8", errors="replace")
    raw = _parse_mini_toml(text, "tool.torcha-verse.hardcoding")
    merged = dict(DEFAULT_CI_SETTINGS)
    for key in ("path", "whitelist", "ci_fail_on", "enabled"):
        if key in raw:
            merged[key] = _coerce_value(raw[key], key)
    return merged


def _find_pyproject(start: Path) -> Optional[Path]:
    """Walk up from ``start`` until ``pyproject.toml`` is found.

    Stops at the filesystem root.  Returns ``None`` if the file
    is not found.
    """
    for candidate in (start, *start.parents):
        path = candidate / "pyproject.toml"
        if path.exists() and path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    settings = load_hardcoding_ci_settings()
    sys.stdout.write("Hardcoding CI settings (D1 stage three):\n")
    for key, value in sorted(settings.items()):
        sys.stdout.write("  {:<14} = {!r}\n".format(key, value))
