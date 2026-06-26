"""Configuration constants for the hardcoding scanner (v0.6.x).

Module-level constants are kept here (rather than inside function
bodies) so they don't get flagged by the scanner itself -- the
D1 hardcoding convention explicitly enumerates "module-level
constants" as a non-violation pattern.  Putting them in their own
file also keeps :mod:`scripts.check.hardcoding` under the soft
500-line cap.
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, Pattern, Tuple

__all__ = [
    "STRING_MIN_LENGTH",
    "LIST_MAX_ELEMENTS",
    "STRING_LITERAL",
    "NUMERIC_LITERAL",
    "PATH_LITERAL",
    "LIST_LITERAL",
    "SEVERITY_CRITICAL",
    "SEVERITY_WARN",
    "SEVERITY_INFO",
    "SEVERITY_ORDER",
    "WILDCARD_TYPES",
    "LOG_METHODS",
    "IMPORT_CALLS",
    "EXEMPT_NUMBERS",
    "RUNTIME_FUNCS",
    "PATH_ATTRS",
    "STRUCTURAL_PACKAGES",
    "STRUCTURAL_MIN",
    "STRUCTURAL_MAX",
    "EXCLUDE_DIRS",
    "CONTENT_LIMIT",
    "PATH_RE",
]


#: Minimum string length that triggers rule #1.
STRING_MIN_LENGTH: int = 10

#: Maximum number of list elements allowed before rule #4 triggers.
LIST_MAX_ELEMENTS: int = 3

#: Violation type identifiers.
STRING_LITERAL: str = "string_literal"
NUMERIC_LITERAL: str = "numeric_literal"
PATH_LITERAL: str = "path_literal"
LIST_LITERAL: str = "list_literal"

#: Severity levels.
SEVERITY_CRITICAL: str = "critical"
SEVERITY_WARN: str = "warn"
SEVERITY_INFO: str = "info"
SEVERITY_ORDER: Tuple[str, ...] = (SEVERITY_CRITICAL, SEVERITY_WARN, SEVERITY_INFO)

#: Wildcard tokens accepted in the whitelist ``type`` field.
WILDCARD_TYPES: FrozenSet[str] = frozenset({"*", "all"})

#: Logging method names whose string arguments are exempt from rule #1.
LOG_METHODS: FrozenSet[str] = frozenset({
    "debug", "info", "warning", "warn", "error", "critical",
    "exception", "log", "fatal",
})

#: Call names whose string arguments are import-related (exempt rule #1).
IMPORT_CALLS: FrozenSet[str] = frozenset({"import_module", "__import__"})

#: Numeric literals exempt from rule #2 (0, 1, -1 and their float forms).
EXEMPT_NUMBERS: FrozenSet[Any] = frozenset({0, 1, -1, 0.0, 1.0, -1.0})

#: Module-level functions whose call result is "reading from runtime config".
RUNTIME_FUNCS: FrozenSet[str] = frozenset({
    "getenv", "environ", "expanduser", "expandvars", "getenv",
    "getattr", "argv", "get",  # os.environ.get(...), dict.get(...)
})

#: Path-like attribute accessors that signal "this string is just a path
#: prefix, not a hardcoded value".
PATH_ATTRS: FrozenSet[str] = frozenset({
    "expanduser", "expandvars", "resolve", "absolute", "parent", "joinpath",
})

#: Top-level package prefixes that the ``is_structural_init`` heuristic
#: considers to be "model definitions" -- numeric literals in their
#: ``__init__`` methods are tagged as ``info`` (structural) instead of
#: ``critical`` (runtime config).
STRUCTURAL_PACKAGES: Tuple[str, ...] = ("models/",)

#: Numeric range for the ``is_structural_init`` heuristic.  Values
#: outside this range are *not* considered structural (e.g. ``1e6`` is
#: almost always a config knob like ``max_seq_len``).
STRUCTURAL_MIN: int = 2
STRUCTURAL_MAX: int = 10000

#: Directory names that are never scanned.
EXCLUDE_DIRS: FrozenSet[str] = frozenset({
    "config", "__pycache__", "build", "dist", "node_modules",
    ".git", ".venv", ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
})

#: Maximum length of violation content before truncation.
CONTENT_LIMIT: int = 120

#: Heuristic regular expression for path-like strings.  A string is
#: considered path-like when it contains ``/`` or ``\\`` *and* matches one
#: of the strong indicators below.  The generic "two segments separated by a
#: backslash" case is intentionally avoided so that regular-expression
#: escapes such as ``[^a-z0-9\\s]`` are not misclassified as paths.
PATH_RE: Pattern[str] = re.compile(
    r"(?:^[/~.])"                     # absolute / home / relative-dot prefix
    r"|(?:[A-Za-z]:[\\/])"            # Windows drive letter
    r"|(?:[\w.\-]+/[\w.\-]+/[\w.\-]*)"  # 2+ forward-slash path segments
    r"|(?:[\w.\-]+/[\w.\-]+\.\w{1,8})"  # forward slash + file extension
    r"|(?:[\w.\-]+\\[\w.\-]+\.\w{1,8})"  # backslash + file extension
)
