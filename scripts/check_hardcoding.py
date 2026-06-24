"""Hardcoding scanner for the TorchaVerse framework.

Scans Python source files (excluding the ``config/`` directory) for
patterns that indicate *hard-coded* values which should instead come
from configuration files or the :class:`~core.module_bus.ModuleBus`
registry.  Analysis is performed with the standard library
:mod:`ast` module -- no third-party parsing libraries are required.

Detected patterns
-----------------
1. ``string_literal``
        String literals longer than 10 characters that appear inside a
        function body.  Docstrings, ``import``-related call arguments,
        ``__all__`` entries and arguments to *logging* calls are
        excluded.

2. ``numeric_literal``
        Numeric literals that appear inside an ``__init__`` constructor.
        The values ``0``, ``1``, ``-1`` (and ``True``/``False``/``None``)
        are excluded as they are commonly used for initialisation.

3. ``path_literal``
        String literals that look like filesystem paths (contain ``/``
        or ``\\`` and resemble a path).

4. ``list_literal``
        List literals with more than three elements that appear inside a
        function body.

Usage
-----
::

    python scripts/check_hardcoding.py --path . --format text
    python scripts/check_hardcoding.py --whitelist config/hardcoded_whitelist.yaml

Exit codes
---------
* ``0`` -- no violations found.
* ``1`` -- one or more violations found.
* ``2`` -- usage / configuration error.

The scanner always emits a report (even when violations are present) so
it can be wired into CI without masking the underlying issues.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "Violation",
    "Exemption",
    "scan_file",
    "scan_directory",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration constants (module-level so they are not "hard-coded" inside
# function bodies).
# ---------------------------------------------------------------------------
#: Minimum string length that triggers rule #1.
STRING_MIN_LENGTH: int = 10

#: Maximum number of list elements allowed before rule #4 triggers.
LIST_MAX_ELEMENTS: int = 3

#: Violation type identifiers.
STRING_LITERAL: str = "string_literal"
NUMERIC_LITERAL: str = "numeric_literal"
PATH_LITERAL: str = "path_literal"
LIST_LITERAL: str = "list_literal"

#: Wildcard tokens accepted in the whitelist ``type`` field.
_WILDCARD_TYPES: frozenset[str] = frozenset({"*", "all"})

#: Logging method names whose string arguments are exempt from rule #1.
_LOG_METHODS: frozenset[str] = frozenset({
    "debug", "info", "warning", "warn", "error", "critical",
    "exception", "log", "fatal",
})

#: Call names whose string arguments are import-related (exempt rule #1).
_IMPORT_CALLS: frozenset[str] = frozenset({"import_module", "__import__"})

#: Numeric literals exempt from rule #2 (0, 1, -1 and their float forms).
_EXEMPT_NUMBERS: frozenset[Any] = frozenset({0, 1, -1, 0.0, 1.0, -1.0})

#: Directory names that are never scanned.
_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "config", "__pycache__", "build", "dist", "node_modules",
    ".git", ".venv", ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
})

#: Maximum length of violation content before truncation.
_CONTENT_LIMIT: int = 120

#: Heuristic regular expression for path-like strings.  A string is
#: considered path-like when it contains ``/`` or ``\\`` *and* matches one
#: of the strong indicators below.  The generic "two segments separated by a
#: backslash" case is intentionally avoided so that regular-expression
#: escapes such as ``[^a-z0-9\\s]`` are not misclassified as paths.
_PATH_RE: re.Pattern[str] = re.compile(
    r"(?:^[/~.])"                     # absolute / home / relative-dot prefix
    r"|(?:[A-Za-z]:[\\/])"            # Windows drive letter
    r"|(?:[\w.\-]+/[\w.\-]+/[\w.\-]*)"  # 2+ forward-slash path segments
    r"|(?:[\w.\-]+/[\w.\-]+\.\w{1,8})"  # forward slash + file extension
    r"|(?:[\w.\-]+\\[\w.\-]+\.\w{1,8})"  # backslash + file extension
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Violation:
    """A single hardcoding violation found by the scanner.

    Attributes:
        file: Path of the offending file, relative to the scan root.
        line: 1-based line number where the violation occurs.
        col: 0-based column offset of the offending node.
        type: Violation type identifier (one of the ``*_LITERAL``
            constants).
        content: A short textual representation of the offending value.
    """

    file: str
    line: int
    col: int
    type: str
    content: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "file": self.file,
            "line": self.line,
            "column": self.col,
            "type": self.type,
            "content": self.content,
        }


@dataclass
class Exemption:
    """A single whitelist exemption.

    A violation is exempt when *all* specified fields match.  Omitted
    (``None``) fields match anything.

    Attributes:
        file: glob pattern matched against the violation's relative
            path (e.g. ``"core/*.py"``).
        type: Violation type to exempt, or ``"*"``/``"all"`` for any.
        line: Exact line number to exempt, or ``None`` for any line.
        content_contains: Substring that must appear in the violation
            content, or ``None`` to ignore content.
    """

    file: str
    type: str = "*"
    line: Optional[int] = None
    content_contains: Optional[str] = None

    def matches(self, violation: Violation) -> bool:
        """Return ``True`` if this exemption covers ``violation``."""
        if not fnmatch.fnmatch(violation.file, self.file):
            return False
        if self.type not in _WILDCARD_TYPES and violation.type != self.type:
            return False
        if self.line is not None and violation.line != self.line:
            return False
        if self.content_contains is not None and self.content_contains not in violation.content:
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = _CONTENT_LIMIT) -> str:
    """Truncate ``text`` to ``limit`` characters, appending an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _looks_like_path(value: str) -> bool:
    """Return ``True`` if ``value`` resembles a filesystem path.

    Single-character separators (``"/"``, ``"\\"``, ``"."``, ``"~"``) are
    intentionally excluded to avoid flagging common idiom literals such
    as ``"/" not in value``.
    """
    if len(value) < 2:
        return False
    if "/" not in value and "\\" not in value:
        return False
    return bool(_PATH_RE.search(value))


def _is_all_name(node: ast.AST) -> bool:
    """Return ``True`` if ``node`` is the name ``__all__``."""
    return isinstance(node, ast.Name) and node.id == "__all__"


def _collect_str_ids(node: Optional[ast.AST], sink: set[int]) -> None:
    """Add the ids of all string ``ast.Constant`` nodes under ``node``."""
    if node is None:
        return
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            sink.add(id(sub))


def _is_logging_call(node: ast.Call) -> bool:
    """Return ``True`` if ``node`` looks like a logging call."""
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS


def _is_import_call(node: ast.Call) -> bool:
    """Return ``True`` if ``node`` looks like an import-related call."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in _IMPORT_CALLS:
        return True
    return isinstance(func, ast.Attribute) and func.attr in _IMPORT_CALLS


def _collect_exclusions(tree: ast.AST) -> tuple[set[int], set[int]]:
    """Pre-compute the ids of string constants exempt from rule #1.

    Returns a tuple ``(docstring_ids, excluded_str_ids)`` where the
    latter covers docstrings, ``__all__`` entries, logging-call
    arguments and import-call arguments.
    """
    docstring_ids: set[int] = set()
    excluded_str_ids: set[int] = set()

    # Docstrings: the first statement of a module/class/function when it
    # is a bare string expression.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    docstring_ids.add(id(first.value))
                    excluded_str_ids.add(id(first.value))

    # __all__ assignment string values.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(_is_all_name(tgt) for tgt in node.targets):
                _collect_str_ids(node.value, excluded_str_ids)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _is_all_name(node.target):
                _collect_str_ids(node.value, excluded_str_ids)

    # Logging-call and import-call string arguments.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (_is_logging_call(node) or _is_import_call(node)):
            for arg in node.args:
                _collect_str_ids(arg, excluded_str_ids)
            for kw in node.keywords:
                _collect_str_ids(kw.value, excluded_str_ids)

    return docstring_ids, excluded_str_ids


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------
class _HardcodingVisitor(ast.NodeVisitor):
    """Walks an AST tree collecting hardcoding violations."""

    def __init__(
        self,
        relpath: str,
        docstring_ids: set[int],
        excluded_str_ids: set[int],
    ) -> None:
        self.relpath: str = relpath
        self._docstring_ids: set[int] = docstring_ids
        self._excluded_str_ids: set[int] = excluded_str_ids
        self.violations: list[Violation] = []
        # Stack of (name, kind) for each function-like scope entered.
        self._func_stack: list[tuple[str, str]] = []

    @property
    def in_function(self) -> bool:
        """``True`` when inside any function/lambda body."""
        return bool(self._func_stack)

    @property
    def in_init(self) -> bool:
        """``True`` when inside an ``__init__`` method (any depth)."""
        return any(
            name == "__init__"
            for name, kind in self._func_stack
            if kind == "func"
        )

    # -- scope management ------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append((node.name, "func"))
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._func_stack.append(("<lambda>", "lambda"))
        self.generic_visit(node)
        self._func_stack.pop()

    # -- literal checks --------------------------------------------------
    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, str):
            self._check_string(node, value)
        elif isinstance(value, bool) or value is None:
            pass  # booleans / None are never numeric violations
        elif isinstance(value, (int, float, complex)):
            self._check_numeric(node, value)
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        if self.in_function and len(node.elts) > LIST_MAX_ELEMENTS:
            self._add(node, LIST_LITERAL, "[{} elements]".format(len(node.elts)))
        self.generic_visit(node)

    # -- rule implementations -------------------------------------------
    def _check_string(self, node: ast.Constant, value: str) -> None:
        is_exempt = (
            id(node) in self._docstring_ids
            or id(node) in self._excluded_str_ids
        )
        # Rule #1: long string literal inside a function body.
        if (
            self.in_function
            and len(value) > STRING_MIN_LENGTH
            and not is_exempt
        ):
            self._add(node, STRING_LITERAL, _truncate(repr(value)))
        # Rule #3: path-like string literal (checked everywhere).
        if not is_exempt and _looks_like_path(value):
            self._add(node, PATH_LITERAL, _truncate(repr(value)))

    def _check_numeric(self, node: ast.Constant, value: Any) -> None:
        # Rule #2: numeric literal inside __init__ (0, 1, -1 excluded).
        if self.in_init and value not in _EXEMPT_NUMBERS:
            self._add(node, NUMERIC_LITERAL, repr(value))

    # -- bookkeeping -----------------------------------------------------
    def _add(self, node: ast.AST, vtype: str, content: str) -> None:
        self.violations.append(
            Violation(
                file=self.relpath,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                type=vtype,
                content=content,
            )
        )


# ---------------------------------------------------------------------------
# File / directory scanning
# ---------------------------------------------------------------------------
def _relative_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX-style string."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_excluded(rel_parts: tuple[str, ...]) -> bool:
    """Return ``True`` if any path component marks an excluded directory."""
    for part in rel_parts[:-1]:  # exclude dirs only, not the file name
        if part in _EXCLUDE_DIRS or part.startswith("."):
            return True
    return False


def scan_file(path: Path, root: Path) -> list[Violation]:
    """Scan a single Python file and return its violations.

    Files that cannot be read or parsed are silently skipped.

    Args:
        path: Absolute path to the ``.py`` file.
        root: The scan root used to compute the relative path.

    Returns:
        A list of :class:`Violation` objects (possibly empty).
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    relpath = _relative_path(path, root)
    docstring_ids, excluded_str_ids = _collect_exclusions(tree)
    visitor = _HardcodingVisitor(relpath, docstring_ids, excluded_str_ids)
    visitor.visit(tree)
    return visitor.violations


def _iter_python_files(root: Path) -> Any:
    """Yield ``.py`` files under ``root`` that are not excluded."""
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if _is_excluded(rel_parts):
            continue
        yield path


def scan_directory(
    root: Path,
    exemptions: Optional[list[Exemption]] = None,
) -> list[Violation]:
    """Scan every non-excluded ``.py`` file under ``root``.

    Args:
        root: Directory (or single file) to scan.
        exemptions: Optional list of :class:`Exemption` objects; matching
            violations are dropped from the result.

    Returns:
        A sorted list of :class:`Violation` objects.
    """
    exemptions = exemptions or []
    violations: list[Violation] = []

    if root.is_file():
        files = [root]
    else:
        files = list(_iter_python_files(root))

    for path in files:
        for violation in scan_file(path, root if root.is_dir() else root.parent):
            if not any(exemption.matches(violation) for exemption in exemptions):
                violations.append(violation)

    violations.sort(key=lambda v: (v.file, v.line, v.col, v.type))
    return violations


# ---------------------------------------------------------------------------
# Whitelist loading
# ---------------------------------------------------------------------------
def load_whitelist(path: Path) -> list[Exemption]:
    """Load exemption entries from a YAML whitelist file.

    The expected schema is::

        exemptions:
          - file: "core/*.py"
            type: "string_literal"   # or "*" / "all"
            line: 42                  # optional
            content_contains: "..."   # optional

    Args:
        path: Path to the YAML whitelist file.

    Returns:
        A list of :class:`Exemption` objects.

    Raises:
        SystemExit: If PyYAML is not installed or the file is invalid.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read --whitelist ({}).".format(exc)
        ) from exc

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise SystemExit("Cannot read whitelist {}: {}".format(path, exc)) from exc
    except yaml.YAMLError as exc:
        raise SystemExit("Invalid YAML in {}: {}".format(path, exc)) from exc

    if not isinstance(data, dict):
        raise SystemExit("Whitelist {} must be a YAML mapping.".format(path))

    entries = data.get("exemptions", []) or []
    if not isinstance(entries, list):
        raise SystemExit("'exemptions' in {} must be a list.".format(path))

    exemptions: list[Exemption] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(
                "Exemption #{} in {} must be a mapping.".format(index, path)
            )
        file = entry.get("file")
        if not file:
            raise SystemExit(
                "Exemption #{} in {} is missing required 'file'.".format(index, path)
            )
        exemptions.append(
            Exemption(
                file=str(file),
                type=str(entry.get("type", "*")),
                line=entry.get("line"),
                content_contains=entry.get("content_contains"),
            )
        )
    return exemptions


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _count_by_type(violations: list[Violation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.type] = counts.get(violation.type, 0) + 1
    return counts


def format_text(violations: list[Violation]) -> str:
    """Render violations as human-readable text."""
    lines: list[str] = []
    if violations:
        lines.append("Hardcoding violations ({}):".format(len(violations)))
        lines.append("-" * 60)
        for violation in violations:
            lines.append(
                "{file}:{line}:{col}: {type}: {content}".format(
                    file=violation.file,
                    line=violation.line,
                    col=violation.col,
                    type=violation.type,
                    content=violation.content,
                )
            )
    else:
        lines.append("No hardcoding violations found.")
    lines.append("-" * 60)
    summary = ", ".join(
        "{}={}".format(kind, count)
        for kind, count in sorted(_count_by_type(violations).items())
    )
    lines.append("Summary: {} violation(s){}".format(
        len(violations),
        " ({})".format(summary) if summary else "",
    ))
    return "\n".join(lines)


def format_json(violations: list[Violation]) -> str:
    """Render violations as a JSON document."""
    payload = {
        "count": len(violations),
        "summary": _count_by_type(violations),
        "violations": [violation.as_dict() for violation in violations],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_hardcoding",
        description="Scan Python sources for hard-coded values.",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Directory (or file) to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--whitelist",
        default=None,
        help="Optional YAML file listing exemptions.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. Defaults to 'text'.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the hardcoding scanner.

    Args:
        argv: Optional command-line arguments (defaults to ``sys.argv``).

    Returns:
        ``0`` if no violations, ``1`` if violations found, ``2`` on
        usage error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        sys.stderr.write("Error: path does not exist: {}\n".format(root))
        return 2

    exemptions: list[Exemption] = []
    if args.whitelist:
        whitelist_path = Path(args.whitelist).expanduser().resolve()
        if not whitelist_path.exists():
            sys.stderr.write("Error: whitelist not found: {}\n".format(whitelist_path))
            return 2
        exemptions = load_whitelist(whitelist_path)

    violations = scan_directory(root, exemptions)

    if args.format == "json":
        sys.stdout.write(format_json(violations) + "\n")
    else:
        sys.stdout.write(format_text(violations) + "\n")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
