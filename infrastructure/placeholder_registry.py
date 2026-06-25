"""Placeholder registry: parse ``docs/placeholder_registry.md`` + scan source.

This module is the *machine-readable* companion to
:doc:`/docs/placeholder_registry`.  It exposes:

* :class:`PlaceholderCategory` -- the five canonical categories.
* :class:`PlaceholderEntry` -- a single registered placeholder
  (file path, line, category, description).
* :func:`load_registry` -- parse the markdown registry into a list of
  :class:`PlaceholderEntry`.
* :func:`scan_source` -- scan a single file or directory for
  ``pass`` statements and ``raise NotImplementedError`` occurrences,
  ignoring tests / caches / virtualenvs.
* :func:`find_unregistered` -- return scanner hits that are NOT
  present in the registry, suitable for CI gating.
* :class:`PlaceholderScannerError` -- raised on malformed registry
  entries.

The module is intentionally dependency-free (only :mod:`re`,
:mod:`pathlib`, :mod:`dataclasses`) so it can be reused by the
``scripts/check_placeholders.py`` linter, by tests, and by ad-hoc
developer scripts without pulling in the rest of TorchaVerse.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- registry + scanner (this module).
* L4 ``nodes`` / ``scripts`` -- use :func:`find_unregistered` for
  CI gating.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "PlaceholderCategory",
    "PlaceholderEntry",
    "PlaceholderScannerError",
    "load_registry",
    "scan_source",
    "find_unregistered",
    "DEFAULT_REGISTRY_PATH",
    "SCAN_IGNORE_DIRS",
]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
#: Default location of the human-authored registry.  Relative paths in the
#: registry are resolved against the project root.
DEFAULT_REGISTRY_PATH: str = "docs/placeholder_registry.md"

#: Subdirectories to skip when scanning.  These never contain production
#: placeholders and would otherwise drown the scanner output.
SCAN_IGNORE_DIRS: Tuple[str, ...] = (
    "tests",  # tests have their own conventions
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
class PlaceholderCategory(str, Enum):
    """The five canonical placeholder categories.

    Members:

        PROTOCOL: Protocol / abstract-method ``NotImplementedError``;
            subclasses must implement.
        TP_PP: Tensor / pipeline parallelism placeholder under
            :mod:`infrastructure.device_manager`, wrapped by
            :func:`safe_call` so single-GPU envs do not crash.
        PROTOCOL_STUB: Protocol class method stub (no body, no
            ``NotImplementedError`` either -- rare).
        DEGRADE_TRY_EXCEPT: ``pass`` inside ``except`` blocks for
            resource cleanup / graceful-degradation paths.
        DEGRADE_NOOP: ``pass`` inside ``if``/``elif`` branches for
            explicit no-op cases.
    """

    PROTOCOL = "protocol"
    TP_PP = "tp_pp"
    PROTOCOL_STUB = "protocol_stub"
    DEGRADE_TRY_EXCEPT = "degrade_try_except"
    DEGRADE_NOOP = "degrade_noop"


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaceholderEntry:
    """A single registered placeholder, parsed from the markdown table.

    Attributes:
        file: Path relative to the project root, e.g.
            ``"infrastructure/device_manager.py"``.
        line: 1-based line number where the placeholder lives.
        category: One of the :class:`PlaceholderCategory` values.
        description: Short human-readable rationale (extracted from
            the registry table's "理由" / "说明" column).
    """

    file: str
    line: int
    category: PlaceholderCategory
    description: str = ""

    def matches(self, file: str, line: int) -> bool:
        """Return ``True`` if this entry matches the given ``file:line``."""
        return self.file == file and self.line == line


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PlaceholderScannerError(ValueError):
    """Raised when the registry markdown cannot be parsed."""


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------
# Match a single row in a registry table:
#   `| 1 | \`path/to/file.py:42\` | \`Class\` | \`method\` | description |`
# We require at least 5 columns; the first column is the index (number),
# the second is `path:line` (the only column we extract strictly).
_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<idx>\d+)\s*\|"          # index
    r"\s*`?(?P<file>[^`|:]+):(?P<line>\d+)`?\s*\|"  # file:line (backticks optional)
    r"(?P<rest>.*?)\s*\|?\s*$",         # remaining columns, optional trailing |
    re.MULTILINE,
)

# Map a description column header to a category.  If a row's
# sub-section heading (preceding the table) mentions a different keyword
# we pick the most specific match.  Order matters: most specific first.
_HEADING_CATEGORY_HINTS: Sequence[Tuple[str, PlaceholderCategory]] = (
    ("协议/抽象方法", PlaceholderCategory.PROTOCOL),
    ("TP/PP placeholder", PlaceholderCategory.TP_PP),
    ("Protocol stub", PlaceholderCategory.PROTOCOL_STUB),
    ("try/except 兜底", PlaceholderCategory.DEGRADE_TRY_EXCEPT),
    ("if-branch noop", PlaceholderCategory.DEGRADE_NOOP),
    ("degrade_try_except", PlaceholderCategory.DEGRADE_TRY_EXCEPT),
    ("degrade_noop", PlaceholderCategory.DEGRADE_NOOP),
    ("protocol", PlaceholderCategory.PROTOCOL),
    ("tp_pp", PlaceholderCategory.TP_PP),
    ("protocol_stub", PlaceholderCategory.PROTOCOL_STUB),
)


def _category_for_heading(heading: str) -> PlaceholderCategory:
    """Pick a category based on a sub-section heading text."""
    h = heading.strip()
    for needle, cat in _HEADING_CATEGORY_HINTS:
        if needle in h:
            return cat
    # Default fallback -- matches the most common pattern in our registry.
    return PlaceholderCategory.DEGRADE_TRY_EXCEPT


def load_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    project_root: Optional[str | Path] = None,
) -> List[PlaceholderEntry]:
    """Parse the markdown registry into :class:`PlaceholderEntry` records.

    Args:
        registry_path: Path to the markdown file.  Relative paths are
            resolved against ``project_root`` (or CWD when ``None``).
        project_root: Project root used to resolve ``registry_path`` and
            to verify that the listed ``file:line`` entries exist.

    Returns:
        A list of :class:`PlaceholderEntry`, in the order they appear
        in the registry.

    Raises:
        FileNotFoundError: If ``registry_path`` does not exist.
        PlaceholderScannerError: If a row is malformed (bad line number,
            missing fields, etc.).
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    reg_file = Path(registry_path)
    if not reg_file.is_absolute():
        reg_file = (root / reg_file).resolve()
    if not reg_file.is_file():
        raise FileNotFoundError("registry not found: {}".format(reg_file))

    text = reg_file.read_text(encoding="utf-8")
    entries: List[PlaceholderEntry] = []
    current_heading: str = ""
    # The file uses a few different table layouts (some with 5 cols, some
    # with more, and at least one with no leading "idx" column).  We walk
    # the file line-by-line and rebuild the heading context.
    rows: List[Tuple[str, str]] = []  # (heading, body)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            current_heading = stripped.lstrip("#").strip()
            continue
        if not stripped.startswith("|"):
            continue
        # Skip table header / separator lines.
        # A markdown table separator is mostly dashes (e.g. `|---|---|---|`),
        # so match that pattern precisely.  Only treat a row as a header
        # when it contains all three column labels ("文件", "行", "类别")
        # together -- NOT when it merely contains "文件" as part of a
        # description (e.g. "按文件格式覆盖").
        if re.match(r"^\|[\s\-|:]+\|?$", stripped):
            continue
        if "文件" in stripped and "行" in stripped and "类别" in stripped:
            continue
        rows.append((current_heading, line))

    for heading, raw in rows:
        m = _TABLE_ROW_RE.match(raw)
        if not m:
            # Some legitimate tables may not match the strict regex
            # (e.g. sub-section summary rows like "| 合计 | 43 | ... |").
            # Skip silently rather than crash; the scanner will catch any
            # *real* unregistered placeholder at scan time.
            continue
        try:
            line_no = int(m.group("line"))
        except ValueError as exc:
            raise PlaceholderScannerError(
                "bad line number in row: {!r}".format(raw)
            ) from exc
        file_rel = m.group("file").strip().strip("`")
        category = _category_for_heading(heading)
        # Description: take the last pipe-separated cell that is not a
        # short identifier (class / method / reason).
        rest = m.group("rest")
        cells = [c.strip() for c in rest.split("|") if c.strip()]
        desc = cells[-1] if cells else ""
        entries.append(
            PlaceholderEntry(
                file=file_rel,
                line=line_no,
                category=category,
                description=desc,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Source scanner
# ---------------------------------------------------------------------------
# A line that contains only `pass` (optionally indented) -- matches
# ``    pass`` or ``pass`` standalone.  Must NOT match ``password`` etc.
_PASS_LINE_RE = re.compile(r"^\s*pass\s*(?:#.*)?$")
# A line containing ``raise NotImplementedError`` (with or without msg).  # placeholder-registry: ignore
_RAISE_NIE_RE = re.compile(r"\braise\s+NotImplementedError\b")
# Inline ignore marker: ``# placeholder-registry: ignore``.  When a
# line ends with this comment the scanner treats the hit as opt-out
# (e.g. when a docstring describes the rule itself).
_INLINE_IGNORE_RE = re.compile(
    r"#\s*placeholder-registry\s*:\s*ignore\b",
    re.IGNORECASE,
)


def _is_code_file(p: Path) -> bool:
    """Return ``True`` if ``p`` should be scanned for placeholders."""
    return p.suffix == ".py"


def _should_skip_dir(name: str) -> bool:
    """Return ``True`` if directory ``name`` should be skipped."""
    return name in SCAN_IGNORE_DIRS or name.startswith(".")


@dataclass(frozen=True)
class ScanHit:
    """A single scanner hit (pass or NotImplementedError)."""

    file: str  # relative path
    line: int  # 1-based
    kind: str  # "pass" or "NotImplementedError"
    text: str  # the matched line (stripped)


def _is_docstring_text(line: str) -> bool:
    """Return ``True`` if ``line`` looks like a docstring text that quotes
    a placeholder keyword inside backticks (e.g. ``"``pass`` statements"``).

    The scanner should ignore such descriptions -- they mention the
    keyword as a literal example, not as a real placeholder.
    """
    return "`pass`" in line or "`NotImplementedError`" in line


def _scan_file(path: Path, project_root: Path) -> Iterator[ScanHit]:
    """Yield :class:`ScanHit` for a single Python file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    rel = path.resolve().relative_to(project_root.resolve())
    for i, line in enumerate(text.splitlines(), start=1):
        # Honour inline opt-out markers and quoted docstring text.
        if _INLINE_IGNORE_RE.search(line):
            continue
        if _is_docstring_text(line):
            continue
        if _PASS_LINE_RE.match(line):
            yield ScanHit(
                file=str(rel), line=i, kind="pass", text=line.strip(),
            )
        elif _RAISE_NIE_RE.search(line):
            yield ScanHit(
                file=str(rel), line=i, kind="NotImplementedError",
                text=line.strip(),
            )


def scan_source(
    target: str | Path,
    *,
    project_root: Optional[str | Path] = None,
) -> List[ScanHit]:
    """Scan a single file or directory tree for placeholder candidates.

    Args:
        target: A file or directory to scan.  Relative paths resolve
            against ``project_root`` (or CWD when ``None``).
        project_root: Project root used to compute relative paths in
            the returned hits.

    Returns:
        A list of :class:`ScanHit`, in (file, line) order.  Tests,
        caches, and virtualenvs are skipped.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    target = Path(target)
    if not target.is_absolute():
        target = (root / target).resolve()
    if not target.exists():
        return []
    if target.is_file():
        if not _is_code_file(target):
            return []
        return list(_scan_file(target, root))
    hits: List[ScanHit] = []
    for path in sorted(target.rglob("*.py")):
        # Skip ignored directories by checking each ancestor.
        rel_parts = path.relative_to(target).parts
        if any(_should_skip_dir(part) for part in rel_parts):
            continue
        # Also skip if the path itself is inside an ignored subdirectory
        # anywhere up the tree (covers symlinked / nested cases).
        if any(part in SCAN_IGNORE_DIRS for part in path.parts):
            continue
        hits.extend(_scan_file(path, root))
    return hits


# ---------------------------------------------------------------------------
# Diff: scan - registry
# ---------------------------------------------------------------------------
def find_unregistered(
    hits: Iterable[ScanHit],
    registry: Sequence[PlaceholderEntry],
) -> List[ScanHit]:
    """Return scanner hits that are NOT present in ``registry``.

    Useful for CI gating -- if the list is non-empty, the developer
    added a placeholder that hasn't been documented yet.
    """
    reg_keys = {(e.file, e.line) for e in registry}
    return [h for h in hits if (h.file, h.line) not in reg_keys]


# ---------------------------------------------------------------------------
# Convenience: build a registry index by (file, line)
# ---------------------------------------------------------------------------
def registry_index(
    registry: Sequence[PlaceholderEntry],
) -> dict[Tuple[str, int], PlaceholderEntry]:
    """Build a ``(file, line) -> entry`` mapping for fast lookups."""
    return {(e.file, e.line): e for e in registry}
