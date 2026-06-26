"""Scan entry points for the hardcoding scanner (v0.6.x).

The :func:`scan_file` and :func:`scan_directory` functions are
the public entry points used by the CLI and by tests.  The
implementation parses a Python source file with :mod:`ast`,
attaches ``.parent`` links to every node, then walks the tree
with a :class:`HardcodingVisitor` collecting violations.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Optional, Union

from scripts.check.hardcoding_rules import Rule, get_rule

from ._ast_helpers import attach_parents, collect_docstring_ids
from ._constants import EXCLUDE_DIRS
from ._types import Violation
from ._visitor import HardcodingVisitor

__all__ = ["scan_file", "scan_directory"]


def _looks_like_path_object(value: str) -> bool:
    """Heuristic: a string is a path candidate if it contains ``/``
    or ``\\`` or ends with a known path component.  Used to
    disambiguate the v0.4.x ``scan_file(path, root)`` signature
    where the second positional argument can be a relative path
    string or a :class:`Path` object.
    """
    if not value:
        return False
    return "/" in value or "\\" in value or value.startswith(".") or value.endswith((".py",))


def _resolve_rules(only_rule: Optional[str], rules: Optional[List[Rule]]) -> List[Rule]:
    """Resolve the rule list, applying ``only_rule`` if present.

    Raises ``ValueError`` when ``only_rule`` names a rule that does
    not exist in :data:`DEFAULT_RULES` (mirrors the v0.4.x
    ``scripts/check_hardcoding.py`` behaviour).
    """
    if only_rule is not None:
        rule = get_rule(only_rule)
        if rule is None:
            raise ValueError(f"unknown rule: {only_rule!r}")
        return [rule]
    if rules is not None:
        return list(rules)
    from scripts.check.hardcoding_rules import DEFAULT_RULES
    return list(DEFAULT_RULES)


def scan_file(
    path: Union[Path, str],
    relpath_or_root: Optional[Union[Path, str]] = None,
    rules: Optional[List[Rule]] = None,
    root: Optional[Path] = None,
    only_rule: Optional[str] = None,
) -> List[Violation]:
    """Scan a single Python source file and return its violations.

    The v0.4.x signature is ``scan_file(path, root)`` -- the
    second positional argument is the *scan root* (used to compute
    the relative file path stored on :attr:`Violation.file`).
    The v0.6.x signature adds explicit ``relpath=`` and ``root=``
    keyword arguments for callers that want to be unambiguous.

    Args:
        path: Absolute path to the file.
        relpath_or_root: Either the relative path (a plain string
            without path separators) *or* the scan root (a
            :class:`Path` object or a string with separators)
            used to compute the relative path when ``relpath`` is
            not given.
        rules: Optional list of :class:`Rule` instances.  When
            ``None`` the :data:`DEFAULT_RULES` are used.
        root: Optional scan root (overrides ``relpath_or_root``
            when both are given).
        only_rule: When provided, only the named rule is run.

    Returns:
        A list of :class:`Violation` instances, in source order.
    """
    p = Path(path)
    relpath: Optional[str] = None
    candidate_root: Optional[Path] = root
    if relpath_or_root is not None:
        if isinstance(relpath_or_root, Path):
            candidate_root = relpath_or_root
        elif isinstance(relpath_or_root, str):
            # Plain relative path (no separators) -> relpath.
            # Anything with separators or ``./`` prefix -> root.
            if _looks_like_path_object(relpath_or_root):
                candidate_root = Path(relpath_or_root)
            else:
                relpath = relpath_or_root
    if relpath is None:
        if candidate_root is not None:
            try:
                relpath = str(p.relative_to(candidate_root))
            except ValueError:
                relpath = p.name
        else:
            relpath = p.name
    resolved_rules = _resolve_rules(only_rule, rules)
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(p))
    except SyntaxError:
        return []
    attach_parents(tree)
    docstring_ids: set = set()
    excluded_str_ids: set = set()
    collect_docstring_ids(tree, docstring_ids, excluded_str_ids)
    visitor = HardcodingVisitor(relpath, docstring_ids, excluded_str_ids, rules=resolved_rules)
    visitor.visit(tree)
    return visitor.violations


def _walk(root: Path):
    """Wrap :func:`os.walk` for the directory tree at ``root``."""
    if root.is_file():
        # Scanning a single file: yield (parent, [], [name]).
        yield (str(root.parent), [], [root.name])
        return
    for dirpath, dirnames, filenames in os.walk(root):
        yield dirpath, dirnames, filenames


def scan_directory(
    root: Union[Path, str],
    relpath_root_or_exemptions: Optional[Union[List, str]] = None,
    exemptions: Optional[List] = None,
    rules: Optional[List[Rule]] = None,
    only_rule: Optional[str] = None,
) -> List[Violation]:
    """Walk a directory tree and return all violations under it.

    Directories listed in :data:`EXCLUDE_DIRS` are skipped.
    The v0.4.x positional signature ``scan_directory(root, exemptions)``
    is preserved (and ambiguous arg detection is delegated to the
    caller -- if the second arg is a ``list`` it is treated as
    ``exemptions``, otherwise as ``relpath_root``).
    """
    root_path = Path(root)
    # Disambiguate positional arg: list -> exemptions, str -> relpath_root.
    if isinstance(relpath_root_or_exemptions, list):
        exemptions = relpath_root_or_exemptions
        relpath_root = ""
    elif relpath_root_or_exemptions is None:
        relpath_root = ""
    else:
        relpath_root = relpath_root_or_exemptions
    resolved_rules = _resolve_rules(only_rule, rules)
    violations: List[Violation] = []
    for dirpath, dirnames, filenames in _walk(root_path):
        # Prune excluded dirs in-place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            abspath = Path(dirpath) / name
            rel = str(abspath.relative_to(root_path))
            if relpath_root:
                rel = f"{relpath_root.rstrip('/')}/{rel}"
            violations.extend(scan_file(abspath, rel, rules=resolved_rules))
    # Apply exemptions in-place; the v0.4.x contract is that
    # ``scan_directory`` accepts an ``exemptions=`` keyword and
    # the matching violations are removed (or downgraded) before
    # returning.
    if exemptions:
        for v in violations:
            for ex in exemptions:
                ex.apply(v)
        # Drop "terminal" exemptions (those that do not specify
        # ``severity`` or ``protocol_format``) -- they are
        # understood to fully remove the violation.
        violations = [
            v for v in violations
            if any(
                ex.matches(v) and (ex.protocol_format or ex.severity is not None)
                for ex in exemptions
            ) or not any(ex.matches(v) for ex in exemptions)
        ]
    return violations
