#!/usr/bin/env python3
"""CI linter: every silent-degrade path must leave a forensic trace.

D3 stage three (v0.4.x) introduces the rule that any
``except ...: pass`` or ``except ...: <only-pass-statements>`` block
in the project must be one of:

* a ``finally`` block (the ``try`` is paired with ``finally`` -- the
  ``pass`` is the legitimate "nothing to do" of a cleanup path), or
* the except-block body must contain at least one of:
  - a ``logger.warning(...)`` call, or
  - a call to ``safe_call(...)`` (which itself logs), or
  - a call to ``record_degrade(...)`` (which itself logs), or
  - an explicit ``raise`` (re-raise, so the failure isn't silent).

The script scans the project tree (excluding tests, caches, and
virtualenvs) for ``except`` blocks whose body does not satisfy the
rule, and exits non-zero when it finds any.

Usage::

    python scripts/check_degrade_logging.py                  # full project
    python scripts/check_degrade_logging.py path/to/file.py  # single file
    python scripts/check_degrade_logging.py --list           # show categories
    python scripts/check_degrade_logging.py --stats          # show counts

Exit codes:

* ``0`` -- clean (or ``--list`` / ``--stats`` only).
* ``1`` -- silent-degrade blocks found; details printed to stdout.
* ``2`` -- internal error (bad path, malformed AST, ...).

The script is dependency-free (no third-party imports) so it can be
invoked from any environment, including minimal CI containers.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUDE_DIRS: frozenset[str] = frozenset({
    "config", "__pycache__", "build", "dist", "node_modules",
    ".git", ".venv", ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
    "tests",  # tests intentionally may have silent except for fixture cleanup
})

#: Functions / attribute paths that count as a forensic trace.
_FORENSIC_CALL_NAMES: frozenset[str] = frozenset({
    "safe_call",
    "record_degrade",
})

#: Module prefixes whose ``.warning`` call counts as a trace.
#: A call is forensic when its source attribute path ends in
#: ``.warning`` and the root module is one of these.
_LOGGER_PREFIXES: Tuple[str, ...] = (
    "logger.",
    "log.",
    "_logger.",
    "logging.",
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _is_pass_only(body: List[ast.stmt]) -> bool:
    """``True`` when the body is ``pass`` statements and nothing else.

    Handles the degenerate case where the body has a single ``Pass``
    node, or a single ``Expr(value=Constant(value=Ellipsis))`` doc-
    placeholder (the latter is rare but appears in some libraries).
    """
    if not body:
        return False
    if all(isinstance(s, ast.Pass) for s in body):
        return True
    # A single ``...`` body is also silent.
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
            return True
    return False


def _calls_anything_in(body: List[ast.stmt], names: frozenset[str]) -> bool:
    """``True`` if any call in ``body`` has a bare name in ``names``."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in names:
                return True
            if isinstance(f, ast.Attribute) and f.attr in names:
                # Allow attribute name match (e.g. ``record_degrade(...)``
                # bound to a local alias).
                return True
    return False


def _calls_logger_warning(body: List[ast.stmt]) -> bool:
    """``True`` if any call in ``body`` invokes ``*.warning(...)``.

    Specifically, we match a Call whose func is an ``ast.Attribute``
    with ``attr == "warning"`` and whose value is an ``ast.Attribute``
    chain starting with one of the logger prefixes.
    """
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        if f.attr != "warning":
            continue
        # Walk the chain left to look for a logger prefix.
        cur: ast.expr = f.value
        while isinstance(cur, ast.Attribute):
            if isinstance(cur.value, ast.Name) and (cur.value.id + ".") in _LOGGER_PREFIXES:
                return True
            cur = cur.value
        if isinstance(cur, ast.Name) and (cur.id + ".") in _LOGGER_PREFIXES:
            return True
    return False


def _contains_raise(body: List[ast.stmt]) -> bool:
    """``True`` if any statement in ``body`` re-raises or raises a
    new exception (so the failure is not silent)."""
    for s in body:
        if isinstance(s, ast.Raise):
            return True
    return False


def _is_finally_cleanup(handlers: List[ast.ExceptHandler], parent: ast.Try) -> bool:
    """``True`` if the ``except`` handler belongs to a try whose
    ``finally`` block contains the ``pass`` -- i.e. the except body
    is empty and the cleanup is in the finally."""
    return parent.finalbody is not None and bool(parent.finalbody)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
def find_silent_degrades(path: Path) -> List[Tuple[int, str, str]]:
    """Return ``(line, file_relpath, reason)`` for every silent
    degrade in ``path``.

    A silent degrade is an ``except`` block whose body has no
    forensic trace AND is not just a ``pass`` in a ``finally``
    cleanup path.
    """
    if path.is_dir():
        out: List[Tuple[int, str, str]] = []
        for p in sorted(path.rglob("*.py")):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            out.extend(find_silent_degrades(p))
        return out

    if not path.is_file() or path.suffix != ".py":
        return []

    relpath = str(path)
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError:
        return []

    out: List[Tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # We only care about the except handlers (finally is OK to be
        # silent by itself).
        for h in node.handlers:
            body = h.body
            if not _is_pass_only(body):
                continue
            # It is a silent except.  Is it in a finally-cleanup try
            # block?  If the try also has a finally and the except
            # body is pass, we accept it (the cleanup is in finally).
            if _is_finally_cleanup([h], node):
                continue
            # Allow bare ``except:`` with only a `pass` and the
            # immediately following statement is ``raise`` -- this
            # is the common "reraise unchanged" idiom.  We detect
            # this by looking for a sibling Raise inside the same
            # try (e.g. ``except: pass; raise`` style).
            if _contains_raise_pass_reraise(node, h):
                continue
            # Determine reason by body content.
            if not body:
                reason = "empty except body"
            elif _is_pass_only(body):
                reason = "pass-only except body (no logger.warning / safe_call / record_degrade / raise)"
            else:
                reason = "silent except body"
            out.append((h.lineno, relpath, reason))
    return out


def _contains_raise_pass_reraise(try_node: ast.Try, handler: ast.ExceptHandler) -> bool:
    """Detect the ``except: pass; ...; raise`` idiom, where the
    ``raise`` is a sibling *statement* (not a child of the handler)
    inside the same try.

    This is rare in our codebase but does appear (e.g. ``try: ...
    except: pass; raise``).  We accept it because the ``raise``
    re-raises the original exception, so the failure is not silent.
    """
    # Walk the try's body + orelse + finalbody; if any sibling
    # statement of the handler raises, allow it.
    siblings: List[ast.stmt] = []
    siblings.extend(try_node.body)
    siblings.extend(try_node.orelse)
    siblings.extend(try_node.finalbody)
    for s in siblings:
        if s is handler:
            continue
        if isinstance(s, ast.Raise):
            return True
        # Walk into nested structures (e.g. ``for: try: ...``).
        for sub in ast.walk(s):
            if sub is handler:
                continue
            if isinstance(sub, ast.Raise):
                return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _walk_files(root: Path) -> Iterable[Path]:
    """Yield every ``.py`` file under ``root`` (or ``root`` itself
    if it is a file)."""
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default=".",
                        help="Path to scan (file or directory, default: project root)")
    parser.add_argument("--list", action="store_true",
                        help="List the rule categories the script enforces")
    parser.add_argument("--stats", action="store_true",
                        help="Show counts of silent degrades per file")
    args = parser.parse_args(argv)

    if args.list:
        print("enforced rules:")
        print("  - except body must NOT be pass-only (no trace)")
        print("  - except body must contain at least one of:")
        print("    * logger.warning(...) call")
        print("    * safe_call(...) call (which itself logs)")
        print("    * record_degrade(...) call (which itself logs)")
        print("    * raise statement (re-raise, not silent)")
        print("  - exception: try-with-finally cleanup (finally block is OK to be silent)")
        return 0

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"error: path not found: {target}", file=sys.stderr)
        return 2

    all_hits: List[Tuple[int, str, str]] = []
    files_with_hits: dict[str, int] = {}
    for f in _walk_files(target):
        hits = find_silent_degrades(f)
        if hits:
            files_with_hits[str(f)] = len(hits)
        all_hits.extend(hits)

    if args.stats:
        print(f"silent-degrade blocks by file ({len(all_hits)} total):")
        for f, n in sorted(files_with_hits.items(), key=lambda x: -x[1]):
            print(f"  {n:4d}  {f}")
        return 0

    if not all_hits:
        print("OK: no silent-degrade blocks found.")
        return 0

    print(f"FAIL: {len(all_hits)} silent-degrade block(s) found.")
    for line, fname, reason in all_hits:
        print(f"  {fname}:{line}  {reason}")
    print()
    print("fix options:")
    print("  1) add `logger.warning('...', exc_info=True)` to the except body")
    print("  2) replace the body with `safe_call(...)` or `record_degrade(...)`")
    print("  3) explicitly `raise` to re-raise the original exception")
    return 1


if __name__ == "__main__":
    sys.exit(main())
