"""AST helpers for the hardcoding scanner (v0.6.x).

The helpers cover three concerns:

* docstring / import-call collection (so the visitor can skip
  string constants that are *not* user-tunable runtime config);
* parent-link attachment (so a constant can ask "am I inside an
  ``os.environ.get`` call?");
* path / log / runtime-attr heuristics (the rule engines ask
  these to downgrade / suppress violations).

All helpers are pure functions; the visitor / scan modules call
them in order.
"""

from __future__ import annotations

import ast
from typing import Any, Set, Tuple

from ._constants import (
    IMPORT_CALLS,
    LOG_METHODS,
    PATH_ATTRS,
    PATH_RE,
    RUNTIME_FUNCS,
    STRUCTURAL_MAX,
    STRUCTURAL_MIN,
    STRUCTURAL_PACKAGES,
)

__all__ = [
    "is_import_call",
    "collect_str_ids",
    "collect_docstring_ids",
    "attach_parents",
    "looks_like_path",
    "is_log_message_format",
    "is_runtime_attr",
    "is_structural_init",
]


def is_import_call(node: ast.Call) -> bool:
    """Return ``True`` when ``node`` is ``import_module(...)`` / ``__import__(...)``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in IMPORT_CALLS:
        return True
    if isinstance(func, ast.Attribute) and func.attr in IMPORT_CALLS:
        return True
    return False


def collect_str_ids(node: ast.AST, out: set) -> None:
    """Collect ``id(node)`` of every string :class:`ast.Constant` under ``node``."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.add(id(child))


def collect_docstring_ids(
    tree: ast.AST, docstring_ids: set, excluded_str_ids: set
) -> tuple[set, set]:
    """Populate docstring / import-call string id sets for the tree.

    The two sets are used by the AST visitor to skip docstring
    contents and import-related string arguments (which are not
    "hard-coded values" in the operator-config sense).
    """
    # Docstrings: the first statement of every Module / FunctionDef /
    # AsyncFunctionDef / ClassDef, when it is a string Constant.
    for node in ast.walk(tree):
        if isinstance(node, (
            ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_ids.add(id(first.value))

    # Import calls: arguments of import_module / __import__.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_import_call(node):
            for arg in node.args:
                collect_str_ids(arg, excluded_str_ids)
            for kw in node.keywords:
                collect_str_ids(kw.value, excluded_str_ids)

    return docstring_ids, excluded_str_ids


def attach_parents(tree: ast.AST) -> None:
    """Walk ``tree`` in-place adding ``.parent`` attributes to each node.

    Used by :func:`is_runtime_attr` so that a string/number constant
    can ask "am I the argument of an ``os.environ.get`` call?".
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def looks_like_path(s: str) -> bool:
    """Return ``True`` when ``s`` looks like a filesystem path."""
    if not isinstance(s, str):
        return False
    if len(s) < 2:
        return False
    if "/" not in s and "\\" not in s:
        return False
    return PATH_RE.search(s) is not None


def is_log_message_format(node: ast.AST) -> bool:
    """Return ``True`` when ``node`` is a log-message format string.

    A node is a log message format when it is the first positional
    argument of a :func:`logger.{level}` call (any standard logging
    method) or any callable whose attribute chain ends in
    ``logger.<level>``.
    """
    parent = getattr(node, "parent", None)
    if parent is None:
        return False
    if not isinstance(parent, ast.Call):
        return False
    if not parent.args or parent.args[0] is not node:
        return False
    func = parent.func
    if isinstance(func, ast.Attribute) and func.attr in LOG_METHODS:
        return True
    return False


def is_runtime_attr(node: ast.AST) -> bool:
    """Return ``True`` when ``node`` is the argument of a runtime-config call.

    The runtime-config call patterns are:

    * ``os.environ.get("FOO", ...)`` / ``os.environ["FOO"]``
    * ``os.getenv("FOO", ...)``
    * ``Path("...").expanduser()`` / ``Path("...").expandvars()``
    * ``sys.argv[N]``
    * Any call to a function in :data:`RUNTIME_FUNCS`
    * An argument of a Call whose result is itself the
      *target* of a chained ``.expanduser()`` / ``.expandvars()`` /
      ``.resolve()`` / ``.absolute()`` / ``.parent`` / ``.joinpath()``
      call (e.g. ``Path("...").expanduser()`` -- the "..." is the
      argument of ``Path`` but the *whole expression* is a
      runtime-config call).
    """
    parent = getattr(node, "parent", None)
    if parent is None:
        return False
    if isinstance(parent, ast.Subscript):
        value = parent.value
        if isinstance(value, ast.Attribute) and value.attr in PATH_ATTRS:
            return True
        # os.environ[ ... ] / sys.argv[ ... ]
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            if value.value.id in ("os", "sys") and value.attr in ("environ", "argv"):
                return True
        return False
    if isinstance(parent, ast.Call):
        func = parent.func
        # Direct function call: getenv(...) / getattr(...).
        if isinstance(func, ast.Name) and func.id in RUNTIME_FUNCS:
            return True
        # Attribute call:
        #   os.environ.get(...)            -> "get" (or RUNTIME_FUNCS)
        #   Path("...").expanduser()       -> "expanduser" in PATH_ATTRS
        #   os.environ[...] / sys.argv[...]  (handled above)
        if isinstance(func, ast.Attribute):
            if func.attr in RUNTIME_FUNCS:
                return True
            if func.attr in PATH_ATTRS:
                return True
        # Chained: ``Path("...").expanduser()`` -- the parent Call
        # is the ``Path(...)`` call, and ``parent.parent`` is an
        # ``Attribute`` (the ``.expanduser`` part) whose *value* is
        # the ``Path(...)`` call.
        grandparent = getattr(parent, "parent", None)
        if isinstance(grandparent, ast.Attribute) and grandparent.value is parent:
            if grandparent.attr in PATH_ATTRS:
                return True
    return False


def is_structural_init(relpath: str, value: Any) -> bool:
    """Return ``True`` when ``value`` is a model-structural numeric hyperparam.

    The v0.4.x D1 convention considers a numeric literal in a
    ``__init__`` method of a *model* package to be "structural"
    (e.g. ``self.num_heads = 8``) when the value is in
    [``STRUCTURAL_MIN``, ``STRUCTURAL_MAX``].  Such literals are
    reported but tagged ``info`` instead of ``critical`` so CI does
    not fail on them.
    """
    if not any(relpath.startswith(prefix) for prefix in STRUCTURAL_PACKAGES):
        return False
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return STRUCTURAL_MIN <= value <= STRUCTURAL_MAX
