"""Lightweight error helper for TorchaVerse.

The previous ``infrastructure.error_handler`` (325 lines) provided a
generic ``ErrorHandler.register_handler`` + ``with_error_handler`` decorator
mechanism.  In practice this pattern silently swallowed exceptions and
made bug diagnosis harder -- the framework already uses a single,
centralized logger and explicit ``try/except`` blocks where recovery is
actually required.

This module exposes a single, honest helper: :func:`safe_call`, which
runs ``fn(*args, **kwargs)`` and either returns the value or converts an
expected exception into a logged warning with an optional fallback
return.  It is intentionally small (~70 lines) and unconfigurable: if
the caller needs richer behavior they should ``try/except`` explicitly.

D3 stage three (v0.4.x) additions
---------------------------------
* :func:`safe_call` now **always** emits a ``logger.warning`` on the
  fallback path.  Callers can override the logger via the
  ``logger=`` argument, but they cannot silence the warning.  This
  guarantees that every silent-degrade path leaves at least one
  forensic trace.
* :func:`safe_call` now publishes a counter to
  :data:`DEGRADE_COUNTERS` keyed by ``op_id`` (a stable identifier
  the caller provides).  The counter is a plain
  :class:`collections.Counter`; the test suite reads it back to
  assert that degrade events are observable.  This is the v0.4.x
  way of "metrics before M1": no Prometheus, no schema, just an
  importable dict.
* :func:`record_degrade` is a thin helper for sites that cannot
  use :func:`safe_call` (e.g. ``finally`` blocks or sandbox-
  generated code) but still want to bump the same counter.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Optional, TypeVar

from .logger import get_logger

__all__ = ["safe_call", "record_degrade", "DEGRADE_COUNTERS"]


T = TypeVar("T")

#: Module-level logger for ``safe_call`` events.
_logger = get_logger("infrastructure.error_helper")

#: Per-``op_id`` degrade counter.  Read by tests / future metrics
#: endpoints (M1 of v1.0.0 will replace this dict with a Prometheus
#: counter without changing the call sites).
#:
#: Key: the ``op_id`` string passed to :func:`safe_call` /
#: :func:`record_degrade`.  Value: how many times the fallback path
#: has fired for that op since process start.
DEGRADE_COUNTERS: Counter = Counter()


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    fallback: Optional[T] = None,
    expected: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    op: str = "",
    op_id: str = "",
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Optional[T]:
    """Call ``fn`` and convert a single expected exception to ``fallback``.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to ``fn``.
        fallback: Value returned when ``fn`` raises ``expected``.
        expected: Exception class (or tuple) that triggers the fallback.
            Defaults to the bare :class:`Exception`; callers are
            encouraged to narrow this to the specific exception they
            anticipate.
        op: Short human-readable description of the operation (used in
            the warning log message).
        op_id: Stable identifier for the degrade path.  Used as the
            key in :data:`DEGRADE_COUNTERS`.  When omitted, the
            counter key is derived from ``op`` and ``fn.__name__``
            (but tests and dashboards should always pass an explicit
            ``op_id`` so the counter name is stable across refactors).
        logger: Override the warning logger.  Defaults to
            :data:`_logger`.  Passing ``None`` is the same as not
            passing it -- warnings are never silent.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        The return value of ``fn``, or ``fallback`` if it raised.
    """
    log = logger or _logger
    counter_key = op_id or "{}::{}".format(op or "safe_call", getattr(fn, "__name__", "<callable>"))
    try:
        return fn(*args, **kwargs)
    except expected as exc:
        label = op or getattr(fn, "__name__", "<callable>")
        log.warning("safe_call: %s failed: %s", label, exc)
        DEGRADE_COUNTERS[counter_key] += 1
        return fallback


def record_degrade(op_id: str, *, exc: Optional[BaseException] = None,
                   op: str = "", logger: Optional[logging.Logger] = None) -> None:
    """Bump :data:`DEGRADE_COUNTERS` and log a warning.

    Use this from ``finally`` blocks, sandbox-generated code, or any
    other site that cannot use :func:`safe_call` (e.g. the
    exception object is already swallowed by an enclosing
    ``except``).  The function is deliberately a single statement
    so it can be inlined at the call site.

    Args:
        op_id: Stable identifier for the degrade path (counter key).
        exc: The exception that triggered the degrade.  Included in
            the warning log message when provided.
        op: Human-readable operation label (defaults to ``op_id``).
        logger: Override the warning logger.
    """
    log = logger or _logger
    label = op or op_id
    if exc is None:
        log.warning("degrade: %s (no exception attached)", label)
    else:
        log.warning("degrade: %s: %s", label, exc)
    DEGRADE_COUNTERS[op_id] += 1
