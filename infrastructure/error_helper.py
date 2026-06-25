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
return.  It is intentionally small (~30 lines) and unconfigurable: if
the caller needs richer behavior they should ``try/except`` explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from .logger import get_logger

__all__ = ["safe_call"]

T = TypeVar("T")

#: Module-level logger for ``safe_call`` events.
_logger = get_logger("infrastructure.error_helper")


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    fallback: Optional[T] = None,
    expected: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    op: str = "",
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
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        The return value of ``fn``, or ``fallback`` if it raised.
    """
    try:
        return fn(*args, **kwargs)
    except expected as exc:
        label = op or getattr(fn, "__name__", "<callable>")
        _logger.warning("safe_call: %s failed: %s", label, exc)
        return fallback
