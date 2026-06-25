"""Thread-safety decorator used by :mod:`infrastructure.resource_budget`.

The :func:`threadsafe` decorator wraps a method so that the wrapped
function is executed under the instance's ``_lock`` attribute
(``threading.RLock``).  It is shared between :class:`BudgetTracker`
and any future class that needs the same per-instance locking
pattern (vector stores, asset stores, etc.).

This module is intentionally tiny -- it exists as a separate file
to keep the :mod:`resource_budget` sub-package free of import-time
``threading`` boilerplate that other modules can re-use.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["threadsafe"]


def threadsafe(method: F) -> F:
    """Wrap ``method`` so it executes under ``self._lock``.

    The wrapped method must be a regular method, an instance method,
    or a classmethod -- the decorator introspects ``self`` from the
    first positional argument and acquires ``self._lock`` (a
    ``threading.RLock`` instance) for the duration of the call.

    Args:
        method: The method to wrap.

    Returns:
        The wrapped method (descriptor).
    """
    @functools.wraps(method)
    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock: threading.RLock = self._lock
        with lock:
            return method(self, *args, **kwargs)

    return _wrapper  # type: ignore[return-value]
