"""Global exception handling for TorchaVerse.

This module provides :class:`ErrorHandler`, a registry-based exception
handler that maps exception types to callable handlers.  It ships with
built-in handlers for common failure modes (CUDA out-of-memory, network
timeouts, format errors) and integrates with the :mod:`logger` module to
record every handled exception.  A :func:`with_error_handler` decorator is
provided for function-level protection.
"""

from __future__ import annotations

import functools
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import torch

from .logger import get_logger

__all__ = ["ErrorHandler", "with_error_handler", "ErrorAction"]

F = TypeVar("F", bound=Callable[..., Any])

#: Type alias for an exception handler callable.
ExceptionHandler = Callable[[BaseException], Any]


class ErrorAction:
    """Sentinel return values describing how to proceed after handling.

    Handlers may return one of these sentinels (or any other value) to
    communicate intent to the caller of :meth:`ErrorHandler.handle`.
    """

    RAISE = "raise"
    SUPPRESS = "suppress"
    RETRY = "retry"


class ErrorHandler:
    """Registry-based global exception handler.

    Handlers are registered against exception types.  When :meth:`handle`
    is invoked it walks the exception's method-resolution order (MRO) to
    find the most specific registered handler.  If no handler matches the
    exception is re-raised.

    The handler is implemented as a singleton so that registrations are
    shared across the whole framework.
    """

    _instance: Optional["ErrorHandler"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ErrorHandler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, logger_name: str = "error_handler") -> None:
        if self._initialized:
            return
        self._initialized = True

        self._handlers: Dict[Type[BaseException], ExceptionHandler] = {}
        self._logger = get_logger(logger_name)
        self._default_handler: Optional[ExceptionHandler] = None
        self._suppress_unknown: bool = False

        # Register the built-in handlers.
        self._register_builtin_handlers()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_handler(
        self,
        exception_type: Type[BaseException],
        handler: ExceptionHandler,
    ) -> None:
        """Register ``handler`` for ``exception_type``.

        Args:
            exception_type: The exception class (or subclass) to handle.
            handler: Callable receiving the exception instance.  Its return
                value is forwarded to the caller of :meth:`handle`.
        """
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, BaseException
        ):
            raise TypeError("exception_type must be a subclass of BaseException.")
        if not callable(handler):
            raise TypeError("handler must be callable.")
        self._handlers[exception_type] = handler
        self._logger.debug(
            "Registered handler for %s.", exception_type.__name__
        )

    def unregister_handler(
        self, exception_type: Type[BaseException]
    ) -> Optional[ExceptionHandler]:
        """Remove a previously registered handler.

        Returns:
            The removed handler, or ``None`` if none was registered.
        """
        return self._handlers.pop(exception_type, None)

    def set_default_handler(self, handler: Optional[ExceptionHandler]) -> None:
        """Set a fallback handler used when no type-specific match exists."""
        self._default_handler = handler

    def set_suppress_unknown(self, suppress: bool) -> None:
        """When ``True``, unknown exceptions are logged but not re-raised."""
        self._suppress_unknown = suppress

    # ------------------------------------------------------------------
    # Handling
    # ------------------------------------------------------------------
    def handle(self, exception: BaseException) -> Any:
        """Process ``exception`` using the best matching handler.

        The exception's MRO is inspected to find the most specific
        registered handler.  When none matches the default handler (if any)
        is used; otherwise the exception is re-raised.

        Args:
            exception: The exception instance to handle.

        Returns:
            The value returned by the matched handler.

        Raises:
            BaseException: Re-raises ``exception`` when no handler matches
                and suppression is disabled.
        """
        handler = self._find_handler(exception)

        # Always log the exception.
        self._log_exception(exception, handled=handler is not None)

        if handler is not None:
            try:
                return handler(exception)
            except BaseException as handler_error:
                self._logger.error(
                    "Handler %s itself raised: %s",
                    getattr(handler, "__name__", handler),
                    handler_error,
                )
                raise

        if self._default_handler is not None:
            return self._default_handler(exception)

        if self._suppress_unknown:
            return ErrorAction.SUPPRESS

        raise exception

    def _find_handler(self, exception: BaseException) -> Optional[ExceptionHandler]:
        """Return the most specific registered handler for ``exception``."""
        for exc_type in type(exception).__mro__:
            if exc_type in self._handlers:
                return self._handlers[exc_type]
        return None

    def _log_exception(
        self, exception: BaseException, handled: bool
    ) -> None:
        """Record ``exception`` in the log."""
        tb = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
        status = "handled" if handled else "unhandled"
        self._logger.error(
            "Exception (%s): %s: %s\n%s",
            status,
            type(exception).__name__,
            exception,
            tb,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "ErrorHandler":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        if exc_val is None:
            return False
        try:
            self.handle(exc_val)
            return True  # suppress the exception
        except BaseException:
            return False  # propagate

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------
    def _register_builtin_handlers(self) -> None:
        """Register handlers for common, framework-relevant exceptions."""
        # CUDA out-of-memory.
        self.register_handler(
            torch.cuda.OutOfMemoryError, self._handle_oom  # type: ignore[arg-type]
        )
        # Generic timeout (covers socket / http timeouts subclassing it).
        self.register_handler(TimeoutError, self._handle_timeout)
        # Format / value errors.
        self.register_handler(ValueError, self._handle_format_error)
        self.register_handler(TypeError, self._handle_format_error)

    def _handle_oom(self, exception: BaseException) -> str:
        """Handle a CUDA out-of-memory error.

        Clears the allocator cache and returns a sentinel so callers can
        decide to retry with a smaller batch.
        """
        self._logger.warning(
            "CUDA out-of-memory detected. Clearing cache to allow retry. "
            "Consider reducing batch size or sequence length."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ErrorAction.RETRY

    def _handle_timeout(self, exception: BaseException) -> str:
        """Handle a timeout error (e.g. API request timeout)."""
        self._logger.warning(
            "A timeout occurred: %s. The operation may be retried.", exception
        )
        return ErrorAction.RETRY

    def _handle_format_error(self, exception: BaseException) -> str:
        """Handle data format / type errors."""
        self._logger.error(
            "Format error: %s. The input did not match the expected schema.",
            exception,
        )
        return ErrorAction.SUPPRESS

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None
        cls._initialized = False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def with_error_handler(
    func: Optional[F] = None,
    *,
    handler: Optional[ErrorHandler] = None,
    reraise: bool = False,
    default: Any = None,
) -> Union[F, Callable[[F], F]]:
    """Decorator that wraps a function with global error handling.

    Can be used with or without arguments::

        @with_error_handler
        def foo(): ...

        @with_error_handler(reraise=True)
        def bar(): ...

    Args:
        func: The function to wrap (when used without parentheses).
        handler: Optional explicit :class:`ErrorHandler`. Defaults to the
            singleton instance.
        reraise: When ``True`` the exception is re-raised after being logged
            and handled.
        default: Value returned when an exception is suppressed.

    Returns:
        The decorated function, or a decorator when called without ``func``.
    """

    def _decorate(target: F) -> F:
        @functools.wraps(target)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            error_handler = handler if handler is not None else ErrorHandler()
            try:
                return target(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - intentional
                result = error_handler.handle(exc)
                if reraise:
                    raise
                # If the handler returned a sentinel, translate it.
                if result is ErrorAction.RAISE:
                    raise
                if result in (ErrorAction.SUPPRESS, ErrorAction.RETRY):
                    return default
                return result

        return _wrapper  # type: ignore[return-value]

    if func is not None and callable(func):
        return _decorate(func)
    return _decorate
