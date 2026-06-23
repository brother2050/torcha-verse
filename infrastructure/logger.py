"""Tiered logging system for TorchaVerse.

This module wraps Python's standard :mod:`logging` library with a
:mod:`rich`-powered console handler and a rotating file handler.  It
exposes a :func:`get_logger` factory used throughout the framework and a
:class:`Logger` wrapper that adds structured-data logging via
:meth:`Logger.log_dict`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = ["Logger", "get_logger", "set_log_level"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

#: Default log format for file output.
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
#: Default date format.
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Track loggers that have already been configured to avoid duplicate handlers.
_configured_loggers: Dict[str, logging.Logger] = {}

#: Module-level default settings (mutable via :func:`configure`).
_default_log_dir: Optional[Path] = None
_default_file_level: int = logging.DEBUG
_default_console_level: int = logging.INFO
_default_max_bytes: int = 10 * 1024 * 1024  # 10 MB
_default_backup_count: int = 5


# ---------------------------------------------------------------------------
# Rich console handler (optional dependency)
# ---------------------------------------------------------------------------
def _make_console_handler(level: int) -> logging.Handler:
    """Create a console handler, preferring :mod:`rich` when available."""
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=level,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt=_DATE_FORMAT))
    except Exception:
        # Fallback to a plain stream handler if rich is unavailable.
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
        )
    return handler


def _make_file_handler(
    log_file: Path, level: int, max_bytes: int, backup_count: int
) -> logging.Handler:
    """Create a :class:`RotatingFileHandler` for ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def configure(
    log_dir: Optional[Union[str, Path]] = None,
    console_level: Union[str, int] = "INFO",
    file_level: Union[str, int] = "DEBUG",
    max_bytes: int = _default_max_bytes,
    backup_count: int = _default_backup_count,
) -> None:
    """Configure global logging defaults.

    Args:
        log_dir: Directory for log files.  When ``None`` file logging is
            disabled unless a per-logger ``log_file`` is supplied.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of rotated backup files to keep.
    """
    global _default_log_dir, _default_console_level, _default_file_level
    global _default_max_bytes, _default_backup_count
    _default_log_dir = Path(log_dir).expanduser().resolve() if log_dir else None
    _default_console_level = _coerce_level(console_level)
    _default_file_level = _coerce_level(file_level)
    _default_max_bytes = max_bytes
    _default_backup_count = backup_count


def _coerce_level(level: Union[str, int]) -> int:
    """Normalise a level given as a string or int."""
    if isinstance(level, int):
        return level
    key = str(level).strip().upper()
    if key not in _LEVELS:
        raise ValueError(
            f"Unknown log level '{level}'. Valid: {sorted(set(_LEVELS))}."
        )
    return _LEVELS[key]


def set_log_level(level: Union[str, int]) -> None:
    """Set the console level for all configured loggers."""
    resolved = _coerce_level(level)
    for logger in _configured_loggers.values():
        logger.setLevel(resolved)
        for handler in logger.handlers:
            if not isinstance(handler, RotatingFileHandler):
                handler.setLevel(resolved)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_logger(
    name: str,
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The logger is cached by ``name`` so repeated calls return the same
    instance without duplicating handlers.  Console output is beautified
    with :mod:`rich` when available, and a rotating file handler is attached
    when ``log_file`` is provided or a global ``log_dir`` is configured.

    Args:
        name: Logger name (usually ``__name__`` or a class name).
        level: Minimum log level for the console handler.
        log_file: Optional explicit log file path.
        console: Whether to attach a console handler.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # handlers filter individually
    logger.propagate = False

    resolved_level = _coerce_level(level)

    if console:
        logger.addHandler(_make_console_handler(resolved_level))

    # Resolve the file path: explicit argument > global log_dir.
    file_path: Optional[Path] = None
    if log_file is not None:
        file_path = Path(log_file).expanduser().resolve()
    elif _default_log_dir is not None:
        file_path = _default_log_dir / f"{name}.log"

    if file_path is not None:
        logger.addHandler(
            _make_file_handler(
                file_path,
                _default_file_level,
                _default_max_bytes,
                _default_backup_count,
            )
        )

    _configured_loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Logger wrapper
# ---------------------------------------------------------------------------
class Logger:
    """High-level logger wrapper with structured-data support.

    Wraps a :class:`logging.Logger` and adds convenience methods such as
    :meth:`log_dict` for recording structured payloads (serialised to JSON).

    Example:
        >>> log = Logger("trainer")
        >>> log.info("Training started")
        >>> log.log_dict("INFO", {"loss": 0.23, "step": 100})
    """

    def __init__(
        self,
        name: str,
        level: Union[str, int] = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        console: bool = True,
    ) -> None:
        self._logger: logging.Logger = get_logger(
            name, level=level, log_file=log_file, console=console
        )
        self._name: str = name

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """The logger name."""
        return self._name

    @property
    def logger(self) -> logging.Logger:
        """The underlying :class:`logging.Logger`."""
        return self._logger

    # ------------------------------------------------------------------
    # Standard level methods
    # ------------------------------------------------------------------
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a DEBUG message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an INFO message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a WARNING message."""
        self._logger.warning(msg, *args, **kwargs)

    warn = warning  # alias

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an ERROR message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a CRITICAL message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an ERROR message with exception traceback."""
        self._logger.exception(msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Structured logging
    # ------------------------------------------------------------------
    def log_dict(
        self,
        level: Union[str, int],
        data: Dict[str, Any],
        message: Optional[str] = None,
    ) -> None:
        """Log a structured dictionary as JSON.

        Args:
            level: Log level (string or int).
            data: Dictionary payload to serialise.
            message: Optional human-readable prefix prepended to the JSON.
        """
        resolved = _coerce_level(level)
        try:
            payload = json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = repr(data)
        text = f"{message} | {payload}" if message else payload
        self._logger.log(resolved, text)

    def log(
        self, level: Union[str, int], msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log ``msg`` at the given ``level``."""
        self._logger.log(_coerce_level(level), msg, *args, **kwargs)

    # ------------------------------------------------------------------
    def set_level(self, level: Union[str, int]) -> None:
        """Update the console log level for this logger."""
        resolved = _coerce_level(level)
        self._logger.setLevel(resolved)
        for handler in self._logger.handlers:
            if not isinstance(handler, RotatingFileHandler):
                handler.setLevel(resolved)

    def add_file_handler(
        self,
        log_file: Union[str, Path],
        level: Union[str, int] = "DEBUG",
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
    ) -> RotatingFileHandler:
        """Attach an additional rotating file handler.

        Args:
            log_file: Path to the log file.
            level: Minimum level for this handler.
            max_bytes: Rotation size; defaults to the global setting.
            backup_count: Number of backups; defaults to the global setting.

        Returns:
            The created :class:`RotatingFileHandler`.
        """
        handler = _make_file_handler(
            Path(log_file).expanduser().resolve(),
            _coerce_level(level),
            max_bytes if max_bytes is not None else _default_max_bytes,
            backup_count if backup_count is not None else _default_backup_count,
        )
        self._logger.addHandler(handler)
        return handler
