"""Audit event logging for TorchaVerse.

This module provides an append-only audit trail for security-sensitive and
operational events: model downloads, training runs, inference calls,
exports, sharing, deletions, configuration changes and security events.

Unlike the singleton :class:`~infrastructure.logger.Logger`, the
:class:`AuditLogger` is **not** a singleton -- multiple instances can
coexist (e.g. one per subsystem) and each writes to its own JSONL sink.
Events are buffered in memory and flushed periodically (or when the buffer
fills) to a date-partitioned JSONL file under
``~/.local/share/torcha-verse/audit/<date>.jsonl`` (Linux/macOS) or
``%LOCALAPPDATA%/torcha-verse/audit/<date>.jsonl`` (Windows).

The logger is fully thread-safe (a single :class:`threading.Lock` guards
both the in-memory buffer and file writes) and supports historical queries
via :meth:`AuditLogger.query`.

Example:
    >>> logger = AuditLogger()
    >>> logger.log("DOWNLOAD", actor="user@example.com",
    ...            action="fetch_model", resource_id="meta-llama/Llama-3-8B",
    ...            details={"bytes": 16_000_000_000}, severity="info")
    >>> events = logger.query(event_type="DOWNLOAD")
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .logger import get_logger

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "EventType",
    "Severity",
]

#: Environment variable used to override the audit log directory.
_ENV_AUDIT_DIR: str = "TORCHAVERSE_AUDIT_DIR"

#: Default in-memory buffer size before an automatic flush is triggered.
_DEFAULT_BUFFER_SIZE: int = 100


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    """Enumerated audit event categories."""

    DOWNLOAD = "DOWNLOAD"
    TRAIN = "TRAIN"
    INFER = "INFER"
    EXPORT = "EXPORT"
    SHARE = "SHARE"
    DELETE = "DELETE"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SECURITY = "SECURITY"


class Severity(str, Enum):
    """Enumerated severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
def _default_audit_dir() -> Path:
    """Resolve the default audit log directory in a platform-aware way."""
    env = os.environ.get(_ENV_AUDIT_DIR)
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / "torcha-verse" / "audit"
    return Path.home() / ".local" / "share" / "torcha-verse" / "audit"


# ---------------------------------------------------------------------------
# AuditEvent dataclass
# ---------------------------------------------------------------------------
@dataclass
class AuditEvent:
    """A single audit record.

    Attributes:
        timestamp: UTC datetime when the event was recorded.
        event_type: One of :class:`EventType` (stored as its string value).
        actor: Identifier of the entity performing the action (user, service,
            system component, ...).
        action: Short verb describing what was done (e.g. ``"fetch_model"``).
        resource_id: Optional identifier of the affected resource.
        details: Arbitrary structured payload with extra context.
        severity: One of :class:`Severity` (stored as its string value).
    """

    timestamp: datetime
    event_type: str
    actor: str
    action: str
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = Severity.INFO.value

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor": self.actor,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": self.details,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------
class AuditLogger:
    """Append-only, thread-safe audit event logger.

    Events are appended to an in-memory buffer and periodically flushed to a
    date-partitioned JSONL file.  Each instance owns its own buffer and
    lock, so multiple loggers can run concurrently without interfering.

    Args:
        log_dir: Directory for JSONL files.  Defaults to a platform-aware
            location (see module docstring).
        buffer_size: Number of events buffered before an automatic flush.
            A buffer size of ``0`` flushes after every event.
        name: Logical name for this logger (used in diagnostics).
        auto_flush_on_exit: When ``True`` (default) register an ``atexit``
            handler that flushes any remaining buffered events on process
            exit.

    Example:
        >>> logger = AuditLogger(buffer_size=1)
        >>> logger.log("TRAIN", actor="trainer", action="start",
        ...            resource_id="llama-8b", severity="info")
        >>> logger.flush()
        >>> results = logger.query(event_type="TRAIN")
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        name: str = "audit",
        auto_flush_on_exit: bool = True,
    ) -> None:
        if buffer_size < 0:
            raise ValueError(f"buffer_size must be >= 0, got {buffer_size}.")

        self._log_dir: Path = (
            Path(log_dir).expanduser().resolve()
            if log_dir is not None
            else _default_audit_dir()
        )
        self._buffer_size: int = int(buffer_size)
        self._name: str = name
        self._buffer: List[AuditEvent] = []
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._diag = get_logger(f"{self.__class__.__name__}:{name}")

        if auto_flush_on_exit:
            atexit.register(self.close)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def log_dir(self) -> Path:
        """Directory where JSONL audit files are written."""
        return self._log_dir

    @property
    def name(self) -> str:
        """Logical name of this logger."""
        return self._name

    @property
    def buffer_size(self) -> int:
        """Configured automatic-flush threshold."""
        return self._buffer_size

    @property
    def pending(self) -> int:
        """Number of events currently buffered but not yet flushed."""
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(
        self,
        event_type: Union[str, EventType],
        actor: str,
        action: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: Union[str, Severity] = Severity.INFO,
    ) -> AuditEvent:
        """Record an audit event.

        Args:
            event_type: One of :class:`EventType` (or its string value).
            actor: Identifier of the entity performing the action.
            action: Short verb describing what was done.
            resource_id: Optional identifier of the affected resource.
            details: Optional structured payload with extra context.
            severity: One of :class:`Severity` (or its string value).

        Returns:
            The created :class:`AuditEvent`.

        Raises:
            ValueError: If ``event_type`` or ``severity`` is not recognised.
        """
        resolved_type = self._coerce_event_type(event_type)
        resolved_severity = self._coerce_severity(severity)

        if not actor:
            raise ValueError("actor must be a non-empty string.")
        if not action:
            raise ValueError("action must be a non-empty string.")

        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=resolved_type,
            actor=actor,
            action=action,
            resource_id=resource_id,
            details=dict(details) if details else {},
            severity=resolved_severity,
        )

        flush_needed = False
        with self._lock:
            if self._closed:
                raise RuntimeError("AuditLogger is closed; cannot log events.")
            self._buffer.append(event)
            if self._buffer_size <= 0 or len(self._buffer) >= self._buffer_size:
                flush_needed = True

        if flush_needed:
            self.flush()
        return event

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    def flush(self) -> int:
        """Write all buffered events to the JSONL sink and clear the buffer.

        Returns:
            The number of events flushed.
        """
        with self._lock:
            if not self._buffer:
                return 0
            to_write = list(self._buffer)
            self._buffer.clear()

        if not to_write:
            return 0

        self._log_dir.mkdir(parents=True, exist_ok=True)
        # Partition by the event's UTC date so each day gets its own file.
        partitions: Dict[str, List[AuditEvent]] = {}
        for event in to_write:
            key = event.timestamp.strftime("%Y-%m-%d")
            partitions.setdefault(key, []).append(event)

        for date_key, events in partitions.items():
            path = self._log_dir / f"{date_key}.jsonl"
            with open(path, "a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(
                        json.dumps(event.to_dict(), ensure_ascii=False)
                        + "\n"
                    )
        return len(to_write)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query(
        self,
        event_type: Optional[Union[str, EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: Optional[Union[str, Severity]] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Query historical audit events.

        Searches both the in-memory buffer and every JSONL file on disk.
        All filters are optional and combined with logical AND.

        Args:
            event_type: Filter by event type.
            start_time: Only events at or after this datetime.
            end_time: Only events at or before this datetime.
            actor: Filter by actor (exact match).
            action: Filter by action (exact match).
            resource_id: Filter by resource id (exact match).
            severity: Filter by severity.
            limit: Maximum number of events to return (most recent first).

        Returns:
            A list of matching :class:`AuditEvent` objects, sorted from
            oldest to newest.
        """
        type_filter = (
            self._coerce_event_type(event_type) if event_type else None
        )
        sev_filter = (
            self._coerce_severity(severity) if severity else None
        )

        # Normalise naive datetimes to UTC for consistent comparison.
        start_utc = self._to_utc(start_time)
        end_utc = self._to_utc(end_time)

        results: List[AuditEvent] = []

        # 1. On-disk events.
        if self._log_dir.exists():
            for jsonl_file in sorted(self._log_dir.glob("*.jsonl")):
                results.extend(self._read_file(jsonl_file))

        # 2. In-memory buffered events (not yet flushed).
        with self._lock:
            results.extend(self._buffer)

        # 3. Filter.
        filtered = [
            event
            for event in results
            if self._matches(
                event,
                type_filter,
                start_utc,
                end_utc,
                actor,
                action,
                resource_id,
                sev_filter,
            )
        ]

        # 4. Sort oldest -> newest.
        filtered.sort(key=lambda e: e.timestamp)

        if limit is not None and limit >= 0:
            filtered = filtered[-limit:] if limit else []
        return filtered

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush pending events and mark the logger as closed."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.flush()
        except Exception as exc:  # pragma: no cover - best effort
            self._diag.warning("Error during audit logger close: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_event_type(value: Union[str, EventType]) -> str:
        """Normalise an event type to its canonical string value."""
        if isinstance(value, EventType):
            return value.value
        key = str(value).strip().upper()
        try:
            return EventType(key).value
        except ValueError:
            valid = ", ".join(e.value for e in EventType)
            raise ValueError(
                f"Unknown event_type {value!r}. Valid: {valid}."
            )

    @staticmethod
    def _coerce_severity(value: Union[str, Severity]) -> str:
        """Normalise a severity to its canonical string value."""
        if isinstance(value, Severity):
            return value.value
        key = str(value).strip().lower()
        try:
            return Severity(key).value
        except ValueError:
            valid = ", ".join(s.value for s in Severity)
            raise ValueError(
                f"Unknown severity {value!r}. Valid: {valid}."
            )

    @staticmethod
    def _to_utc(value: Optional[datetime]) -> Optional[datetime]:
        """Convert a naive datetime to UTC; pass through ``None``."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _read_file(path: Path) -> List[AuditEvent]:
        """Parse a JSONL file into :class:`AuditEvent` objects."""
        events: List[AuditEvent] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(AuditLogger._event_from_dict(data))
        except OSError:
            return []
        return events

    @staticmethod
    def _event_from_dict(data: Dict[str, Any]) -> AuditEvent:
        """Reconstruct an :class:`AuditEvent` from a dictionary."""
        timestamp_raw = data.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(timestamp_raw)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)
        return AuditEvent(
            timestamp=timestamp,
            event_type=str(data.get("event_type", "")),
            actor=str(data.get("actor", "")),
            action=str(data.get("action", "")),
            resource_id=data.get("resource_id"),
            details=data.get("details") or {},
            severity=str(data.get("severity", Severity.INFO.value)),
        )

    @staticmethod
    def _matches(
        event: AuditEvent,
        event_type: Optional[str],
        start: Optional[datetime],
        end: Optional[datetime],
        actor: Optional[str],
        action: Optional[str],
        resource_id: Optional[str],
        severity: Optional[str],
    ) -> bool:
        """Return ``True`` if ``event`` satisfies all non-None filters."""
        if event_type is not None and event.event_type != event_type:
            return False
        if start is not None and event.timestamp < start:
            return False
        if end is not None and event.timestamp > end:
            return False
        if actor is not None and event.actor != actor:
            return False
        if action is not None and event.action != action:
            return False
        if resource_id is not None and event.resource_id != resource_id:
            return False
        if severity is not None and event.severity != severity:
            return False
        return True

    def __repr__(self) -> str:
        with self._lock:
            pending = len(self._buffer)
        return (
            f"AuditLogger(name={self._name!r}, log_dir={self._log_dir!s}, "
            f"pending={pending})"
        )
