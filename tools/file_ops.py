"""File operations tool with path-safety enforcement.

This module provides :class:`FileOpsTool`, a :class:`BaseTool`
implementation that performs read, write, list, exists, and mkdir
operations on the local filesystem.

Security:

* Access is confined to an allow-list of permitted root directories.
* System-sensitive directories (``/etc``, ``/root``, ``/usr``, ``/bin``,
  ``/sbin``, ``/var``, ``/proc``, ``/sys``, ``/dev``) are always blocked.
* Symlinks are resolved and checked, preventing escape via symbolic links.
* Path traversal (``..``) is rejected after normalisation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from core.tool_registry import BaseTool
from infrastructure.logger import get_logger

__all__ = ["FileOpsTool"]

# ---------------------------------------------------------------------------
# Default blocked system directories (POSIX).
# ---------------------------------------------------------------------------
_BLOCKED_DIRS: tuple = (
    "/etc",
    "/root",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/lib",
    "/lib64",
    "/snap",
    "/run",
    "/srv",
    "/lost+found",
    "/mnt",
    "/media",
)


class FileOpsTool(BaseTool):
    """Read, write, and manage files within permitted directories.

    The tool supports five operations selected by the ``operation``
    parameter:

    * ``read``      -- read a file's text contents.
    * ``write``     -- write text content to a file.
    * ``list_dir``  -- list the entries of a directory.
    * ``exists``    -- check whether a path exists.
    * ``mkdir``     -- create a directory (and parents).

    All paths are validated against a block-list of system directories
    and, optionally, confined to an allow-list of permitted roots.

    Args:
        allowed_dirs: Optional sequence of directories that the tool is
            permitted to access.  When ``None`` the tool allows any
            non-system path.
        max_file_size_mb: Maximum file size (in MB) that may be read or
            written, to prevent loading huge files into memory.
    """

    name: str = "file_ops"
    description: str = "Read, write, and manage files"
    parameter_schema: Dict[str, Any] = {
        "operation": {
            "type": "string",
            "enum": ["read", "write", "list_dir", "exists", "mkdir"],
            "description": "The file operation to perform",
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "Target file or directory path",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "Content to write (only for 'write' operation)",
            "required": False,
        },
    }

    def __init__(
        self,
        allowed_dirs: Optional[Sequence[Union[str, Path]]] = None,
        max_file_size_mb: int = 50,
    ) -> None:
        self.allowed_dirs: List[Path] = [
            Path(d).expanduser().resolve() for d in (allowed_dirs or [])
        ]
        self.max_file_size_mb: int = max(1, int(max_file_size_mb))
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, **params: Any) -> Any:
        """Execute a file operation.

        Args:
            **params: Keyword arguments matching :attr:`parameter_schema`.
                Must include ``operation`` and ``path``.

        Returns:
            The result of the operation (file contents, listing, boolean,
            or ``None`` for write/mkdir).
        """
        operation = params.get("operation")
        path = params.get("path")

        if not operation:
            raise ValueError("Missing required parameter: 'operation'.")
        if not path:
            raise ValueError("Missing required parameter: 'path'.")

        if operation == "read":
            return self.read(path)
        if operation == "write":
            content = params.get("content", "")
            return self.write(path, content)
        if operation == "list_dir":
            return self.list_dir(path)
        if operation == "exists":
            return self.exists(path)
        if operation == "mkdir":
            return self.mkdir(path)

        raise ValueError(
            f"Unknown operation: '{operation}'. "
            f"Supported: read, write, list_dir, exists, mkdir."
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def read(self, path: Union[str, Path]) -> str:
        """Read the text contents of a file.

        Args:
            path: Path to the file.

        Returns:
            The file contents as a string.
        """
        resolved = self._validate_path(path, must_exist=True)
        self._check_file_size(resolved)
        with open(resolved, "r", encoding="utf-8") as handle:
            return handle.read()

    def write(self, path: Union[str, Path], content: str) -> str:
        """Write text content to a file.

        Args:
            path: Path to the file.
            content: The text to write.

        Returns:
            A confirmation message with the number of bytes written.
        """
        if not isinstance(content, str):
            raise TypeError("'content' must be a string.")

        resolved = self._validate_path(path)
        # Enforce the size limit on the content being written.
        size = len(content.encode("utf-8"))
        if size > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(
                f"Content size ({size} bytes) exceeds the maximum "
                f"of {self.max_file_size_mb} MB."
            )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as handle:
            handle.write(content)
        return f"Wrote {size} bytes to {resolved}"

    def list_dir(self, path: Union[str, Path]) -> List[str]:
        """List the entries in a directory.

        Args:
            path: Path to the directory.

        Returns:
            A sorted list of entry names.
        """
        resolved = self._validate_path(path, must_exist=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {resolved}")
        return sorted(entry.name for entry in resolved.iterdir())

    def exists(self, path: Union[str, Path]) -> bool:
        """Check whether a path exists.

        Args:
            path: Path to check.

        Returns:
            ``True`` if the path exists, ``False`` otherwise.
        """
        resolved = self._validate_path(path)
        return resolved.exists()

    def mkdir(self, path: Union[str, Path]) -> str:
        """Create a directory (and any missing parents).

        Args:
            path: Path to the directory to create.

        Returns:
            A confirmation message.
        """
        resolved = self._validate_path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return f"Created directory {resolved}"

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------
    def _validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
    ) -> Path:
        """Validate and resolve a path against the security policy.

        Args:
            path: The raw path provided by the caller.
            must_exist: When ``True`` raise ``FileNotFoundError`` if the
                path does not exist.

        Returns:
            The resolved absolute :class:`~pathlib.Path`.

        Raises:
            ValueError: If the path is blocked or outside the allowed
                directories.
            FileNotFoundError: If ``must_exist`` is ``True`` and the path
                does not exist.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or Path object.")

        raw = Path(path).expanduser()

        # Reject obviously absolute system paths before resolution.
        resolved = self._safe_resolve(raw)

        # Check against the block-list.
        resolved_str = str(resolved)
        for blocked in _BLOCKED_DIRS:
            if resolved_str == blocked or resolved_str.startswith(blocked + os.sep):
                raise ValueError(
                    f"Access to system directory '{blocked}' is forbidden."
                )

        # When an allow-list is configured, enforce confinement.
        if self.allowed_dirs:
            if not any(
                self._is_within(resolved, allowed)
                for allowed in self.allowed_dirs
            ):
                raise ValueError(
                    f"Path '{resolved}' is outside the permitted "
                    f"directories: {[str(d) for d in self.allowed_dirs]}."
                )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")

        return resolved

    @staticmethod
    def _safe_resolve(path: Path) -> Path:
        """Resolve a path safely, handling broken symlinks.

        Uses ``resolve(strict=False)`` so that non-existent paths are
        still normalised.  Symlinks that point outside the allowed area
        are caught by the subsequent block-list / allow-list checks
        because ``resolve`` follows them.
        """
        try:
            return path.resolve(strict=False)
        except (OSError, RuntimeError):
            # Fallback: normalise without following symlinks.
            return path.absolute()

    @staticmethod
    def _is_within(path: Path, base: Path) -> bool:
        """Return ``True`` if ``path`` is inside ``base``."""
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    def _check_file_size(self, path: Path) -> None:
        """Ensure a file is within the configured size limit.

        Args:
            path: Path to the file.

        Raises:
            ValueError: If the file exceeds the maximum size.
        """
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise OSError(f"Cannot stat file '{path}': {exc}") from exc
        if size > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(
                f"File size ({size} bytes) exceeds the maximum "
                f"of {self.max_file_size_mb} MB."
            )
