"""Sandboxed Python code execution tool.

This module provides :class:`PythonExecutorTool`, a :class:`BaseTool`
implementation that runs arbitrary Python code in an isolated subprocess.

Security measures:

* The code is executed in a **separate Python process** spawned via
  :mod:`subprocess`, so a crash or infinite loop cannot take down the
  host application.
* A hard **timeout** (default 10 s) kills the subprocess if it runs too
  long.
* The subprocess inherits a restricted environment that attempts to block
  file writes, network access, and system-command execution by importing
  a sandbox preamble before running the user code.  The preamble
  neutralises dangerous builtins (``open``, ``__import__`` for blocked
  modules, ``exec``, ``eval``) and removes dangerous modules from
  :data:`sys.modules`.
* Resource limits (CPU time, memory) are applied via :mod:`resource`
  on POSIX platforms.

The tool captures ``stdout``, ``stderr``, and the return value printed
via a sentinel.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.tool_registry import BaseTool
from infrastructure.logger import get_logger

__all__ = ["PythonExecutorTool", "ExecutionResult"]

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT: int = 10
_DEFAULT_MEMORY_LIMIT_MB: int = 512

# Modules that are blocked from import inside the sandbox.
_BLOCKED_MODULES: tuple = (
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "paramiko",
    "subprocess",
    "os",
    "shutil",
    "ctypes",
    "multiprocessing",
    "pickle",
    "marshal",
    "webbrowser",
)

# Sentinel used to delimit the printed return value.
_RETURN_SENTINEL: str = "__TORCHAVERSE_RETURN_VALUE__"


# ---------------------------------------------------------------------------
# Sandbox preamble
# ---------------------------------------------------------------------------
# This code is prepended to the user's snippet.  It restricts the available
# builtins and blocks dangerous imports.  It is intentionally conservative;
# the subprocess boundary is the primary defence.
#
# NOTE: This is a *plain* string (not an f-string) because it contains many
# literal braces (dict literals, nested f-strings) that would otherwise be
# misinterpreted as format fields.  The two dynamic values are injected via
# ``str.replace`` below.
_SANDBOX_PREAMBLE: str = textwrap.dedent(
    """\
    import builtins as _builtins
    import sys as _sys
    import json as _json

    _BLOCKED = __BLOCKED_MODULES__

    # --- Block dangerous imports -----------------------------------------
    # NOTE: we deliberately do *not* keep a reference to the real
    # ``__import__`` in a module-level global.  Earlier versions exposed
    # ``_real_import`` here, which meant that any user code that managed
    # to read the preamble (e.g. via ``_real_import('os')``) could
    # trivially bypass the import blocklist.  Instead we install a
    # safe wrapper and immediately discard the original.
    def _make_safe_import(_real_import):
        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split('.')[0]
            if root in _BLOCKED:
                raise ImportError(
                    "Import of '" + root + "' is blocked in the sandbox."
                )
            return _real_import(name, globals, locals, fromlist, level)
        return _safe_import

    _builtins.__import__ = _make_safe_import(_builtins.__import__)

    # --- Remove already-imported dangerous modules ----------------------
    for _mod in list(_sys.modules):
        _root = _mod.split('.')[0]
        if _root in _BLOCKED:
            del _sys.modules[_mod]

    # --- Neutralise dangerous builtins ----------------------------------
    _builtins.open = lambda *a, **k: (_ for _ in ()).throw(
        PermissionError("open() is blocked in the sandbox.")
    )
    if hasattr(_builtins, "exec"):
        _builtins.exec = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("exec() is blocked in the sandbox.")
        )
    if hasattr(_builtins, "eval"):
        _builtins.eval = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("eval() is blocked in the sandbox.")
        )
    if hasattr(_builtins, "compile"):
        _builtins.compile = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("compile() is blocked in the sandbox.")
        )
    if hasattr(_builtins, "__build_class__"):
        _builtins.__build_class__ = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("class definition is blocked in the sandbox.")
        )
    if hasattr(_builtins, "breakpoint"):
        _builtins.breakpoint = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("breakpoint() is blocked in the sandbox.")
        )

    # --- Capture the return value of the user expression ----------------
    _SENTINEL = __RETURN_SENTINEL__

    def _emit_return(value):
        try:
            payload = _json.dumps({"value": value}, default=str)
        except Exception:
            payload = repr(value)
        print(_SENTINEL + payload)

    # Provide a helper the user can call to return a value explicitly.
    def return_value(value):
        _emit_return(value)

    _builtins.return_value = return_value

    # Defensive cleanup: explicitly delete every internal name we just
    # created so they cannot be re-imported / re-bound by user code.
    # ``_make_safe_import`` is intentionally referenced via ``_builtins``
    # so that the closure keeps the import wrapper alive even after the
    # factory itself goes out of scope.
    for _name in ("_make_safe_import", "_BLOCKED", "_emit_return",
                  "_SENTINEL", "_json", "_sys"):
        try:
            del globals()[_name]
        except KeyError:
            pass
    """
).replace("__BLOCKED_MODULES__", repr(_BLOCKED_MODULES)).replace(
    "__RETURN_SENTINEL__", repr(_RETURN_SENTINEL)
)


@dataclass
class ExecutionResult:
    """The outcome of a code execution.

    Attributes:
        success: Whether the code ran without errors.
        stdout: Captured standard output.
        stderr: Captured standard error.
        return_value: The value returned by the code (if any).
        exit_code: The subprocess exit code.
        timed_out: Whether the execution exceeded the timeout.
        error: Error message when ``success`` is ``False``.
        metadata: Additional execution metadata.
    """

    success: bool = True
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    exit_code: int = 0
    timed_out: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": self.return_value,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error": self.error,
            "metadata": self.metadata,
        }


class PythonExecutorTool(BaseTool):
    """Execute Python code in an isolated subprocess sandbox.

    The tool writes the user code (prepended with a sandbox preamble)
    to a temporary file and runs it with the same Python interpreter.
    ``stdout``, ``stderr``, and an optional return value are captured.

    Example::

        >>> tool = PythonExecutorTool()
        >>> result = tool.execute(code="print(1 + 2)")
        >>> result.stdout.strip()
        '3'
    """

    name: str = "python_executor"
    description: str = "Execute Python code and return the output"
    parameter_schema: Dict[str, Any] = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
            "required": True,
        }
    }

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    ) -> None:
        """Initialise the executor.

        Args:
            timeout: Maximum execution time in seconds.
            memory_limit_mb: Maximum virtual memory in megabytes
                (POSIX only).
        """
        self.timeout: int = max(1, int(timeout))
        self.memory_limit_mb: int = max(1, int(memory_limit_mb))
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, **params: Any) -> ExecutionResult:
        """Execute Python code in the sandbox.

        Args:
            **params: Keyword arguments matching :attr:`parameter_schema`.
                Must include ``code``.

        Returns:
            An :class:`ExecutionResult` with captured output.
        """
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult(
                success=False,
                error="Parameter 'code' must be a non-empty string.",
            )

        return self._run(code)

    def _run(self, code: str) -> ExecutionResult:
        """Run ``code`` in a subprocess and capture the output.

        Args:
            code: The Python source code to execute.

        Returns:
            The :class:`ExecutionResult`.
        """
        # Build the full script: preamble + user code + return capture.
        full_script = _SANDBOX_PREAMBLE + "\n" + code + "\n"

        # Write to a temporary file so line numbers in tracebacks are
        # relative to the user code (the preamble is small).
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".py", prefix="sandbox_", text=True
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(full_script)

            env = self._build_env()

            try:
                completed = subprocess.run(  # noqa: S603 - intentional
                    [sys.executable, "-I", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    cwd=tempfile.gettempdir(),
                    # Apply resource limits via preexec_fn on POSIX.
                    preexec_fn=self._set_resource_limits
                    if platform.system() != "Windows"
                    else None,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=-1,
                    timed_out=True,
                    error=f"Execution timed out after {self.timeout}s.",
                )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return_value = self._extract_return_value(stdout)
            stdout = self._strip_return_sentinel(stdout)

            success = completed.returncode == 0
            error: Optional[str] = None
            if not success:
                error = stderr.strip() or f"Process exited with code {completed.returncode}."

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                return_value=return_value,
                exit_code=completed.returncode,
                timed_out=False,
                error=error,
                metadata={"timeout": self.timeout, "memory_limit_mb": self.memory_limit_mb},
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                self._logger.debug("Python exec: tmp file cleanup %s failed: %s", tmp_path, exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _build_env() -> Dict[str, str]:
        """Build a restricted environment for the subprocess.

        Network-related environment variables are removed and
        ``PYTHONDONTWRITEBYTECODE`` is set to avoid polluting the
        filesystem with ``.pyc`` files.
        """
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.lower().startswith(("http_proxy", "https_proxy", "ftp_proxy"))
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        return env

    def _set_resource_limits(self) -> None:
        """Apply CPU and memory resource limits (POSIX only).

        Called as ``preexec_fn`` in the child process before ``exec``.
        """
        try:
            import resource

            # Limit CPU time (seconds).
            cpu_seconds = max(1, self.timeout)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            # Limit virtual memory.
            mem_bytes = self.memory_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError) as exc:
                # RLIMIT_AS may be unavailable on some platforms.
                self._logger.debug("RLIMIT_AS setrlimit failed (best-effort): %s", exc)
        except Exception as exc:
            # Resource limits are best-effort; never crash the child.
            self._logger.debug("Resource limit setup failed: %s", exc)

    @staticmethod
    def _extract_return_value(stdout: str) -> Any:
        """Extract the return value printed with the sentinel.

        Args:
            stdout: The captured standard output.

        Returns:
            The parsed return value, or ``None`` if no value was emitted.
        """
        import json

        for line in stdout.splitlines():
            if line.startswith(_RETURN_SENTINEL):
                payload = line[len(_RETURN_SENTINEL):]
                try:
                    data = json.loads(payload)
                    return data.get("value")
                except (json.JSONDecodeError, ValueError):
                    return payload
        return None

    @staticmethod
    def _strip_return_sentinel(stdout: str) -> str:
        """Remove the sentinel line from the captured stdout."""
        lines = stdout.splitlines(keepends=True)
        cleaned = [
            line for line in lines
            if not line.startswith(_RETURN_SENTINEL)
        ]
        return "".join(cleaned)
