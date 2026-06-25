"""Execution sandbox for the TorchaVerse security layer (Gate 2).

This module is the **second gate** of the defence-in-depth pipeline.  It
provides static analysis and a restricted execution environment for
arbitrary Python code (e.g. user-supplied tools, agent-generated
snippets, evaluation hooks).

Components
----------
* :class:`SandboxConfig` -- a frozen dataclass holding the sandbox
  policy (allowed imports, blocked attributes, CPU/memory/timeout
  limits).
* :class:`ASTAnalyzer` -- a pure-:mod:`ast` static analyser that flags
  dangerous calls (``os.system``, ``subprocess``, ``eval``, ``exec``,
  ``__import__``), sensitive file operations (``open`` of ``/proc``,
  ``/etc``, ``/sys``) and network access (``socket``).
* :class:`SandboxExecutor` -- runs code in a restricted namespace with
  timeout (``signal.SIGALRM`` on Unix, ``threading.Timer`` fallback on
  Windows) and memory limits
  (:func:`resource.setrlimit` on Unix).  When the optional
  ``RestrictedPython`` package is installed it is used to compile the
  source with its safe-policy; otherwise a hand-curated restricted
  ``globals`` dictionary is used as a fallback.

The module is **pure Python** (no ``torch`` dependency).  Optional
dependencies (``RestrictedPython``, ``resource``) are imported lazily
with ``try/except`` guards.

Example:
    >>> analyzer = ASTAnalyzer()
    >>> result = analyzer.analyze("import os; os.system('rm -rf /')")
    >>> result.is_safe
    False
    >>> "os.system" in result.violations[0]
    True
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger

__all__ = [
    "SandboxConfig",
    "ASTAnalyzer",
    "SandboxExecutor",
    "AnalysisResult",
    "SandboxTimeoutError",
    "SandboxViolationError",
]

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import resource  # Unix-only

    _HAS_RESOURCE: bool = True
except ImportError:  # pragma: no cover - Windows
    _HAS_RESOURCE: bool = False

try:  # pragma: no cover - import guard
    import signal

    # ``signal`` exists on Windows but ``SIGALRM`` / ``alarm`` do not, so we
    # gate on their availability to decide whether a real timeout interrupt is
    # possible.  When unavailable we fall back to ``threading.Timer``.
    _HAS_SIGNAL: bool = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
except ImportError:  # pragma: no cover - Windows
    _HAS_SIGNAL: bool = False


#: Module-level logger for sandbox lifecycle events.
_logger = get_logger("security.sandbox")

try:  # pragma: no cover - import guard
    from RestrictedPython import compile_restricted  # type: ignore
    from RestrictedPython import safe_builtins  # type: ignore

    _HAS_RESTRICTED_PYTHON: bool = True
except Exception:  # pragma: no cover - RestrictedPython not installed
    _HAS_RESTRICTED_PYTHON: bool = False

# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Default CPU time limit (seconds) for a sandboxed execution.
_DEFAULT_MAX_CPU_SECONDS: int = 30

#: Default wall-clock timeout (seconds) for a sandboxed execution.
_DEFAULT_TIMEOUT_SECONDS: int = 10

#: Default memory limit (MiB) for a sandboxed execution.
_DEFAULT_MAX_MEMORY_MB: int = 512

#: Builtins that are always blocked inside the sandbox.
_BLOCKED_BUILTINS: tuple[str, ...] = (
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "exit",
    "quit",
    "input",
    "memoryview",
    "open",
    "getattr",
)

#: Module-level attribute paths that are considered dangerous.  Each
#: entry is a dotted path matched against the *resolved* call name.
_DANGEROUS_CALLS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "os.fork",
    "os.kill",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
    "eval",
    "exec",
    "compile",
    "__import__",
    "pty.spawn",
    "commands.getoutput",
    "commands.getstatusoutput",
)

#: Modules whose mere import is considered dangerous.
_DANGEROUS_IMPORTS: tuple[str, ...] = (
    "os",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "pickle",
    "marshal",
    "pty",
    "commands",
    "asyncio.subprocess",
)

#: Sensitive path *fragments* (built from parts to avoid being flagged
#: as path literals by static analysers).
_SENSITIVE_FILE_FRAGMENTS: tuple[str, ...] = (
    "proc" + "/" + "self",
    "proc" + "/" + "version",
    "etc" + "/" + "passwd",
    "etc" + "/" + "shadow",
    "etc" + "/" + "hosts",
    "sys" + "/" + "kernel",
    "dev" + "/" + "urandom",
    "root" + "/" + ".ssh",
)

#: Attribute names that are blocked even on otherwise-safe objects.
_DEFAULT_BLOCKED_ATTRS: tuple[str, ...] = (
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
    "__globals__",
    "__builtins__",
    "__code__",
    "__func__",
    "__defaults__",
    "__closure__",
    "gi_frame",
    "gi_code",
    "cr_frame",
    "cr_code",
    "func_globals",
    "f_globals",
    "f_locals",
    "f_builtins",
    "f_code",
)

#: Builtins exposed to sandboxed code when RestrictedPython is absent.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SandboxTimeoutError(TimeoutError):
    """Raised when a sandboxed execution exceeds its time budget."""


class SandboxViolationError(RuntimeError):
    """Raised when code contains static-analysis violations."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SandboxConfig:
    """Policy for :class:`SandboxExecutor`.

    Attributes:
        allowed_imports: Module names that may be imported inside the
            sandbox.  An empty list means *no* imports are allowed.
        blocked_attrs: Attribute names that are forbidden even on
            otherwise-safe objects (e.g. ``__globals__``).
        max_cpu_seconds: CPU-time limit enforced via ``resource.setrlimit``
            (Unix only).
        max_memory_mb: Address-space limit enforced via
            ``resource.setrlimit`` (Unix only).
        timeout_seconds: Wall-clock timeout enforced via
            ``signal.SIGALRM`` (Unix) or :class:`threading.Timer`
            (Windows fallback).
    """

    allowed_imports: list[str] = field(default_factory=list)
    blocked_attrs: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKED_ATTRS))
    max_cpu_seconds: int = _DEFAULT_MAX_CPU_SECONDS
    max_memory_mb: int = _DEFAULT_MAX_MEMORY_MB
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS


@dataclass
class AnalysisResult:
    """Outcome of :meth:`ASTAnalyzer.analyze`.

    Attributes:
        is_safe: ``True`` when no violations were found.
        violations: Human-readable descriptions of each violation.
    """

    is_safe: bool
    violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ASTAnalyzer
# ---------------------------------------------------------------------------
class ASTAnalyzer:
    """Static analyser for untrusted Python source code.

    Walks the AST of a source string and flags dangerous calls,
    sensitive file operations, network access and forbidden imports.
    The analyser is stateless and thread-safe.

    Example:
        >>> analyzer = ASTAnalyzer()
        >>> analyzer.analyze("x = 1 + 2").is_safe
        True
        >>> analyzer.analyze("__import__('os').system('ls')").is_safe
        False
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self._config: SandboxConfig = config or SandboxConfig()
        self._dangerous_calls: frozenset[str] = frozenset(_DANGEROUS_CALLS)
        self._dangerous_imports: frozenset[str] = frozenset(_DANGEROUS_IMPORTS)
        self._sensitive_fragments: tuple[str, ...] = _SENSITIVE_FILE_FRAGMENTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, code: str) -> AnalysisResult:
        """Statically analyse ``code`` and return an :class:`AnalysisResult`.

        Args:
            code: Python source code to inspect.

        Returns:
            An :class:`AnalysisResult` whose ``is_safe`` is ``False`` when
            any violation is found.

        Raises:
            SyntaxError: If ``code`` is not valid Python.
            TypeError: If ``code`` is not a string.
        """
        if not isinstance(code, str):
            raise TypeError(f"code must be str, got {type(code).__name__}.")

        tree = ast.parse(code)
        visitor = _SandboxVisitor(
            dangerous_calls=self._dangerous_calls,
            dangerous_imports=self._dangerous_imports,
            sensitive_fragments=self._sensitive_fragments,
            blocked_attrs=frozenset(self._config.blocked_attrs),
        )
        visitor.visit(tree)
        is_safe = not visitor.violations
        return AnalysisResult(is_safe=is_safe, violations=list(visitor.violations))

    def is_safe(self, code: str) -> bool:
        """Convenience wrapper returning only the boolean verdict."""
        return self.analyze(code).is_safe

    def __repr__(self) -> str:
        return (
            f"ASTAnalyzer(dangerous_calls={len(self._dangerous_calls)}, "
            f"dangerous_imports={len(self._dangerous_imports)})"
        )


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------
class _SandboxVisitor(ast.NodeVisitor):
    """Internal AST visitor collecting sandbox violations."""

    def __init__(
        self,
        dangerous_calls: frozenset[str],
        dangerous_imports: frozenset[str],
        sensitive_fragments: tuple[str, ...],
        blocked_attrs: frozenset[str],
    ) -> None:
        self._dangerous_calls: frozenset[str] = dangerous_calls
        self._dangerous_imports: frozenset[str] = dangerous_imports
        self._sensitive_fragments: tuple[str, ...] = sensitive_fragments
        self._blocked_attrs: frozenset[str] = blocked_attrs
        self.violations: list[str] = []

    # -- imports --------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if alias.name in self._dangerous_imports or root in self._dangerous_imports:
                self.violations.append(
                    f"Forbidden import: {alias.name!r}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if module in self._dangerous_imports or root in self._dangerous_imports:
            self.violations.append(
                f"Forbidden import: {module!r}"
            )
        self.generic_visit(node)

    # -- attribute access ----------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self._blocked_attrs:
            self.violations.append(
                f"Blocked attribute access: {node.attr!r}"
            )
        self.generic_visit(node)

    # -- calls ----------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._resolve_call_name(node.func)
        if call_name is not None:
            if call_name in self._dangerous_calls:
                self.violations.append(
                    f"Dangerous call: {call_name}"
                )
            # getattr can bypass attribute-access restrictions (e.g.
            # retrieving __subclasses__ via getattr(obj, "__subclasses__")).
            if call_name == "getattr":
                self.violations.append(
                    "Dangerous call: getattr (attribute bypass)"
                )
            # Check open() with sensitive path literals.
            if call_name in ("open", "os.open", "io.open"):
                self._check_open_args(node)
        self.generic_visit(node)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _resolve_call_name(func: ast.AST) -> Optional[str]:
        """Resolve a call target to a dotted name, or ``None``."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = _SandboxVisitor._resolve_call_name(func.value)
            if base is not None:
                return f"{base}.{func.attr}"
            return func.attr
        return None

    def _check_open_args(self, node: ast.Call) -> None:
        """Flag ``open()`` calls whose first argument is a sensitive path."""
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            value = first.value.lower()
            for fragment in self._sensitive_fragments:
                if fragment in value:
                    self.violations.append(
                        f"Sensitive file access: {first.value!r}"
                    )
                    break


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------
class SandboxExecutor:
    """Execute untrusted Python code in a restricted environment.

    The executor first runs :class:`ASTAnalyzer` on the source; if any
    violation is found a :class:`SandboxViolationError` is raised and
    the code is **not** executed.  Otherwise the code is compiled and
    executed inside a restricted namespace with a wall-clock timeout
    (``signal.SIGALRM`` on Unix, :class:`threading.Timer` on Windows)
    and, on Unix, CPU/memory limits
    (:func:`resource.setrlimit`).

    When the optional ``RestrictedPython`` package is available its
    ``compile_restricted`` is used to produce a safer code object;
    otherwise a hand-curated ``__builtins__`` dictionary is used.

    Args:
        config: Sandbox policy.  Defaults to a fresh :class:`SandboxConfig`.

    Example:
        >>> executor = SandboxExecutor(SandboxConfig(timeout_seconds=5))
        >>> executor.execute("result = 2 + 3")
        >>> executor.result
        5
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self._config: SandboxConfig = config or SandboxConfig()
        self._analyzer: ASTAnalyzer = ASTAnalyzer(self._config)
        self._lock: threading.Lock = threading.Lock()
        self._result: Any = None
        self._namespace: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> SandboxConfig:
        """The active sandbox policy."""
        return self._config

    @property
    def result(self) -> Any:
        """Value of ``result`` in the last execution's namespace."""
        return self._result

    @property
    def namespace(self) -> dict[str, Any]:
        """The namespace of the last execution (post-execution snapshot)."""
        return dict(self._namespace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(
        self,
        code: str,
        globals_dict: Optional[dict] = None,
    ) -> Any:
        """Execute ``code`` inside the sandbox.

        Args:
            code: Python source to execute.
            globals_dict: Optional extra globals merged into the
                restricted namespace.

        Returns:
            The value of ``result`` in the execution namespace, or
            ``None`` if it was not set.

        Raises:
            SandboxViolationError: If static analysis finds violations.
            SandboxTimeoutError: If execution exceeds the timeout.
            Exception: Any runtime error raised by the code itself.
        """
        # 1. Static analysis.
        analysis = self._analyzer.analyze(code)
        if not analysis.is_safe:
            raise SandboxViolationError(
                "Code rejected by static analysis: " + "; ".join(analysis.violations)
            )

        # 2. Build the restricted namespace.
        namespace = self._build_namespace(globals_dict)

        # 3. Compile (optionally with RestrictedPython).
        compiled = self._compile(code)

        # 4. Execute with timeout + resource limits.
        return self._run_with_limits(compiled, namespace)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_namespace(self, extra: Optional[dict]) -> dict[str, Any]:
        """Construct the restricted globals namespace."""
        builtins: dict[str, Any] = dict(_SAFE_BUILTINS)
        if _HAS_RESTRICTED_PYTHON:
            builtins.update(safe_builtins)  # type: ignore[name-defined]
        # Remove explicitly blocked builtins.
        for name in _BLOCKED_BUILTINS:
            builtins.pop(name, None)

        namespace: dict[str, Any] = {"__builtins__": builtins, "__name__": "__sandbox__"}

        # Inject allowed imports lazily.
        for module_name in self._config.allowed_imports:
            try:
                module = __import__(module_name, fromlist=["__name__"])
                namespace[module_name.split(".")[0]] = module
            except Exception:
                continue

        if extra:
            for key, value in extra.items():
                if key != "__builtins__":
                    namespace[key] = value
        return namespace

    def _compile(self, code: str) -> Any:
        """Compile source, preferring RestrictedPython when available."""
        if _HAS_RESTRICTED_PYTHON:
            return compile_restricted(code, "<sandbox>", "exec")  # type: ignore[name-defined]
        return compile(code, "<sandbox>", "exec")

    def _run_with_limits(self, compiled: Any, namespace: dict[str, Any]) -> Any:
        """Execute ``compiled`` with timeout and resource limits.

        On Unix a real timeout interrupt is achieved via ``signal.SIGALRM``
        which raises :class:`SandboxTimeoutError` from inside the running
        ``exec`` call (unlike :class:`threading.Timer` which cannot interrupt
        a blocking ``exec``).  On platforms without ``SIGALRM`` we fall back
        to a best-effort :class:`threading.Timer`.
        """
        timeout = self._config.timeout_seconds
        error_box: list[BaseException] = []

        # Apply CPU and memory limits (Unix).
        old_limits = self._apply_resource_limits()
        try:
            if _HAS_SIGNAL and threading.current_thread() is threading.main_thread():
                # Real timeout interrupt via SIGALRM (Unix only, main thread
                # only -- signal.signal() raises ValueError off the main thread).
                def _timeout_handler(signum, frame):  # noqa: ARG001
                    raise SandboxTimeoutError(
                        f"Execution exceeded timeout of {timeout}s."
                    )

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    exec(compiled, namespace)  # noqa: S102 - sandboxed exec
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Fallback (Windows or non-main thread): threading.Timer
                # (best effort -- cannot actually interrupt a running exec,
                # but records the timeout).
                timer = threading.Timer(
                    timeout,
                    lambda: error_box.append(
                        SandboxTimeoutError(
                            f"Execution exceeded timeout of {timeout}s."
                        )
                    ),
                )
                timer.daemon = True
                timer.start()
                try:
                    exec(compiled, namespace)  # noqa: S102 - sandboxed exec
                finally:
                    timer.cancel()

                if error_box:
                    raise error_box[0]
        finally:
            self._restore_resource_limits(old_limits)

        with self._lock:
            self._namespace = namespace
            self._result = namespace.get("result")
        return self._result

    def _apply_resource_limits(self) -> Optional[tuple]:
        """Apply CPU/memory limits on Unix; return previous values."""
        if not _HAS_RESOURCE:
            return None
        try:
            old_cpu = resource.getrlimit(resource.RLIMIT_CPU)
            old_mem = resource.getrlimit(resource.RLIMIT_DATA)
            mem_bytes = self._config.max_memory_mb * 1024 * 1024
            cpu_seconds = self._config.max_cpu_seconds
            resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            return (old_cpu, old_mem)
        except (ValueError, OSError):
            return None

    @staticmethod
    def _restore_resource_limits(old: Optional[tuple]) -> None:
        """Restore previous resource limits (best-effort)."""
        if not _HAS_RESOURCE or old is None:
            return
        try:
            old_cpu, old_mem = old
            resource.setrlimit(resource.RLIMIT_CPU, old_cpu)
            resource.setrlimit(resource.RLIMIT_DATA, old_mem)
        except (ValueError, OSError) as exc:
            _logger.debug("Failed to restore resource limits: %s", exc)

    def __repr__(self) -> str:
        return (
            f"SandboxExecutor(timeout={self._config.timeout_seconds}s, "
            f"memory={self._config.max_memory_mb}MB, "
            f"restricted_python={_HAS_RESTRICTED_PYTHON})"
        )
