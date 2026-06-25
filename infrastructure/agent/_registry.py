"""The :class:`ToolRegistry` for the v0.6.x agent sub-package.

A :class:`ToolRegistry` is a small, thread-safe map from tool
name to :class:`ToolSpec`.  It is intentionally tiny so the
:class:`AgentBus` core (in :mod:`._bus`) can focus on the
ReAct loop and the LLM-driven orchestration, while the
registry stays trivially testable in isolation.

The registry is the *only* component that invokes user
functions -- the bus calls :meth:`ToolRegistry.invoke` and
gets back a :class:`ToolResult` that captures both success
and error states.  This keeps exception handling out of the
bus loop.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from ..logger import get_logger
from ._types import ToolResult, ToolSpec

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """An in-process map from tool name to :class:`ToolSpec`."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.agent.tools")

    def register(self, spec: ToolSpec) -> None:
        """Register ``spec`` (raises on duplicate)."""
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"Tool {spec.name!r} already registered.")
            self._tools[spec.name] = spec
        self._logger.info("Registered tool %s", spec.name)

    def try_register(self, spec: ToolSpec) -> bool:
        """Register if absent; return ``True`` if registered, ``False`` if duplicate."""
        with self._lock:
            if spec.name in self._tools:
                return False
            self._tools[spec.name] = spec
            return True

    def unregister(self, name: str) -> bool:
        """Remove a tool by name; return ``True`` if it was present."""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec:
        """Return the :class:`ToolSpec` for ``name`` (raises :class:`KeyError`)."""
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as exc:
                raise KeyError(f"no tool named {name!r}") from exc

    def try_get(self, name: str) -> Optional[ToolSpec]:
        """Return the :class:`ToolSpec` for ``name`` or ``None``."""
        with self._lock:
            return self._tools.get(name)

    def list(self) -> List[str]:
        """Return the registered tool names, sorted alphabetically."""
        with self._lock:
            return sorted(self._tools.keys())

    def describe(self) -> List[Dict[str, Any]]:
        """Return a JSON-serialisable description of every tool.

        Used by the agent prompt so the LLM can pick a tool by
        name and reason about its parameters.
        """
        with self._lock:
            return [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.parameters),
                    "tags": list(spec.tags),
                }
                for spec in self._tools.values()
            ]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Invoke the tool ``name`` with ``**kwargs`` and return a :class:`ToolResult`.

        Exceptions raised by the tool are caught and surfaced
        as a failed :class:`ToolResult` -- the bus loop never
        has to handle raw exceptions.
        """
        spec = self.try_get(name)
        if spec is None:
            return ToolResult(name=name, ok=False, error=f"unknown tool: {name!r}")
        try:
            output = spec.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Tool %s raised %s: %s", name, type(exc).__name__, exc,
            )
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(name=name, ok=True, output=output)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.list()

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)
