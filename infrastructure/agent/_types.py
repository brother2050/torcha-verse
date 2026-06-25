"""Data classes for the v0.6.x :class:`AgentBus`.

The :mod:`infrastructure.agent` sub-package is split into four
focused modules.  This one contains the *plain data* that
:mod:`._registry` and :mod:`._bus` operate on:

* :class:`ToolSpec` -- a callable tool exposed to the agent,
  with parameter type strings (free-form documentation) and
  a function to invoke.
* :class:`ToolResult` -- the outcome of a single tool
  invocation, capturing both success and error states.
* :class:`AgentRunResult` -- the full transcript of a single
  :meth:`AgentBus.run` call, including the ordered step list
  and the final answer.

These dataclasses are deliberately tiny and side-effect free
so they can be safely JSON-serialised by the
``/serving/v1/agent/run`` HTTP endpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["ToolSpec", "ToolResult", "AgentRunResult"]


#: Regex used to validate a tool name (``[A-Za-z_][A-Za-z0-9_]*``).
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ToolSpec:
    """A callable tool exposed to the agent.

    Attributes:
        name: Stable tool name (e.g. ``"rag_query"``).  Must be a
            non-empty identifier; used as the JSON key in the
            model's tool_call output.
        description: One-line human-readable description of what
            the tool does.  Surfaced verbatim to the model so it
            can decide whether to call the tool.
        parameters: Mapping of parameter name -> type string
            (``"str"``, ``"int"``, ``"float"``, ``"bool"``,
            ``"json"``).  Free-form documentation; the runtime
            does not validate types beyond the JSON parsing.
        func: The Python callable to invoke.  Receives the
            parameters as keyword arguments and may return any
            JSON-serialisable value.
        tags: Free-form tag list (e.g. ``["rag", "read"]``).
    """

    name: str
    description: str
    parameters: Dict[str, str]
    func: Callable[..., Any]
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolSpec.name must be a non-empty string.")
        if not _TOOL_NAME_RE.match(self.name):
            raise ValueError(
                f"ToolSpec.name must match [A-Za-z_][A-Za-z0-9_]*; "
                f"got {self.name!r}."
            )
        if not isinstance(self.description, str):
            raise ValueError("ToolSpec.description must be a string.")
        if not isinstance(self.parameters, dict):
            raise ValueError("ToolSpec.parameters must be a dict[str, str].")
        for k, v in self.parameters.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    f"ToolSpec parameter name must be a non-empty str: {k!r}"
                )
            if not isinstance(v, str):
                raise ValueError(
                    f"ToolSpec parameter {k!r} type must be a string; "
                    f"got {type(v).__name__}."
                )
        if not callable(self.func):
            raise TypeError(
                f"ToolSpec.func must be callable; got {type(self.func).__name__}."
            )


@dataclass
class ToolResult:
    """The outcome of a single tool invocation.

    Attributes:
        name: The tool name.
        ok: ``True`` when the tool returned without raising.
        output: The tool's return value (JSON-serialisable).
        error: When ``ok`` is ``False``, a human-readable error
            message (never raises out of the agent loop).
    """

    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None


@dataclass
class AgentRunResult:
    """The full transcript of a single :meth:`AgentBus.run` call.

    Attributes:
        query: The original user query.
        final_answer: The agent's final natural-language answer
            (string), or the last ``Observation`` text when the
            loop ran out of steps.
        steps: Ordered list of ``(thought, tool_call, observation)``
            tuples the agent emitted during the run.
        iterations: Number of LLM + tool rounds the loop took.
        ok: ``True`` when the agent emitted a final answer;
            ``False`` when the loop terminated because
            ``max_steps`` or ``max_parse_failures`` was reached.
    """

    query: str
    final_answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    ok: bool = True
