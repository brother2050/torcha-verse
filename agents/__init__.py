"""Agent subsystem for TorchaVerse.

This package provides the agent infrastructure:

* :mod:`agents.base_agent` -- :class:`BaseAgent`, memory, planner, reflector.
* :mod:`agents.react_agent` -- :class:`ReActAgent` (ReAct loop).
* :mod:`agents.tool_call_agent` -- :class:`ToolCallAgent` (native
  function calling).
* :mod:`agents.flows` -- multi-agent coordination patterns
  (:class:`SequentialFlow`, :class:`HierarchicalFlow`,
  :class:`DebateFlow`, :class:`FlowOrchestrator`).
"""

from __future__ import annotations

from .base_agent import (
    BaseAgent,
    LongTermMemory,
    Memory,
    Planner,
    Reflector,
    Result,
    ShortTermMemory,
    Step,
    WorkingMemory,
)
from .flows import (
    DebateFlow,
    Flow,
    FlowOrchestrator,
    HierarchicalFlow,
    SequentialFlow,
)
from .react_agent import ReActAgent
from .tool_call_agent import ToolCallAgent

__all__ = [
    # base
    "BaseAgent",
    "Step",
    "Result",
    "Memory",
    "ShortTermMemory",
    "LongTermMemory",
    "WorkingMemory",
    "Planner",
    "Reflector",
    # agents
    "ReActAgent",
    "ToolCallAgent",
    # flows
    "SequentialFlow",
    "HierarchicalFlow",
    "DebateFlow",
    "Flow",
    "FlowOrchestrator",
]
