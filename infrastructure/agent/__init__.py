"""Lightweight tool-calling agent for the v0.4.x single-system stack.

The :class:`AgentBus` is a small **dependency-free** tool
registry and ReAct-style executor that lets the ``agent_*`` L4
nodes and the ``/serving/v1/agent/*`` HTTP endpoints answer a
free-form query by orchestrating the framework's existing
capabilities.

Design choices
--------------

* **No external agent framework** (no LangChain, no
  LlamaIndex).  The agent is a small, well-tested loop the
  team can read end-to-end in 200 lines -- fits the v0.4.x
  "single system" roadmap.
* **Tool registry** is built on the existing
  :class:`core.module_bus.ModuleBus` and
  :class:`nodes.NodeRegistry` so every registered L4 node and
  every explicitly registered Python function is callable as
  a tool.
* **ReAct-style prompting**: each step the agent emits either
  *thought* (text), a *tool_call* (JSON object with ``name``
  / ``args``) or a *final_answer* marker.  The loop
  terminates when ``final_answer`` is emitted, ``max_steps``
  is reached, or the model produces unparseable output for
  ``max_parse_failures`` consecutive times.
* **Audit + budget** -- every tool invocation is logged
  through :class:`AuditLogger` and respects the agent's
  :class:`ResourceBudget` so a runaway agent cannot exhaust
  the vram/ram allotment.
* **Deterministic** when ``seed`` is supplied: the inner
  :class:`LLMProvider` is asked to use ``do_sample=False`` so
  CI can run an end-to-end agent test.

The v0.6.x refactor splits the previous single-file
``infrastructure/agent.py`` (692 lines) into four focused
modules:

* :mod:`infrastructure.agent._types` -- :class:`ToolSpec` /
  :class:`ToolResult` / :class:`AgentRunResult`.
* :mod:`infrastructure.agent._parse` -- the three compiled
  ReAct regexes + :func:`coerce_value` + :func:`parse_action_args`.
* :mod:`infrastructure.agent._registry` -- the
  :class:`ToolRegistry`.
* :mod:`infrastructure.agent._bus` -- the :class:`AgentBus`
  core + :func:`default_agent_bus` /
  :func:`reset_default_agent_bus`.

The public API is unchanged --
``from infrastructure.agent import AgentBus, ToolSpec,
ToolResult, AgentRunResult, ToolRegistry, default_agent_bus``
keeps working.
"""

from __future__ import annotations

from ._bus import AgentBus, default_agent_bus, reset_default_agent_bus
from ._registry import ToolRegistry
from ._types import AgentRunResult, ToolResult, ToolSpec

__all__ = [
    "AgentBus",
    "ToolSpec",
    "ToolResult",
    "AgentRunResult",
    "ToolRegistry",
    "default_agent_bus",
    "reset_default_agent_bus",
]
