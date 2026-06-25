"""L4 nodes for the tool-calling agent stack.

Two nodes are exposed:

* :class:`AgentRunNode` (``agent_run``) -- run a single
  ReAct-style agent loop for a free-form query.
* :class:`AgentListToolsNode` (``agent_list_tools``) -- return
  the names of every tool currently registered with the
  default :class:`infrastructure.agent.AgentBus`.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseNode, NodeContext, NodeSpec, register_node

__all__ = ["AgentRunNode", "AgentListToolsNode"]


@register_node("agent_run")
class AgentRunNode(BaseNode):
    """Run a ReAct-style agent loop for a free-form query.

    Inputs:
        ``query`` (str, required).
        ``max_steps`` (int, optional): override the bus default.
        ``temperature`` (float, optional): sampling temperature.

    Returns:
        A dict with ``"query"``, ``"final_answer"``, ``"steps"``
        and ``"ok"``.
    """

    spec: NodeSpec = NodeSpec(
        type="agent_run",
        name="Agent Run",
        description="Run a ReAct-style tool-calling agent loop and return the final answer + transcript.",
        inputs={
            "query": "TEXT",
            "max_steps": "Optional[INT]",
            "temperature": "Optional[FLOAT]",
        },
        outputs={
            "query": "TEXT",
            "final_answer": "TEXT",
            "steps": "JSON",
            "iterations": "INT",
            "ok": "BOOL",
        },
        tags=["agent", "tool"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.agent import default_agent_bus

        query = str(inputs.get("query", ""))
        if not query.strip():
            raise ValueError("agent_run requires a non-empty `query`.")
        max_steps_raw = inputs.get("max_steps")
        temperature_raw = inputs.get("temperature")
        max_steps = int(max_steps_raw) if max_steps_raw is not None else None
        temperature = float(temperature_raw) if temperature_raw is not None else 0.0

        bus = default_agent_bus()
        if ctx.audit is not None:
            ctx.audit.log(
                "AGENT_RUN",
                actor="node.agent_run",
                action="run",
                resource_id="agent",
                details={"max_steps": max_steps, "temperature": temperature},
                severity="info",
            )
        result = bus.run(query, max_steps=max_steps, temperature=temperature)
        return {
            "query": result.query,
            "final_answer": result.final_answer,
            "steps": result.steps,
            "iterations": result.iterations,
            "ok": result.ok,
        }


@register_node("agent_list_tools")
class AgentListToolsNode(BaseNode):
    """Return every tool registered with the default agent bus.

    Inputs: none.

    Returns:
        A dict with ``"tools"`` (sorted list of names) and
        ``"descriptions"`` (the structured description used by
        the agent prompt).
    """

    spec: NodeSpec = NodeSpec(
        type="agent_list_tools",
        name="Agent List Tools",
        description="Return the names and structured descriptions of all agent tools.",
        inputs={},
        outputs={"tools": "JSON", "descriptions": "JSON"},
        tags=["agent", "metadata"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.agent import default_agent_bus

        bus = default_agent_bus()
        return {
            "tools": bus.tools.list(),
            "descriptions": bus.tools.describe(),
        }
