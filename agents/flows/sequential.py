"""Sequential flow for the TorchaVerse Agent subsystem.

A :class:`SequentialFlow` chains agents together so that each agent's
output becomes the next agent's input (A -> B -> C).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from infrastructure.logger import get_logger
from agents.base_agent import BaseAgent, Result, Step

__all__ = ["SequentialFlow"]


class SequentialFlow:
    """A sequential (pipeline) multi-agent flow.

    Agents are executed in order.  The output of each agent is fed as
    the task to the next agent, producing a pipeline effect.

    Args:
        agents: Ordered list of agents to execute.
        name: Optional name for the flow.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        name: str = "sequential",
    ) -> None:
        if not agents:
            raise ValueError("SequentialFlow requires at least one agent.")
        self.agents: List[BaseAgent] = list(agents)
        self.name: str = name
        self._logger = get_logger(f"SequentialFlow[{name}]")

    # ------------------------------------------------------------------
    def execute(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the sequential pipeline.

        Args:
            task: The initial task.
            max_steps: Optional per-agent step limit override.

        Returns:
            A :class:`Result` containing the final output and the
            combined execution trace from all agents.
        """
        all_steps: List[Step] = []
        current_input: str = task
        metadata: Dict[str, Any] = {"flow": self.name, "agent_results": []}

        self._logger.info("Starting sequential flow with %d agent(s).", len(self.agents))

        for i, agent in enumerate(self.agents):
            self._logger.debug("Agent %d (%s) processing input.", i + 1, agent.role)
            result = agent.run(current_input, max_steps=max_steps)

            all_steps.extend(result.steps)
            metadata["agent_results"].append(
                {
                    "agent": agent.role,
                    "index": i,
                    "output": result.output[:200],
                    "steps": len(result.steps),
                }
            )

            # The output becomes the next agent's input.
            current_input = result.output

            self._logger.debug(
                "Agent %d produced %d chars of output.", i + 1, len(result.output)
            )

        metadata["total_steps"] = len(all_steps)

        self._logger.info(
            "Sequential flow completed: %d total steps, final output %d chars.",
            len(all_steps), len(current_input),
        )

        return Result(
            output=current_input,
            steps=all_steps,
            metadata=metadata,
        )
