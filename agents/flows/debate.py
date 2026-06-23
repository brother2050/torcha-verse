"""Debate flow for the TorchaVerse Agent subsystem.

A :class:`DebateFlow` runs multiple agents in rounds, where each agent
reviews and critiques the others' outputs.  After a configurable number
of rounds, a judge agent (or the last round's consensus) produces the
final answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from infrastructure.logger import get_logger
from agents.base_agent import BaseAgent, Result, Step

__all__ = ["DebateFlow"]


class DebateFlow:
    """An adversarial debate multi-agent flow.

    Multiple agents propose answers and critique each other over several
    rounds.  A judge agent (optional) synthesises the final answer.

    Args:
        agents: List of debating agents.
        rounds: Number of debate rounds.
        judge: Optional judge agent for final synthesis.  When
            ``None`` the last round's best answer is returned.
        name: Optional name for the flow.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        rounds: int = 3,
        judge: Optional[BaseAgent] = None,
        name: str = "debate",
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateFlow requires at least 2 agents.")
        if rounds < 1:
            raise ValueError("rounds must be >= 1.")
        self.agents: List[BaseAgent] = list(agents)
        self.rounds: int = rounds
        self.judge: Optional[BaseAgent] = judge
        self.name: str = name
        self._logger = get_logger(f"DebateFlow[{name}]")

    # ------------------------------------------------------------------
    def execute(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the debate flow.

        Args:
            task: The initial task / question to debate.
            max_steps: Optional per-agent step limit override.

        Returns:
            A :class:`Result` containing the final consensus answer
            and the combined execution trace.
        """
        all_steps: List[Step] = []
        metadata: Dict[str, Any] = {
            "flow": self.name,
            "rounds": self.rounds,
            "num_agents": len(self.agents),
            "round_results": [],
        }

        self._logger.info(
            "Starting debate: %d agents, %d round(s).", len(self.agents), self.rounds
        )

        # --- Round 1: Initial answers ---
        positions: Dict[str, str] = {}
        for i, agent in enumerate(self.agents):
            self._logger.debug("Round 1, agent %d (%s) proposing.", i + 1, agent.role)
            result = agent.run(task, max_steps=max_steps)
            all_steps.extend(result.steps)
            positions[agent.role] = result.output
            metadata["round_results"].append(
                {"round": 1, "agent": agent.role, "output_length": len(result.output)}
            )

        # --- Subsequent rounds: critique and revise ---
        for round_num in range(2, self.rounds + 1):
            new_positions: Dict[str, str] = {}
            for i, agent in enumerate(self.agents):
                # Build a critique prompt from other agents' positions.
                others = {
                    role: pos
                    for role, pos in positions.items()
                    if role != agent.role
                }
                critique_prompt = self._build_critique_prompt(task, agent.role, others)

                self._logger.debug(
                    "Round %d, agent %d (%s) revising.", round_num, i + 1, agent.role
                )
                result = agent.run(critique_prompt, max_steps=max_steps)
                all_steps.extend(result.steps)
                new_positions[agent.role] = result.output
                metadata["round_results"].append(
                    {
                        "round": round_num,
                        "agent": agent.role,
                        "output_length": len(result.output),
                    }
                )

            positions = new_positions

        # --- Final synthesis ---
        if self.judge is not None:
            self._logger.info("Judge synthesising final answer.")
            synthesis_prompt = self._build_synthesis_prompt(task, positions)
            final_result = self.judge.run(synthesis_prompt, max_steps=max_steps)
            all_steps.extend(final_result.steps)
            final_output = final_result.output
            metadata["judge"] = self.judge.role
        else:
            # No judge: pick the longest answer as a simple heuristic.
            final_output = max(positions.values(), key=len) if positions else ""
            metadata["selection_method"] = "longest"

        metadata["total_steps"] = len(all_steps)

        self._logger.info(
            "Debate completed: %d total steps, final output %d chars.",
            len(all_steps), len(final_output),
        )

        return Result(
            output=final_output,
            steps=all_steps,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _build_critique_prompt(
        self,
        task: str,
        agent_role: str,
        others: Dict[str, str],
    ) -> str:
        """Build the critique prompt for an agent.

        Args:
            task: The original task.
            agent_role: The role of the agent being prompted.
            others: Mapping of other agents' roles to their positions.

        Returns:
            The critique prompt.
        """
        others_text = "\n\n".join(
            f"{role}:\n{position[:500]}"
            for role, position in others.items()
        )
        return (
            f"You are {agent_role}. The following is the original task "
            f"and the positions of other participants in a debate.\n\n"
            f"Original task: {task}\n\n"
            f"Other positions:\n{others_text}\n\n"
            f"Review the other positions, identify strengths and "
            f"weaknesses, and provide your revised answer."
        )

    # ------------------------------------------------------------------
    def _build_synthesis_prompt(
        self,
        task: str,
        positions: Dict[str, str],
    ) -> str:
        """Build the synthesis prompt for the judge.

        Args:
            task: The original task.
            positions: Mapping of agent roles to their final positions.

        Returns:
            The synthesis prompt.
        """
        positions_text = "\n\n".join(
            f"{role}:\n{position[:500]}" for role, position in positions.items()
        )
        return (
            f"You are the judge of a debate. Synthesise the following "
            f"positions into a single, well-reasoned final answer.\n\n"
            f"Original task: {task}\n\n"
            f"Final positions:\n{positions_text}\n\n"
            f"Final synthesised answer:"
        )
