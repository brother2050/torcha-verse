"""Hierarchical coordination flow for the TorchaVerse Agent subsystem.

A :class:`HierarchicalFlow` uses a **manager** agent to decompose a
task into subtasks, assigns each subtask to a **worker** agent, and
then synthesises the workers' outputs into a final result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from infrastructure.logger import get_logger
from agents.base_agent import BaseAgent, Result, Step

__all__ = ["HierarchicalFlow"]


class HierarchicalFlow:
    """A hierarchical (manager-worker) multi-agent flow.

    The manager agent decomposes the task into subtasks, each worker
    agent handles one subtask, and the manager synthesises the results.

    Args:
        manager: The manager agent responsible for decomposition and
            synthesis.
        workers: List of worker agents that execute subtasks.
        name: Optional name for the flow.
    """

    def __init__(
        self,
        manager: BaseAgent,
        workers: Sequence[BaseAgent],
        name: str = "hierarchical",
    ) -> None:
        if not workers:
            raise ValueError("HierarchicalFlow requires at least one worker agent.")
        self.manager: BaseAgent = manager
        self.workers: List[BaseAgent] = list(workers)
        self.name: str = name
        self._logger = get_logger(f"HierarchicalFlow[{name}]")

    # ------------------------------------------------------------------
    def execute(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the hierarchical flow.

        The manager first decomposes the task into subtasks, each worker
        executes its assigned subtask, and the manager synthesises the
        results.

        Args:
            task: The initial task.
            max_steps: Optional per-agent step limit override.

        Returns:
            A :class:`Result` containing the synthesised output and the
            combined execution trace.
        """
        all_steps: List[Step] = []
        metadata: Dict[str, Any] = {"flow": self.name, "workers": []}

        self._logger.info(
            "Starting hierarchical flow: 1 manager, %d worker(s).", len(self.workers)
        )

        # --- Phase 1: Manager decomposes the task ---
        decomposition_prompt = self._build_decomposition_prompt(task)
        decomp_result = self.manager.run(decomposition_prompt, max_steps=max_steps)
        all_steps.extend(decomp_result.steps)

        subtasks = self._parse_subtasks(decomp_result.output)
        self._logger.info("Manager decomposed task into %d subtask(s).", len(subtasks))

        # --- Phase 2: Workers execute subtasks ---
        worker_outputs: List[Dict[str, Any]] = []
        for i, subtask in enumerate(subtasks):
            worker = self.workers[i % len(self.workers)]
            self._logger.debug(
                "Worker %d (%s) handling subtask %d.", i + 1, worker.role, i + 1
            )
            worker_result = worker.run(subtask, max_steps=max_steps)
            all_steps.extend(worker_result.steps)
            worker_outputs.append(
                {
                    "worker": worker.role,
                    "subtask": subtask[:200],
                    "output": worker_result.output,
                }
            )
            metadata["workers"].append(
                {
                    "worker": worker.role,
                    "subtask_index": i,
                    "output_length": len(worker_result.output),
                }
            )

        # --- Phase 3: Manager synthesises results ---
        synthesis_prompt = self._build_synthesis_prompt(task, worker_outputs)
        synth_result = self.manager.run(synthesis_prompt, max_steps=max_steps)
        all_steps.extend(synth_result.steps)

        metadata["total_steps"] = len(all_steps)
        metadata["num_subtasks"] = len(subtasks)

        self._logger.info(
            "Hierarchical flow completed: %d total steps.", len(all_steps)
        )

        return Result(
            output=synth_result.output,
            steps=all_steps,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _build_decomposition_prompt(self, task: str) -> str:
        """Build the prompt for the manager to decompose the task.

        Args:
            task: The original task.

        Returns:
            The decomposition prompt.
        """
        return (
            f"You are a task manager. Break down the following task into "
            f"{len(self.workers)} subtasks, one per line, prefixed by a "
            f"number.  Each subtask should be self-contained and "
            f"assignable to a worker agent.\n\n"
            f"Task: {task}\n\nSubtasks:"
        )

    # ------------------------------------------------------------------
    def _build_synthesis_prompt(
        self,
        task: str,
        worker_outputs: List[Dict[str, Any]],
    ) -> str:
        """Build the prompt for the manager to synthesise results.

        Args:
            task: The original task.
            worker_outputs: List of worker output dictionaries.

        Returns:
            The synthesis prompt.
        """
        results_text = "\n\n".join(
            f"Worker '{wo['worker']}' (subtask: {wo['subtask'][:100]}...):\n{wo['output']}"
            for wo in worker_outputs
        )
        return (
            f"You are a task manager. Synthesise the following worker "
            f"results into a single coherent answer for the original "
            f"task.\n\n"
            f"Original task: {task}\n\n"
            f"Worker results:\n{results_text}\n\n"
            f"Synthesised answer:"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subtasks(text: str) -> List[str]:
        """Parse subtask lines from the manager's decomposition output.

        Args:
            text: The manager's response.

        Returns:
            A list of subtask strings.
        """
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in text.strip().split("\n")
            if line.strip()
        ]
        subtasks = [l for l in lines if l]
        return subtasks if subtasks else [text]
