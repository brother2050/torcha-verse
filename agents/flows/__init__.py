"""Agent flow orchestration for the TorchaVerse Agent subsystem.

This package provides multi-agent coordination patterns:

* :class:`SequentialFlow` -- pipeline agents (A -> B -> C).
* :class:`HierarchicalFlow` -- manager-worker delegation.
* :class:`DebateFlow` -- adversarial multi-round debate.

The :class:`FlowOrchestrator` provides a unified interface for creating
and executing flows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from infrastructure.logger import get_logger
from agents.base_agent import BaseAgent, Result
from .debate import DebateFlow
from .hierarchical import HierarchicalFlow
from .sequential import SequentialFlow

__all__ = [
    "SequentialFlow",
    "HierarchicalFlow",
    "DebateFlow",
    "Flow",
    "FlowOrchestrator",
]

#: Type alias for any flow object.
Flow = Union[SequentialFlow, HierarchicalFlow, DebateFlow]


class FlowOrchestrator:
    """Unified manager for flow creation and execution.

    Provides factory methods to create flows from a topology name and
    to execute them on a task.

    Supported topologies:

    * ``"sequential"`` -- :class:`SequentialFlow`.
    * ``"hierarchical"`` -- :class:`HierarchicalFlow`.
    * ``"debate"`` -- :class:`DebateFlow`.
    """

    #: Mapping of topology names to flow classes.
    TOPOLOGIES: Dict[str, type] = {
        "sequential": SequentialFlow,
        "hierarchical": HierarchicalFlow,
        "debate": DebateFlow,
    }

    def __init__(self) -> None:
        self._logger = get_logger("FlowOrchestrator")

    # ------------------------------------------------------------------
    def create_flow(
        self,
        agents: Sequence[BaseAgent],
        topology: str = "sequential",
        **kwargs: Any,
    ) -> Flow:
        """Create a flow from a topology name.

        Args:
            agents: The agents to include in the flow.
            topology: One of ``"sequential"``, ``"hierarchical"``,
                ``"debate"``.
            **kwargs: Additional keyword arguments passed to the flow
                constructor (e.g. ``rounds``, ``judge``, ``manager``,
                ``workers``).

        Returns:
            A flow instance.

        Raises:
            ValueError: If the topology is unknown.
        """
        topology = topology.lower()
        if topology not in self.TOPOLOGIES:
            raise ValueError(
                f"Unknown topology '{topology}'. "
                f"Choose from {list(self.TOPOLOGIES)}."
            )

        if topology == "hierarchical":
            manager = kwargs.pop("manager", agents[0] if agents else None)
            workers = kwargs.pop("workers", list(agents[1:]) if len(agents) > 1 else [])
            if manager is None or not workers:
                raise ValueError(
                    "HierarchicalFlow requires a manager and at least one "
                    "worker. Provide them via kwargs or ensure agents has "
                    "at least 2 elements."
                )
            flow: Flow = HierarchicalFlow(manager=manager, workers=workers, **kwargs)
        elif topology == "debate":
            rounds = kwargs.pop("rounds", 3)
            judge = kwargs.pop("judge", None)
            flow = DebateFlow(agents=agents, rounds=rounds, judge=judge, **kwargs)
        else:
            flow = SequentialFlow(agents=agents, **kwargs)

        self._logger.info("Created %s flow with %d agent(s).", topology, len(agents))
        return flow

    # ------------------------------------------------------------------
    def execute(
        self,
        flow: Flow,
        task: str,
        max_steps: Optional[int] = None,
    ) -> Result:
        """Execute a flow on a task.

        Args:
            flow: The flow to execute.
            task: The task description.
            max_steps: Optional per-agent step limit override.

        Returns:
            The :class:`Result` of the flow execution.
        """
        self._logger.info("Executing %s flow.", flow.__class__.__name__)
        return flow.execute(task, max_steps=max_steps)

    # ------------------------------------------------------------------
    def create_and_execute(
        self,
        agents: Sequence[BaseAgent],
        task: str,
        topology: str = "sequential",
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> Result:
        """Create a flow and execute it in one call.

        Args:
            agents: The agents to include.
            task: The task description.
            topology: The flow topology.
            max_steps: Optional per-agent step limit.
            **kwargs: Additional flow constructor arguments.

        Returns:
            The :class:`Result` of the flow execution.
        """
        flow = self.create_flow(agents, topology=topology, **kwargs)
        return self.execute(flow, task, max_steps=max_steps)

    # ------------------------------------------------------------------
    @classmethod
    def available_topologies(cls) -> List[str]:
        """Return a list of available topology names."""
        return list(cls.TOPOLOGIES)
