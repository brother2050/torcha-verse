"""Agent engine for TorchaVerse.

This module provides :class:`AgentEngine`, the capability-layer entry
point for autonomous AI agents.  It manages single-agent and multi-agent
systems, supporting ReAct-style reasoning, tool-calling, and various
flow topologies (sequential, parallel, hierarchical, debate).

Because the ``agents/`` sub-packages (``agents/``, ``agents/flows/``)
are currently empty stubs, the supporting classes (:class:`BaseAgent`,
:class:`ReActAgent`, :class:`ToolCallAgent`, :class:`FlowOrchestrator`)
are implemented directly in this module.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from core.tool_registry import ToolRegistry, ToolResult
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from .text_engine import Message, TextEngine

__all__ = [
    "Step",
    "Result",
    "AgentCore",
    "BaseAgent",
    "ReActAgent",
    "ToolCallAgent",
    "FlowOrchestrator",
    "AgentEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Step:
    """A single step in an agent's execution trace.

    Attributes:
        thought: The agent's reasoning for this step.
        action: The action taken (e.g. tool name).
        action_input: Input to the action.
        observation: Result of the action.
        step_number: Sequential step index.
    """

    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    step_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "step_number": self.step_number,
        }


@dataclass
class Result:
    """The result of an agent execution.

    Attributes:
        output: The final output text.
        steps: The execution trace as a list of :class:`Step`.
        metadata: Additional metadata (timing, agent names, etc.).
    """

    output: str = ""
    steps: List[Step] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "output": self.output,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AgentCore
# ---------------------------------------------------------------------------
@dataclass
class AgentCore:
    """Configuration and state for a single agent.

    Attributes:
        role: The agent's role / persona description.
        tools: List of tool names available to the agent.
        model: The model name used by this agent.
        system_prompt: The system prompt for this agent.
        max_steps: Maximum reasoning steps.
    """

    role: str = "assistant"
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    system_prompt: str = ""
    max_steps: int = 10
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.role.replace(" ", "_").lower()
        if not self.system_prompt:
            self.system_prompt = (
                f"You are a {self.role}. "
                f"Break down the task step by step. "
                f"When you need information, use the available tools. "
                f"When you have the final answer, prefix it with 'FINAL ANSWER:'."
            )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """Abstract base class for all agents.

    Args:
        core: The agent's configuration (:class:`AgentCore`).
        text_engine: The text engine for LLM reasoning.
        tool_registry: Optional tool registry for function calling.
    """

    def __init__(
        self,
        core: AgentCore,
        text_engine: TextEngine,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.core: AgentCore = core
        self.text_engine: TextEngine = text_engine
        self.tool_registry: ToolRegistry = tool_registry or ToolRegistry()
        self._logger = get_logger(f"Agent[{core.name}]")
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the agent on a task.

        Args:
            task: The task description.
            max_steps: Optional override for max steps.

        Returns:
            A :class:`Result`.
        """
        ...

    # ------------------------------------------------------------------
    def stream(self, task: str, max_steps: Optional[int] = None) -> Iterator[Step]:
        """Stream execution steps.

        Args:
            task: The task description.
            max_steps: Optional override.

        Yields:
            :class:`Step` objects as they are produced.
        """
        result = self.run(task, max_steps)
        for step in result.steps:
            yield step

    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, history: List[Step]) -> str:
        """Build the LLM prompt from the task and execution history.

        Args:
            task: The task description.
            history: Previous execution steps.

        Returns:
            The formatted prompt string.
        """
        parts: List[str] = [f"[SYSTEM] {self.core.system_prompt}"]
        parts.append(f"[TASK] {task}")

        if history:
            parts.append("[HISTORY]")
            for step in history:
                parts.append(f"  Step {step.step_number}:")
                if step.thought:
                    parts.append(f"    Thought: {step.thought}")
                if step.action:
                    parts.append(f"    Action: {step.action}")
                    parts.append(f"    Action Input: {step.action_input}")
                if step.observation:
                    parts.append(f"    Observation: {step.observation}")

        # Tool descriptions.
        if self.core.tools:
            tool_descs = self.tool_registry.get_tool_descriptions()
            available = [t for t in tool_descs if t["name"] in self.core.tools]
            if available:
                parts.append(f"[TOOLS] {json.dumps(available, indent=2)}")

        parts.append("[NEXT STEP]")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _parse_response(self, response: str) -> Tuple[str, str, str, bool]:
        """Parse an LLM response into (thought, action, action_input, is_final).

        Args:
            response: The LLM response text.

        Returns:
            A tuple ``(thought, action, action_input, is_final)``.
        """
        # Check for final answer.
        if "FINAL ANSWER:" in response.upper():
            idx = response.upper().index("FINAL ANSWER:")
            answer = response[idx + len("FINAL ANSWER:"):].strip()
            return response[:idx].strip(), "final", answer, True

        # Parse Thought / Action / Action Input.
        thought = ""
        action = ""
        action_input = ""

        thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(.*?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL
        )
        if input_match:
            action_input = input_match.group(1).strip()

        # If no structured parse, use the full response as thought.
        if not thought and not action:
            thought = response.strip()

        return thought, action, action_input, False

    # ------------------------------------------------------------------
    def _execute_tool(self, action: str, action_input: str) -> str:
        """Execute a tool call.

        Args:
            action: Tool name.
            action_input: Tool input (JSON string or plain text).

        Returns:
            The tool result as a string.
        """
        if action not in self.core.tools:
            return f"Error: Tool '{action}' is not available."

        # Parse action input.
        try:
            params = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            params = {"input": action_input}

        result: ToolResult = self.tool_registry.execute_tool(action, params)
        if result.success:
            return str(result.output)
        return f"Error: {result.error}"

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the agent's state."""
        self._history.clear()


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------
class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) agent.

    Follows the ReAct loop: Thought -> Action -> Observation -> repeat
    until a final answer is produced or max steps is reached.

    Args:
        core: The agent's configuration.
        text_engine: The text engine for LLM reasoning.
        tool_registry: Optional tool registry.
    """

    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the ReAct loop.

        Args:
            task: The task description.
            max_steps: Optional override for max steps.

        Returns:
            A :class:`Result` with the execution trace.
        """
        steps_limit = max_steps or self.core.max_steps
        steps: List[Step] = []

        self._logger.info("Starting ReAct loop for task: %s", task[:100])

        for i in range(steps_limit):
            # Build prompt.
            prompt = self._build_prompt(task, steps)

            # Generate response.
            response = self.text_engine.generate(
                prompt, max_tokens=256, temperature=0.3
            )

            # Parse response.
            thought, action, action_input, is_final = self._parse_response(response)

            step = Step(
                thought=thought,
                action=action,
                action_input=action_input,
                step_number=i + 1,
            )

            if is_final:
                step.observation = "Final answer produced."
                steps.append(step)
                self._logger.info("ReAct completed in %d steps.", i + 1)
                return Result(
                    output=action_input,
                    steps=steps,
                    metadata={"agent": self.core.name, "steps_taken": i + 1},
                )

            # Execute tool if action is specified.
            if action and action != "final":
                observation = self._execute_tool(action, action_input)
                step.observation = observation
            else:
                step.observation = "No action taken."

            steps.append(step)
            self._logger.debug("Step %d: action=%s", i + 1, action)

        # Max steps reached without final answer.
        self._logger.warning("ReAct loop reached max steps (%d).", steps_limit)

        # Generate a best-effort final answer.
        prompt = self._build_prompt(task, steps) + "\n[Provide your final answer now.]"
        final_response = self.text_engine.generate(prompt, max_tokens=256)

        return Result(
            output=final_response,
            steps=steps,
            metadata={
                "agent": self.core.name,
                "steps_taken": len(steps),
                "truncated": True,
            },
        )


# ---------------------------------------------------------------------------
# ToolCallAgent
# ---------------------------------------------------------------------------
class ToolCallAgent(BaseAgent):
    """Tool-calling agent.

    Uses the text engine's native function-calling support to decide
    which tools to invoke, then executes them and feeds results back.

    Args:
        core: The agent's configuration.
        text_engine: The text engine for LLM reasoning.
        tool_registry: Optional tool registry.
    """

    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the tool-calling loop.

        Args:
            task: The task description.
            max_steps: Optional override for max steps.

        Returns:
            A :class:`Result`.
        """
        steps_limit = max_steps or self.core.max_steps
        steps: List[Step] = []
        messages: List[Message] = [
            Message(role="system", content=self.core.system_prompt),
            Message(role="user", content=task),
        ]

        # Get tool schemas.
        tool_descs = self.tool_registry.get_tool_descriptions()
        available_tools = [t for t in tool_descs if t["name"] in self.core.tools]

        self._logger.info("Starting ToolCall loop for task: %s", task[:100])

        for i in range(steps_limit):
            # Use the text engine's chat with tools.
            assistant_msg = self.text_engine.chat(
                messages,
                tools=available_tools if available_tools else None,
                max_tokens=256,
            )
            messages.append(assistant_msg)

            step = Step(
                thought=assistant_msg.content,
                step_number=i + 1,
            )

            # Check for tool calls.
            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    step.action = tc.name
                    step.action_input = json.dumps(tc.arguments)

                    # Execute the tool.
                    result = self.tool_registry.execute_tool(tc.name, tc.arguments)
                    step.observation = str(result.output) if result.success else f"Error: {result.error}"

                    # Add tool result to messages.
                    messages.append(
                        Message(
                            role="tool",
                            content=step.observation,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    steps.append(step)

                self._logger.debug("Step %d: executed %d tool calls.", i + 1, len(assistant_msg.tool_calls))
            else:
                # No tool calls -- check if this is a final answer.
                if "FINAL ANSWER:" in assistant_msg.content.upper():
                    idx = assistant_msg.content.upper().index("FINAL ANSWER:")
                    answer = assistant_msg.content[idx + len("FINAL ANSWER:"):].strip()
                    steps.append(step)
                    return Result(
                        output=answer,
                        steps=steps,
                        metadata={"agent": self.core.name, "steps_taken": i + 1},
                    )

                steps.append(step)

                # If no tool calls and no final answer, generate one.
                if i == steps_limit - 1:
                    messages.append(
                        Message(
                            role="user",
                            content="Please provide your final answer now.",
                        )
                    )

        # Generate final answer.
        final_msg = self.text_engine.chat(messages, max_tokens=256)

        return Result(
            output=final_msg.content,
            steps=steps,
            metadata={
                "agent": self.core.name,
                "steps_taken": len(steps),
                "truncated": True,
            },
        )


# ---------------------------------------------------------------------------
# FlowOrchestrator
# ---------------------------------------------------------------------------
class FlowOrchestrator:
    """Orchestrates multi-agent execution flows.

    Supports four topologies:

    * ``"sequential"`` -- agents run one after another, each receiving
      the previous agent's output.
    * ``"parallel"`` -- all agents run independently on the same task;
      results are merged.
    * ``"hierarchical"`` -- a manager agent decomposes the task and
      delegates to worker agents.
    * ``"debate"`` -- agents produce independent answers and a
      synthesiser combines them.

    Args:
        agents: List of agents to orchestrate.
        topology: Flow topology name.
        synthesiser: Optional synthesiser agent for debate/hierarchical.
    """

    TOPOLOGIES: Tuple[str, ...] = ("sequential", "parallel", "hierarchical", "debate")

    def __init__(
        self,
        agents: List[BaseAgent],
        topology: str = "sequential",
        synthesiser: Optional[BaseAgent] = None,
    ) -> None:
        if topology not in self.TOPOLOGIES:
            raise ValueError(
                f"Unknown topology '{topology}'. Choose from {self.TOPOLOGIES}."
            )
        self.agents: List[BaseAgent] = agents
        self.topology: str = topology
        self.synthesiser: Optional[BaseAgent] = synthesiser
        self._logger = get_logger(f"FlowOrchestrator[{topology}]")

    # ------------------------------------------------------------------
    def execute(self, task: str) -> Result:
        """Execute the flow on a task.

        Args:
            task: The task description.

        Returns:
            A :class:`Result` combining all agents' outputs.
        """
        if self.topology == "sequential":
            return self._run_sequential(task)
        elif self.topology == "parallel":
            return self._run_parallel(task)
        elif self.topology == "hierarchical":
            return self._run_hierarchical(task)
        elif self.topology == "debate":
            return self._run_debate(task)
        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

    # ------------------------------------------------------------------
    def _run_sequential(self, task: str) -> Result:
        """Run agents sequentially, chaining outputs."""
        all_steps: List[Step] = []
        current_task = task

        for i, agent in enumerate(self.agents):
            self._logger.info("Sequential: agent %d/%d running.", i + 1, len(self.agents))
            result = agent.run(current_task)
            all_steps.extend(result.steps)
            current_task = result.output

        return Result(
            output=current_task,
            steps=all_steps,
            metadata={
                "topology": "sequential",
                "num_agents": len(self.agents),
                "agent_names": [a.core.name for a in self.agents],
            },
        )

    # ------------------------------------------------------------------
    def _run_parallel(self, task: str) -> Result:
        """Run all agents in parallel on the same task."""
        all_steps: List[Step] = []
        outputs: List[str] = []

        for i, agent in enumerate(self.agents):
            self._logger.info("Parallel: agent %d/%d running.", i + 1, len(self.agents))
            result = agent.run(task)
            all_steps.extend(result.steps)
            outputs.append(f"[{agent.core.name}] {result.output}")

        # Merge outputs.
        if self.synthesiser:
            merge_prompt = (
                f"Combine the following agent outputs into a single "
                f"coherent answer:\n\n" + "\n\n".join(outputs)
            )
            merged = self.synthesiser.run(merge_prompt)
            all_steps.extend(merged.steps)
            final_output = merged.output
        else:
            final_output = "\n\n---\n\n".join(outputs)

        return Result(
            output=final_output,
            steps=all_steps,
            metadata={
                "topology": "parallel",
                "num_agents": len(self.agents),
                "agent_names": [a.core.name for a in self.agents],
            },
        )

    # ------------------------------------------------------------------
    def _run_hierarchical(self, task: str) -> Result:
        """Run a hierarchical flow: manager decomposes, workers execute."""
        all_steps: List[Step] = []

        # The first agent is the manager.
        manager = self.agents[0]
        workers = self.agents[1:]

        # Manager decomposes the task.
        decompose_prompt = (
            f"Break down the following task into {len(workers)} subtasks. "
            f"List each subtask on a separate line prefixed with 'SUBTASK: '.\n\n"
            f"Task: {task}"
        )
        decomp_result = manager.run(decompose_prompt)
        all_steps.extend(decomp_result.steps)

        # Parse subtasks.
        subtasks: List[str] = []
        for line in decomp_result.output.split("\n"):
            if line.strip().upper().startswith("SUBTASK:"):
                subtasks.append(line.strip()[len("SUBTASK:"):].strip())

        # Ensure we have enough subtasks.
        while len(subtasks) < len(workers):
            subtasks.append(task)

        # Workers execute subtasks.
        worker_outputs: List[str] = []
        for i, worker in enumerate(workers):
            subtask = subtasks[i] if i < len(subtasks) else task
            self._logger.info("Hierarchical: worker %d running subtask.", i + 1)
            result = worker.run(subtask)
            all_steps.extend(result.steps)
            worker_outputs.append(result.output)

        # Manager synthesises.
        synth_prompt = (
            f"Synthesise the following subtask results into a final answer:\n\n"
            + "\n\n".join(worker_outputs)
        )
        synth_result = manager.run(synth_prompt)
        all_steps.extend(synth_result.steps)

        return Result(
            output=synth_result.output,
            steps=all_steps,
            metadata={
                "topology": "hierarchical",
                "num_agents": len(self.agents),
                "num_workers": len(workers),
            },
        )

    # ------------------------------------------------------------------
    def _run_debate(self, task: str) -> Result:
        """Run a debate flow: agents argue, synthesiser decides."""
        all_steps: List[Step] = []
        arguments: List[str] = []

        # Each agent produces an independent answer.
        for i, agent in enumerate(self.agents):
            self._logger.info("Debate: agent %d/%d arguing.", i + 1, len(self.agents))
            result = agent.run(task)
            all_steps.extend(result.steps)
            arguments.append(f"[{agent.core.name}] {result.output}")

        # Synthesiser combines.
        if self.synthesiser:
            synth_prompt = (
                f"Multiple agents have provided different answers to the "
                f"following task. Review their arguments and provide the "
                f"best final answer.\n\n"
                f"Task: {task}\n\n"
                f"Arguments:\n" + "\n\n".join(arguments)
            )
            synth_result = self.synthesiser.run(synth_prompt)
            all_steps.extend(synth_result.steps)
            final_output = synth_result.output
        else:
            # Without a synthesiser, return all arguments.
            final_output = "\n\n---\n\n".join(arguments)

        return Result(
            output=final_output,
            steps=all_steps,
            metadata={
                "topology": "debate",
                "num_agents": len(self.agents),
                "agent_names": [a.core.name for a in self.agents],
            },
        )


# ---------------------------------------------------------------------------
# AgentEngine
# ---------------------------------------------------------------------------
class AgentEngine:
    """Top-level agent engine.

    Manages agent creation, flow orchestration, and execution.

    Args:
        text_engine: The text engine for LLM reasoning.
        tool_registry: Optional tool registry.
        config: Optional configuration dictionary.
        device: Optional device override.
    """

    def __init__(
        self,
        text_engine: Optional[TextEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, Any]] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("AgentEngine")

        self._device = device or self._device_manager.get_device()

        # Text engine for LLM reasoning.
        self.text_engine: TextEngine = text_engine or TextEngine(
            "default", device=self._device
        )

        # Tool registry.
        self.tool_registry: ToolRegistry = tool_registry or ToolRegistry()

        # Registered agents.
        self._agents: Dict[str, BaseAgent] = {}

        self._logger.info("AgentEngine initialised.")

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------
    def create_agent(
        self,
        role: str,
        tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        agent_type: str = "react",
        max_steps: int = 10,
        system_prompt: str = "",
    ) -> AgentCore:
        """Create and register a new agent.

        Args:
            role: The agent's role / persona.
            tools: List of tool names available to the agent.
            model: Model name override.
            agent_type: ``"react"`` or ``"toolcall"``.
            max_steps: Maximum reasoning steps.
            system_prompt: Custom system prompt.

        Returns:
            The :class:`AgentCore` configuration.
        """
        core = AgentCore(
            role=role,
            tools=tools or [],
            model=model,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

        # Instantiate the agent.
        if agent_type == "toolcall":
            agent: BaseAgent = ToolCallAgent(
                core=core,
                text_engine=self.text_engine,
                tool_registry=self.tool_registry,
            )
        else:
            agent = ReActAgent(
                core=core,
                text_engine=self.text_engine,
                tool_registry=self.tool_registry,
            )

        self._agents[core.name] = agent
        self._logger.info("Created agent '%s' (type=%s).", core.name, agent_type)
        return core

    # ------------------------------------------------------------------
    # Flow creation
    # ------------------------------------------------------------------
    def create_flow(
        self,
        agents: Union[List[str], List[BaseAgent], List[AgentCore]],
        topology: str = "sequential",
        synthesiser: Optional[Union[str, BaseAgent]] = None,
    ) -> FlowOrchestrator:
        """Create a multi-agent flow.

        Args:
            agents: List of agent names, :class:`AgentCore`, or
                :class:`BaseAgent` objects.
            topology: Flow topology (``"sequential"``, ``"parallel"``,
                ``"hierarchical"``, ``"debate"``).
            synthesiser: Optional synthesiser agent for debate or
                parallel flows.

        Returns:
            A :class:`FlowOrchestrator`.
        """
        # Resolve agents.
        resolved: List[BaseAgent] = []
        for a in agents:
            if isinstance(a, BaseAgent):
                resolved.append(a)
            elif isinstance(a, AgentCore):
                agent = self._agents.get(a.name)
                if agent is None:
                    raise ValueError(f"Agent '{a.name}' is not registered.")
                resolved.append(agent)
            elif isinstance(a, str):
                agent = self._agents.get(a)
                if agent is None:
                    raise ValueError(f"Agent '{a}' is not registered.")
                resolved.append(agent)
            else:
                raise TypeError(f"Unsupported agent type: {type(a)}")

        # Resolve synthesiser.
        synth: Optional[BaseAgent] = None
        if synthesiser is not None:
            if isinstance(synthesiser, BaseAgent):
                synth = synthesiser
            elif isinstance(synthesiser, str):
                synth = self._agents.get(synthesiser)

        flow = FlowOrchestrator(
            agents=resolved,
            topology=topology,
            synthesiser=synth,
        )
        self._logger.info(
            "Created flow (topology=%s, agents=%d).", topology, len(resolved)
        )
        return flow

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, flow: FlowOrchestrator, task: str) -> Result:
        """Execute a flow on a task.

        Args:
            flow: The flow orchestrator.
            task: The task description.

        Returns:
            A :class:`Result`.
        """
        return flow.execute(task)

    # ------------------------------------------------------------------
    def run(self, task: str, max_steps: int = 10) -> Result:
        """Convenience method: create a default agent and run.

        Args:
            task: The task description.
            max_steps: Maximum reasoning steps.

        Returns:
            A :class:`Result`.
        """
        # Create a default ReAct agent if none exists.
        if not self._agents:
            self.create_agent(
                role="general assistant",
                max_steps=max_steps,
            )

        agent = list(self._agents.values())[0]
        return agent.run(task, max_steps=max_steps)

    # ------------------------------------------------------------------
    def stream(self, task: str, max_steps: int = 10) -> Iterator[Step]:
        """Stream execution steps for a task.

        Creates a default agent (if needed) and yields steps as they
        are produced.

        Args:
            task: The task description.
            max_steps: Maximum reasoning steps.

        Yields:
            :class:`Step` objects.
        """
        if not self._agents:
            self.create_agent(
                role="general assistant",
                max_steps=max_steps,
            )

        agent = list(self._agents.values())[0]
        yield from agent.stream(task, max_steps)

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------
    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the engine's tool registry.

        Args:
            name: Tool name.
            func: Callable to execute.
            description: Human-readable description.
            parameter_schema: Parameter schema.
        """
        self.tool_registry.register_tool(name, func, description, parameter_schema)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def agents(self) -> Dict[str, BaseAgent]:
        """Registered agents."""
        return self._agents

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Retrieve a registered agent by name.

        Args:
            name: Agent name.

        Returns:
            The agent or ``None``.
        """
        return self._agents.get(name)

    def __repr__(self) -> str:
        return (
            f"AgentEngine(agents={len(self._agents)}, "
            f"tools={len(self.tool_registry.get_tool_descriptions())})"
        )
