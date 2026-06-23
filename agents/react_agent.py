"""ReAct (Reasoning + Acting) agent for TorchaVerse.

This module implements :class:`ReActAgent`, which follows the ReAct
loop: **Thought -> Action -> Observation -> repeat** until a final
answer is produced or the maximum number of steps is reached.

The agent uses a language model (any object with a ``generate`` method)
to produce reasoning traces and parses them into structured
thought/action pairs.  Actions are executed through the
:class:`~torcha_verse.core.tool_registry.ToolRegistry`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from core.tool_registry import BaseTool, Tool, ToolRegistry, ToolResult
from infrastructure.logger import get_logger
from .base_agent import BaseAgent, Memory, Result, Step

__all__ = ["ReActAgent"]


class ReActAgent(BaseAgent):
    """ReAct agent implementing the Thought-Action-Observation loop.

    The agent prompts the language model with the task, execution
    history, and available tools.  It parses the model's response into
    a thought and an action, executes the action (tool call), and feeds
    the observation back into the next iteration.

    The loop terminates when the model emits a ``Final Answer:`` or
    when ``max_steps`` is reached.

    Args:
        role: The agent's role / persona description.
        model: The language model (must have a ``generate`` method).
        tools: Optional list of tool names, :class:`Tool`, or
            :class:`BaseTool` objects.
        memory: Optional pre-configured :class:`Memory`.
        tool_registry: Optional :class:`ToolRegistry`.
        max_steps: Default maximum reasoning steps.
        system_prompt: Optional custom system prompt.  When ``None`` a
            default ReAct prompt is used.
    """

    #: Default system prompt for the ReAct agent.
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a {role}. "
        "Solve the task step by step using the ReAct format.\n"
        "At each step, output:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <JSON arguments for the tool>\n\n"
        "When you have the final answer, output:\n"
        "Thought: <your reasoning>\n"
        "Final Answer: <your answer>\n\n"
        "Available tools:\n{tools}"
    )

    #: Sentinel action names that indicate the agent is done.
    _FINAL_ACTIONS: tuple = ("final", "finish", "done", "answer")

    def __init__(
        self,
        role: str,
        model: Any,
        tools: Optional[Sequence[Union[str, Tool, BaseTool]]] = None,
        memory: Optional[Memory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            role=role,
            model=model,
            tools=tools,
            memory=memory,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )
        self.system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the ReAct loop on ``task``.

        Args:
            task: The task description.
            max_steps: Optional override for the maximum number of
                steps.

        Returns:
            A :class:`Result` with the final output and execution
            trace.
        """
        steps_limit = max_steps or self.max_steps
        steps: List[Step] = []

        self._logger.info("Starting ReAct loop for task: %s", task[:100])

        for i in range(steps_limit):
            # Build the prompt from the task and execution history.
            prompt = self._build_prompt(task, steps)

            # Generate the model's response.
            response = self._generate(prompt)

            # Parse the thought.
            thought = self.parse_thought(response)

            # Parse the action.
            tool_name, params = self.parse_action(response)

            # Check for a final answer.
            if tool_name.lower() in self._FINAL_ACTIONS:
                answer = params.get("answer", "") if isinstance(params, dict) else str(params)
                step = Step(
                    thought=thought,
                    action="final",
                    observation="Final answer produced.",
                )
                steps.append(step)
                self._logger.info("ReAct completed in %d steps.", i + 1)
                return Result(
                    output=answer,
                    steps=steps,
                    metadata={"agent": self.role, "steps_taken": i + 1},
                )

            # Execute the action.
            if tool_name:
                observation = self.execute_action(tool_name, params)
            else:
                observation = "No action specified."

            step = Step(
                thought=thought,
                action=tool_name,
                observation=observation,
            )
            steps.append(step)

            # Store in short-term memory.
            self.memory.short_term.add("thought", thought)
            self.memory.short_term.add("observation", observation)

            self._logger.debug("Step %d: action=%s", i + 1, tool_name)

        # Max steps reached without a final answer.
        self._logger.warning("ReAct loop reached max steps (%d).", steps_limit)

        # Generate a best-effort final answer.
        prompt = self._build_prompt(task, steps) + "\n\nProvide your final answer now."
        final_response = self._generate(prompt)
        _, params = self.parse_action(final_response)
        answer = params.get("answer", final_response) if isinstance(params, dict) else final_response

        return Result(
            output=answer,
            steps=steps,
            metadata={
                "agent": self.role,
                "steps_taken": len(steps),
                "truncated": True,
            },
        )

    # ------------------------------------------------------------------
    def stream(self, task: str, max_steps: Optional[int] = None) -> Iterator[Step]:
        """Stream execution steps as they are produced.

        This is a generator wrapper around :meth:`run` that yields each
        :class:`Step` as it is created.

        Args:
            task: The task description.
            max_steps: Optional override for the maximum number of
                steps.

        Yields:
            :class:`Step` objects.
        """
        steps_limit = max_steps or self.max_steps
        steps: List[Step] = []

        self._logger.info("Streaming ReAct loop for task: %s", task[:100])

        for i in range(steps_limit):
            prompt = self._build_prompt(task, steps)
            response = self._generate(prompt)

            thought = self.parse_thought(response)
            tool_name, params = self.parse_action(response)

            if tool_name.lower() in self._FINAL_ACTIONS:
                answer = params.get("answer", "") if isinstance(params, dict) else str(params)
                step = Step(
                    thought=thought,
                    action="final",
                    observation="Final answer produced.",
                )
                steps.append(step)
                yield step
                # Store the final answer in working memory.
                self.memory.working.set("final_answer", answer)
                return

            if tool_name:
                observation = self.execute_action(tool_name, params)
            else:
                observation = "No action specified."

            step = Step(
                thought=thought,
                action=tool_name,
                observation=observation,
            )
            steps.append(step)
            yield step

        # Best-effort final answer.
        prompt = self._build_prompt(task, steps) + "\n\nProvide your final answer now."
        final_response = self._generate(prompt)
        _, params = self.parse_action(final_response)
        answer = params.get("answer", final_response) if isinstance(params, dict) else final_response
        self.memory.working.set("final_answer", answer)

    # ------------------------------------------------------------------
    def parse_thought(self, response: str) -> str:
        """Parse the thought from a model response.

        Looks for a ``Thought:`` prefix and extracts the text up to the
        next ``Action:`` or ``Final Answer:`` line.  If no structured
        thought is found, the entire response is returned.

        Args:
            response: The model's response text.

        Returns:
            The extracted thought string.
        """
        # Try to match "Thought: ... (until Action: or Final Answer: or end)"
        match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return response.strip()

    # ------------------------------------------------------------------
    def parse_action(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse the action from a model response.

        Looks for ``Final Answer:``, ``Action:``, and ``Action Input:``
        patterns.  Returns a tuple of ``(tool_name, params)`` where
        ``params`` is a dictionary.

        When a ``Final Answer:`` is detected, the tool name is
        ``"final"`` and ``params`` contains ``{"answer": ...}``.

        Args:
            response: The model's response text.

        Returns:
            A tuple ``(tool_name, params)``.
        """
        # Check for a final answer.
        final_match = re.search(
            r"Final Answer:\s*(.*?)(?:\n|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            return "final", {"answer": final_match.group(1).strip()}

        # Parse Action.
        action_match = re.search(
            r"Action:\s*(.*?)(?:\n|$)",
            response,
            re.IGNORECASE,
        )
        tool_name = action_match.group(1).strip() if action_match else ""

        # Parse Action Input.
        input_match = re.search(
            r"Action Input:\s*(.*?)(?:\n\s*(?:Thought:|Action:|Final Answer:)|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        params_str = input_match.group(1).strip() if input_match else ""

        # Try to parse as JSON; fall back to a plain "input" parameter.
        params: Dict[str, Any]
        if params_str:
            try:
                params = json.loads(params_str)
                if not isinstance(params, dict):
                    params = {"input": params}
            except (json.JSONDecodeError, TypeError):
                params = {"input": params_str}
        else:
            params = {}

        return tool_name, params

    # ------------------------------------------------------------------
    def execute_action(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Execute a tool action.

        Args:
            tool_name: The name of the tool to execute.
            params: The parameters to pass to the tool.

        Returns:
            The tool's output as a string, or an error message.
        """
        if not tool_name:
            return "No action specified."

        if tool_name.lower() in self._FINAL_ACTIONS:
            answer = params.get("answer", str(params))
            return str(answer)

        # Check that the tool is available to this agent.
        if tool_name not in self._tool_names:
            available = ", ".join(self._tool_names) or "(none)"
            return f"Error: Tool '{tool_name}' is not available. Available tools: {available}"

        result: ToolResult = self.tool_registry.execute_tool(tool_name, params)
        if result.success:
            return str(result.output)
        return f"Error: {result.error}"

    # ------------------------------------------------------------------
    def format_tools(self, tools: Optional[Sequence[Union[str, Tool, BaseTool]]] = None) -> str:
        """Format tool descriptions for inclusion in the LLM prompt.

        Args:
            tools: Optional list of tools to format.  When ``None``
                the agent's own tools are used.

        Returns:
            A formatted string listing each tool's name and
            description.
        """
        tool_names = self._normalise_tools(tools) if tools else self._tool_names
        lines: List[str] = []

        for name in tool_names:
            tool = self.tool_registry.get_tool(name)
            if tool is not None:
                params_desc = ", ".join(tool.parameter_schema.keys()) if tool.parameter_schema else "none"
                lines.append(f"- {tool.name}: {tool.description} (params: {params_desc})")
            else:
                lines.append(f"- {name}: (not registered)")

        return "\n".join(lines) if lines else "(no tools available)"

    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, history: List[Step]) -> str:
        """Build the full ReAct prompt.

        Args:
            task: The task description.
            history: Previous execution steps.

        Returns:
            The formatted prompt string.
        """
        tools_text = self.format_tools()
        system = self.system_prompt.format(role=self.role, tools=tools_text)

        parts: List[str] = [system, f"Task: {task}"]

        if history:
            parts.append("History:")
            for i, step in enumerate(history, 1):
                parts.append(f"  Step {i}:")
                if step.thought:
                    parts.append(f"    Thought: {step.thought}")
                if step.action:
                    parts.append(f"    Action: {step.action}")
                if step.observation:
                    parts.append(f"    Observation: {step.observation}")

        parts.append("Next step:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        """Generate a response from the model.

        Args:
            prompt: The prompt string.

        Returns:
            The model's response text.
        """
        try:
            return self.model.generate(prompt, max_tokens=256, temperature=0.3)
        except TypeError:
            # Fallback for models with a different signature.
            return self.model.generate(prompt)
