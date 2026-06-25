"""Tool-calling agent for TorchaVerse.

This module implements :class:`ToolCallAgent`, which extends
:class:`~torcha_verse.agents.react_agent.ReActAgent` to use native
function calling via the model's ``chat`` interface instead of parsing
free-text ReAct traces.

The agent maintains a conversation message list, sends it to the model
along with tool schemas, and processes any returned ``ToolCall`` objects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from core.tool_registry import BaseTool, Tool, ToolRegistry, ToolResult
from dataclasses import dataclass
@dataclass
class Message:
    role: str = "user"
    content: str = ""

class ToolCall:
    pass
from infrastructure.logger import get_logger
from .base_agent import Memory, Result, Step
from .react_agent import ReActAgent

__all__ = ["ToolCallAgent"]


class ToolCallAgent(ReActAgent):
    """Agent that uses native function calling.

    Instead of prompting the model with a ReAct text format and parsing
    the output, this agent uses the model's ``chat`` method with tool
    schemas, allowing the model to emit structured
    :class:`ToolCall` objects directly.

    Args:
        role: The agent's role / persona description.
        model: The language model (must have a ``chat`` method that
            accepts ``messages`` and ``tools`` and returns a
            :class:`Message`).
        tools: Optional list of tool names, :class:`Tool`, or
            :class:`BaseTool` objects.
        memory: Optional pre-configured :class:`Memory`.
        tool_registry: Optional :class:`ToolRegistry`.
        max_steps: Default maximum reasoning steps.
        system_prompt: Optional custom system prompt.
    """

    #: Marker the model uses to signal a final answer (fallback when
    #: the model does not emit tool calls).
    _FINAL_MARKER: str = "FINAL ANSWER:"

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
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the tool-calling loop on ``task``.

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

        self._logger.info("Starting tool-call loop for task: %s", task[:100])

        # Build the initial message list.
        messages: List[Message] = [
            Message(role="system", content=self.role),
            Message(role="user", content=task),
        ]

        # Collect tool schemas for the agent's tools.
        tool_schemas = self._get_tool_schemas()

        for i in range(steps_limit):
            # Call the model's chat method.
            assistant_msg = self._chat(messages, tool_schemas)
            messages.append(assistant_msg)

            # Check for tool calls.
            if assistant_msg.tool_calls:
                thought = assistant_msg.content or "Calling tools."
                step = Step(thought=thought, action="tool_calls")
                steps.append(step)

                # Execute all tool calls.
                results = self.execute_tool_calls(assistant_msg.tool_calls)

                # Record observations and append tool messages.
                observations: List[str] = []
                for tc, result in zip(assistant_msg.tool_calls, results):
                    observations.append(f"{tc.name}: {result}")
                    messages.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

                step.observation = "\n".join(observations)
                self.memory.short_term.add("assistant", thought)
                self.memory.short_term.add("tool_results", step.observation)

                self._logger.debug("Step %d: executed %d tool call(s).", i + 1, len(assistant_msg.tool_calls))
                continue

            # No tool calls -- check for a final answer.
            content = assistant_msg.content or ""
            answer = self._extract_final_answer(content)

            step = Step(
                thought=content,
                action="final" if answer else "respond",
                observation="Final answer produced." if answer else "No tool calls.",
            )
            steps.append(step)

            if answer:
                self._logger.info("Tool-call agent completed in %d steps.", i + 1)
                return Result(
                    output=answer,
                    steps=steps,
                    metadata={"agent": self.role, "steps_taken": i + 1},
                )

            # No final answer and no tool calls -- treat the response
            # as the best-effort output.
            self._logger.info("No tool calls or final answer; returning response.")
            return Result(
                output=content,
                steps=steps,
                metadata={"agent": self.role, "steps_taken": i + 1},
            )

        # Max steps reached.
        self._logger.warning("Tool-call loop reached max steps (%d).", steps_limit)
        last_content = messages[-1].content if messages else ""
        return Result(
            output=last_content or "",
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

        Args:
            task: The task description.
            max_steps: Optional override for the maximum number of
                steps.

        Yields:
            :class:`Step` objects.
        """
        steps_limit = max_steps or self.max_steps
        steps: List[Step] = []

        self._logger.info("Streaming tool-call loop for task: %s", task[:100])

        messages: List[Message] = [
            Message(role="system", content=self.role),
            Message(role="user", content=task),
        ]
        tool_schemas = self._get_tool_schemas()

        for i in range(steps_limit):
            assistant_msg = self._chat(messages, tool_schemas)
            messages.append(assistant_msg)

            if assistant_msg.tool_calls:
                thought = assistant_msg.content or "Calling tools."
                results = self.execute_tool_calls(assistant_msg.tool_calls)
                observations: List[str] = []
                for tc, result in zip(assistant_msg.tool_calls, results):
                    observations.append(f"{tc.name}: {result}")
                    messages.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                step = Step(
                    thought=thought,
                    action="tool_calls",
                    observation="\n".join(observations),
                )
                steps.append(step)
                yield step
                continue

            content = assistant_msg.content or ""
            answer = self._extract_final_answer(content)
            step = Step(
                thought=content,
                action="final" if answer else "respond",
                observation="Final answer produced." if answer else "No tool calls.",
            )
            steps.append(step)
            yield step
            return

    # ------------------------------------------------------------------
    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse tool calls from a raw text response.

        This is a fallback parser used when the model does not return
        structured tool calls but embeds them in text.  It looks for
        JSON blocks or ``Action: / Action Input:`` patterns.

        Args:
            response: The model's response text.

        Returns:
            A list of :class:`ToolCall` objects.
        """
        tool_calls: List[ToolCall] = []

        # Strategy 1: JSON code blocks containing tool calls.
        json_blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            tool_calls.append(
                                ToolCall(
                                    name=item["name"],
                                    arguments=item.get("arguments", item.get("parameters", {})),
                                    id=item.get("id"),
                                )
                            )
            except (json.JSONDecodeError, TypeError):
                continue

        # Strategy 2: Action: / Action Input: pairs (ReAct-style).
        if not tool_calls:
            action_matches = re.finditer(
                r"Action:\s*(.*?)(?:\n|$)",
                response,
                re.IGNORECASE,
            )
            input_matches = re.finditer(
                r"Action Input:\s*(.*?)(?:\n\s*(?:Thought:|Action:|Final Answer:)|$)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            actions = [m.group(1).strip() for m in action_matches]
            inputs = [m.group(1).strip() for m in input_matches]

            for i, action in enumerate(actions):
                params_str = inputs[i] if i < len(inputs) else ""
                try:
                    params = json.loads(params_str)
                    if not isinstance(params, dict):
                        params = {"input": params}
                except (json.JSONDecodeError, TypeError):
                    params = {"input": params_str} if params_str else {}
                tool_calls.append(ToolCall(name=action, arguments=params))

        return tool_calls

    # ------------------------------------------------------------------
    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        """Execute a batch of tool calls.

        Args:
            tool_calls: List of :class:`ToolCall` objects to execute.

        Returns:
            A list of result strings (one per tool call).
        """
        results: List[str] = []
        for tc in tool_calls:
            if tc.name.lower() in self._FINAL_ACTIONS:
                answer = tc.arguments.get("answer", str(tc.arguments))
                results.append(str(answer))
                continue

            if tc.name not in self._tool_names:
                available = ", ".join(self._tool_names) or "(none)"
                results.append(
                    f"Error: Tool '{tc.name}' is not available. Available: {available}"
                )
                continue

            result: ToolResult = self.tool_registry.execute_tool(tc.name, tc.arguments)
            if result.success:
                results.append(str(result.output))
            else:
                results.append(f"Error: {result.error}")

        self._logger.debug("Executed %d tool calls.", len(tool_calls))
        return results

    # ------------------------------------------------------------------
    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the agent's registered tools.

        Returns:
            A list of tool-description dictionaries filtered to the
            agent's own tools.
        """
        all_descs = self.tool_registry.get_tool_descriptions()
        if not all_descs:
            return []

        # Filter to only the agent's tools.
        filtered = [d for d in all_descs if d.get("name") in self._tool_names]
        return filtered or all_descs

    # ------------------------------------------------------------------
    def _chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Message:
        """Call the model's ``chat`` method.

        Args:
            messages: The conversation messages.
            tools: Optional tool schemas.

        Returns:
            The assistant's reply as a :class:`Message`.
        """
        try:
            return self.model.chat(messages, tools=tools, max_tokens=512)
        except TypeError:
            # Fallback for models without a ``tools`` parameter.
            try:
                return self.model.chat(messages)
            except TypeError:
                # Last resort: use generate.
                prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
                text = self.model.generate(prompt, max_tokens=512)
                return Message(role="assistant", content=text)

    # ------------------------------------------------------------------
    def _extract_final_answer(self, content: str) -> str:
        """Extract a final answer from the model's text content.

        Looks for a ``FINAL ANSWER:`` marker.  If found, returns the
        text after it; otherwise returns an empty string.

        Args:
            content: The model's response text.

        Returns:
            The extracted answer, or an empty string.
        """
        match = re.search(
            r"FINAL ANSWER:\s*(.*?)(?:\n|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""
