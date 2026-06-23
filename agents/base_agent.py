"""Base agent infrastructure for the TorchaVerse Agent subsystem.

This module defines the foundational building blocks for autonomous
agents:

* :class:`Step` -- a single step in an agent's execution trace.
* :class:`Result` -- the final result of an agent execution.
* :class:`ShortTermMemory` / :class:`LongTermMemory` /
  :class:`WorkingMemory` / :class:`Memory` -- a tiered memory system.
* :class:`Planner` -- task planning with ReAct, Chain-of-Thought, and
  Tree-of-Thought strategy interfaces.
* :class:`Reflector` -- self-reflection and error correction.
* :class:`BaseAgent` -- the abstract base class all agents inherit
  from.

Concrete agent implementations live in :mod:`agents.react_agent` and
:mod:`agents.tool_call_agent`.
"""

from __future__ import annotations

import abc
import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from core.tool_registry import BaseTool, Tool, ToolRegistry, ToolResult
from infrastructure.logger import get_logger

__all__ = [
    "Step",
    "Result",
    "ShortTermMemory",
    "LongTermMemory",
    "WorkingMemory",
    "Memory",
    "Planner",
    "Reflector",
    "BaseAgent",
]


# ---------------------------------------------------------------------------
# Step data class
# ---------------------------------------------------------------------------
@dataclass
class Step:
    """A single step in an agent's execution trace.

    Attributes:
        thought: The agent's reasoning for this step.
        action: The action taken (e.g. a tool name or ``"final"``).
        observation: The result of the action.
        timestamp: ISO-8601 timestamp of when the step was recorded.
    """

    thought: str = ""
    action: str = ""
    observation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------
@dataclass
class Result:
    """The result of an agent execution.

    Attributes:
        output: The final output text.
        steps: The execution trace as a list of :class:`Step`.
        metadata: Additional metadata (agent name, timing, etc.).
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
# Memory system
# ---------------------------------------------------------------------------
class ShortTermMemory:
    """Short-term (working) conversation memory.

    Stores the current conversation context as a list of messages.
    Each message is a dict with ``role`` and ``content`` keys.

    Args:
        max_messages: Maximum number of messages to retain.  Older
            messages are dropped when the limit is exceeded.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages: int = max_messages
        self._messages: List[Dict[str, str]] = []
        self._logger = get_logger("ShortTermMemory")

    def add(self, role: str, content: str) -> None:
        """Add a message to short-term memory.

        Args:
            role: The message role (``"user"``, ``"assistant"``,
                ``"system"``, ``"tool"``).
            content: The message content.
        """
        self._messages.append({"role": role, "content": content})
        # Enforce the sliding window.
        if len(self._messages) > self.max_messages:
            excess = len(self._messages) - self.max_messages
            self._messages = self._messages[excess:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Return the current message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """Long-term persistent memory backed by a vector store.

    Stores text entries with their embeddings and supports
    similarity-based retrieval.  Requires an embedding function to
    convert text to vectors.

    Args:
        vector_store: An optional pre-configured vector store.  When
            ``None`` an :class:`~torcha_verse.rag.vectorstore.InMemoryVectorStore`
            is created lazily.
        embed_fn: A callable that converts text to a ``torch.Tensor``
            embedding.  Required for ``store`` and ``retrieve``.
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        embed_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._embed_fn: Optional[Callable[[str], Any]] = embed_fn
        self._vector_store: Any = vector_store
        self._logger = get_logger("LongTermMemory")

        if self._vector_store is None:
            # Lazy import to avoid a hard cross-subsystem dependency at
            # module load time.
            try:
                from rag.vectorstore.vector_store import InMemoryVectorStore

                self._vector_store = InMemoryVectorStore()
            except ImportError:
                self._logger.warning(
                    "Could not import InMemoryVectorStore; long-term "
                    "memory will use a simple list-based fallback."
                )
                self._vector_store = None
                self._fallback: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store a text entry in long-term memory.

        Args:
            text: The text to store.
            metadata: Optional metadata dictionary.

        Returns:
            The id of the stored entry, or ``None`` on failure.
        """
        meta = metadata or {}

        if self._vector_store is not None and self._embed_fn is not None:
            import torch

            vector = self._embed_fn(text)
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector)
            ids = self._vector_store.add([vector], [meta], [text])
            self._logger.debug("Stored entry in vector-backed long-term memory.")
            return ids[0] if ids else None

        # Fallback: simple list storage.
        entry = {"text": text, "metadata": meta}
        if hasattr(self, "_fallback"):
            self._fallback.append(entry)
        self._logger.debug("Stored entry in list-based long-term memory.")
        return None

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant entries from long-term memory.

        Args:
            query: The query text.
            top_k: Maximum number of entries.

        Returns:
            A list of result dictionaries with ``content``, ``score``,
            and ``metadata`` keys.
        """
        if self._vector_store is not None and self._embed_fn is not None:
            import torch

            vector = self._embed_fn(query)
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector)
            results = self._vector_store.search(vector, top_k=top_k)
            return [
                {
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

        # Fallback: return all entries (no real similarity search).
        if hasattr(self, "_fallback"):
            return [
                {"content": e["text"], "score": 0.0, "metadata": e["metadata"]}
                for e in self._fallback[:top_k]
            ]
        return []

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all entries."""
        if self._vector_store is not None:
            self._vector_store.clear()
        if hasattr(self, "_fallback"):
            self._fallback.clear()


class WorkingMemory:
    """Working memory for task-execution intermediate state.

    A simple key-value store used to pass data between steps during
    task execution.

    Args:
        max_keys: Maximum number of keys to retain.
    """

    def __init__(self, max_keys: int = 100) -> None:
        self.max_keys: int = max_keys
        self._data: Dict[str, Any] = {}
        self._logger = get_logger("WorkingMemory")

    def set(self, key: str, value: Any) -> None:
        """Store a value under ``key``.

        Args:
            key: The key.
            value: The value.
        """
        self._data[key] = value
        # Enforce the key limit (FIFO eviction).
        if len(self._data) > self.max_keys:
            oldest = next(iter(self._data))
            del self._data[oldest]

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by ``key``.

        Args:
            key: The key.
            default: Default value when the key is absent.

        Returns:
            The stored value or ``default``.
        """
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to remove.

        Returns:
            ``True`` if the key existed.
        """
        return self._data.pop(key, None) is not None

    def keys(self) -> List[str]:
        """Return all keys."""
        return list(self._data.keys())

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of all data."""
        return dict(self._data)


class Memory:
    """Composite memory system combining short-term, long-term, and working memory.

    Args:
        short_term: Optional pre-configured :class:`ShortTermMemory`.
        long_term: Optional pre-configured :class:`LongTermMemory`.
        working: Optional pre-configured :class:`WorkingMemory`.
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        working: Optional[WorkingMemory] = None,
    ) -> None:
        self.short_term: ShortTermMemory = short_term or ShortTermMemory()
        self.long_term: LongTermMemory = long_term or LongTermMemory()
        self.working: WorkingMemory = working or WorkingMemory()

    def reset(self) -> None:
        """Clear all memory tiers."""
        self.short_term.clear()
        self.long_term.clear()
        self.working.clear()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner:
    """Task planner supporting multiple planning strategies.

    Defines interfaces for three planning paradigms:

    * **ReAct** -- interleave reasoning and acting step by step.
    * **Chain-of-Thought (CoT)** -- linear step-by-step reasoning.
    * **Tree-of-Thought (ToT)** -- explore multiple reasoning branches.

    When a ``model`` (with a ``generate`` method) is provided, plans are
    generated by prompting the model.  Otherwise, template-based plans
    are returned.

    Args:
        model: An object with a ``generate(prompt, max_tokens) -> str``
            method.
        strategy: Default planning strategy.
    """

    STRATEGIES: tuple = ("react", "cot", "tot")

    def __init__(
        self,
        model: Optional[Any] = None,
        strategy: str = "react",
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}."
            )
        self.model: Optional[Any] = model
        self.strategy: str = strategy
        self._logger = get_logger("Planner")

    # ------------------------------------------------------------------
    def plan(
        self,
        task: str,
        strategy: Optional[str] = None,
        max_steps: int = 5,
    ) -> List[str]:
        """Generate a plan (list of step descriptions) for ``task``.

        Args:
            task: The task description.
            strategy: Planning strategy override.
            max_steps: Maximum number of planning steps.

        Returns:
            A list of step-description strings.
        """
        strat = strategy or self.strategy
        if strat == "react":
            return self._plan_react(task, max_steps)
        elif strat == "cot":
            return self._plan_cot(task, max_steps)
        elif strat == "tot":
            return self._plan_tot(task, max_steps)
        else:
            raise ValueError(f"Unknown strategy: {strat}")

    # ------------------------------------------------------------------
    def _plan_react(self, task: str, max_steps: int) -> List[str]:
        """Generate a ReAct-style plan.

        ReAct interleaves thought and action: each step considers what
        to do next and which tool to use.

        Args:
            task: The task description.
            max_steps: Maximum steps.

        Returns:
            A list of step descriptions.
        """
        if self.model is None:
            return [
                f"Step {i + 1}: Analyse the task and decide on the next action."
                for i in range(max_steps)
            ]

        prompt = (
            f"Break down the following task into at most {max_steps} steps. "
            f"For each step, describe the thought and the action to take.\n\n"
            f"Task: {task}\n\nPlan:"
        )
        response = self.model.generate(prompt, max_tokens=256)
        return self._parse_steps(response, max_steps)

    # ------------------------------------------------------------------
    def _plan_cot(self, task: str, max_steps: int) -> List[str]:
        """Generate a Chain-of-Thought plan.

        CoT uses linear step-by-step reasoning without explicit actions.

        Args:
            task: The task description.
            max_steps: Maximum steps.

        Returns:
            A list of step descriptions.
        """
        if self.model is None:
            return [
                f"Step {i + 1}: Reason about the task step by step."
                for i in range(max_steps)
            ]

        prompt = (
            f"Think step by step about how to solve the following task. "
            f"Provide at most {max_steps} reasoning steps.\n\n"
            f"Task: {task}\n\nReasoning:"
        )
        response = self.model.generate(prompt, max_tokens=256)
        return self._parse_steps(response, max_steps)

    # ------------------------------------------------------------------
    def _plan_tot(self, task: str, max_steps: int) -> List[str]:
        """Generate a Tree-of-Thought plan.

        ToT explores multiple reasoning branches and selects the most
        promising path.

        Args:
            task: The task description.
            max_steps: Maximum steps.

        Returns:
            A list of step descriptions.
        """
        if self.model is None:
            return [
                f"Step {i + 1}: Explore reasoning branch {i + 1} and evaluate."
                for i in range(max_steps)
            ]

        prompt = (
            f"Explore multiple approaches to solve the following task. "
            f"Consider at most {max_steps} branches, evaluate each, and "
            f"select the best path.\n\n"
            f"Task: {task}\n\nBranches:"
        )
        response = self.model.generate(prompt, max_tokens=256)
        return self._parse_steps(response, max_steps)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_steps(response: str, max_steps: int) -> List[str]:
        """Parse a model response into a list of step descriptions.

        Args:
            response: The model's text response.
            max_steps: Maximum number of steps to return.

        Returns:
            A list of step-description strings.
        """
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]
        steps = [l for l in lines if l]
        return steps[:max_steps] if steps else ["Analyse the task."]


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
class Reflector:
    """Self-reflection and error-correction component.

    Reviews an agent's execution trace and provides a critique that can
    be used to improve subsequent attempts.

    Args:
        model: An object with a ``generate(prompt, max_tokens) -> str``
            method.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model: Optional[Any] = model
        self._logger = get_logger("Reflector")

    # ------------------------------------------------------------------
    def reflect(self, task: str, steps: List[Step]) -> str:
        """Reflect on the execution and return a critique.

        Args:
            task: The original task.
            steps: The execution trace.

        Returns:
            A reflection string.  When no model is available a summary
            is returned.
        """
        if self.model is None:
            return self._summarise(task, steps)

        prompt = self._build_reflection_prompt(task, steps)
        reflection = self.model.generate(prompt, max_tokens=256)
        self._logger.debug("Generated reflection (%d chars).", len(reflection))
        return reflection

    # ------------------------------------------------------------------
    def should_retry(self, task: str, steps: List[Step]) -> bool:
        """Determine whether the task should be retried.

        Heuristically checks the reflection for error indicators.

        Args:
            task: The original task.
            steps: The execution trace.

        Returns:
            ``True`` if a retry is recommended.
        """
        reflection = self.reflect(task, steps).lower()
        indicators = ("retry", "error", "incorrect", "failed", "try again", "wrong")
        return any(ind in reflection for ind in indicators)

    # ------------------------------------------------------------------
    def _build_reflection_prompt(self, task: str, steps: List[Step]) -> str:
        """Build the reflection prompt.

        Args:
            task: The original task.
            steps: The execution trace.

        Returns:
            The prompt string.
        """
        trace = "\n".join(
            f"  Step {i + 1}:\n"
            f"    Thought: {s.thought}\n"
            f"    Action: {s.action}\n"
            f"    Observation: {s.observation}"
            for i, s in enumerate(steps)
        )
        return (
            f"Review the following agent execution trace and provide a "
            f"brief critique.  Identify any errors and suggest whether the "
            f"task should be retried.\n\n"
            f"Task: {task}\n\n"
            f"Trace:\n{trace}\n\nCritique:"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _summarise(task: str, steps: List[Step]) -> str:
        """Produce a simple summary when no model is available.

        Args:
            task: The original task.
            steps: The execution trace.

        Returns:
            A summary string.
        """
        if not steps:
            return "No steps were executed; nothing to reflect on."
        actions = [s.action for s in steps if s.action]
        return (
            f"Executed {len(steps)} step(s) for task '{task[:50]}...'. "
            f"Actions taken: {', '.join(actions) or 'none'}."
        )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------
class BaseAgent(abc.ABC):
    """Abstract base class for all agents.

    An agent is defined by its **role** (persona), the **model** it
    uses for reasoning, the **tools** it can invoke, and its **memory**
    system.

    Args:
        role: The agent's role / persona description.
        model: The language model used for reasoning.  Must have a
            ``generate(prompt, max_tokens, ...) -> str`` method.
        tools: Optional list of tool names, :class:`Tool` objects, or
            :class:`BaseTool` instances available to the agent.
        memory: Optional pre-configured :class:`Memory`.  A fresh one
            is created when ``None``.
        tool_registry: Optional :class:`ToolRegistry` for tool
            execution.  A new one is created when ``None``.
        max_steps: Default maximum reasoning steps.
    """

    def __init__(
        self,
        role: str,
        model: Any,
        tools: Optional[Sequence[Union[str, Tool, BaseTool]]] = None,
        memory: Optional[Memory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
    ) -> None:
        self.role: str = role
        self.model: Any = model
        self.memory: Memory = memory or Memory()
        self.tool_registry: ToolRegistry = tool_registry or ToolRegistry()
        self.max_steps: int = max_steps
        self._tool_names: List[str] = self._normalise_tools(tools)
        self._logger = get_logger(f"Agent[{role}]")

    # ------------------------------------------------------------------
    @property
    def tools(self) -> List[str]:
        """The names of tools available to this agent."""
        return list(self._tool_names)

    # ------------------------------------------------------------------
    def _normalise_tools(
        self,
        tools: Optional[Sequence[Union[str, Tool, BaseTool]]],
    ) -> List[str]:
        """Normalise the ``tools`` argument into a list of tool names.

        Args:
            tools: A sequence of tool names, :class:`Tool`, or
                :class:`BaseTool` objects.

        Returns:
            A list of tool-name strings.
        """
        names: List[str] = []
        for tool in (tools or []):
            if isinstance(tool, str):
                names.append(tool)
            elif isinstance(tool, Tool):
                names.append(tool.name)
            elif isinstance(tool, BaseTool):
                names.append(tool.name or tool.__class__.__name__)
            else:
                self._logger.warning("Ignoring unrecognised tool type: %s", type(tool))
        return names

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def run(self, task: str, max_steps: Optional[int] = None) -> Result:
        """Execute the agent on a task.

        Args:
            task: The task description.
            max_steps: Optional override for the maximum number of
                steps.

        Returns:
            A :class:`Result` containing the output and execution
            trace.
        """
        ...

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def stream(self, task: str, max_steps: Optional[int] = None) -> Iterator[Step]:
        """Stream execution steps as they are produced.

        Args:
            task: The task description.
            max_steps: Optional override for the maximum number of
                steps.

        Yields:
            :class:`Step` objects as they are produced.
        """
        ...

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the agent's memory and state."""
        self.memory.reset()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(role={self.role!r}, "
            f"tools={self._tool_names}, max_steps={self.max_steps})"
        )
