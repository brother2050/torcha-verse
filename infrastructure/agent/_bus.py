"""The :class:`AgentBus` core + the process-wide default singleton.

The :class:`AgentBus` is the framework-side tool-calling agent
that the ``agent_*`` L4 nodes and the ``/serving/v1/agent/*``
HTTP endpoints call to answer free-form queries.

This module depends on three sibling modules:

* :mod:`._types` -- :class:`ToolSpec` / :class:`ToolResult` /
  :class:`AgentRunResult`.
* :mod:`._registry` -- :class:`ToolRegistry`.
* :mod:`._parse` -- ReAct-style regex parsing + arg coercion.

The :func:`default_agent_bus` and :func:`reset_default_agent_bus`
helpers expose a process-wide singleton used by the serving
endpoints and the L4 agent nodes.
"""

from __future__ import annotations

import json
import threading
from typing import Any, List, Optional, Sequence

from ..logger import get_logger
from ._parse import (
    ACTION_RE,
    FINAL_ANSWER_RE,
    THOUGHT_RE,
    parse_action_args,
)
from ._registry import ToolRegistry
from ._types import AgentRunResult, ToolSpec

__all__ = ["AgentBus", "default_agent_bus", "reset_default_agent_bus"]


class AgentBus:
    """The framework-side tool-calling agent.

    Args:
        tools: Optional :class:`ToolRegistry`.  When ``None`` a
            fresh registry is created and a small set of
            convenience tools (RAG query, text completion,
            ``list_rag_indexes``) is registered.
        llm_provider: Optional :class:`LLMProvider`.  When
            ``None`` the :func:`fetch_and_load_text` default
            provider is used.
        max_steps: Maximum number of thought/action/observation
            cycles before the loop forces a final answer.
        max_parse_failures: Maximum number of consecutive
            steps where the LLM output cannot be parsed.  When
            this limit is hit the agent emits a deterministic
            "I could not parse the model's output" final answer.
        system_prompt: Override the default system prompt
            (handy for CI smoke tests).
    """

    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a tool-calling agent.  For every step you must emit a "
        "single Thought, optionally an Action in the form "
        "``Action: name(key=value, ...)``, and when you have enough "
        "information emit ``Final Answer: <text>``.  Use only the tools "
        "provided; do not invent new tools.  Keep each Thought to one "
        "sentence."
    )

    def __init__(
        self,
        *,
        tools: Optional[ToolRegistry] = None,
        llm_provider: Any = None,
        max_steps: int = 6,
        max_parse_failures: int = 2,
        system_prompt: Optional[str] = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}.")
        if max_parse_failures < 0:
            raise ValueError(
                f"max_parse_failures must be >= 0, got {max_parse_failures}."
            )
        self._tools: ToolRegistry = tools or ToolRegistry()
        self._max_steps: int = int(max_steps)
        self._max_parse_failures: int = int(max_parse_failures)
        self._system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.agent.bus")
        if llm_provider is None:
            from models.providers import fetch_and_load_text  # local import
            self._llm_provider = fetch_and_load_text()
        else:
            self._llm_provider = llm_provider
        self._register_default_tools()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    # ------------------------------------------------------------------
    # Default tool registration
    # ------------------------------------------------------------------
    def _register_default_tools(self) -> None:
        """Register a small set of convenience tools.

        These wrap the RAG / text completion surfaces so the
        default agent is useful out of the box.  Callers that
        want a clean registry can pass their own.
        """
        from infrastructure.rag import default_rag_index_store

        rag_store = default_rag_index_store()

        def _rag_query(index_name: str, query: str, top_k: int = 3) -> str:
            idx = rag_store.get(index_name)
            from infrastructure.rag import RAGRetriever
            retriever = RAGRetriever(idx, default_top_k=top_k)
            _, context = retriever.retrieve_with_context(query, top_k=top_k)
            return context

        def _list_rag_indexes() -> str:
            return json.dumps(rag_store.list())

        def _text_complete(prompt: str, max_new_tokens: int = 64) -> str:
            return self._llm_provider.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        self._tools.try_register(
            ToolSpec(
                name="rag_query",
                description=(
                    "Query a RAG index for the top-k matching chunks and "
                    "return an assembled context string.  Use when the user "
                    "asks a question that may be answered from indexed documents."
                ),
                parameters={"index_name": "str", "query": "str", "top_k": "int"},
                func=_rag_query,
                tags=["rag", "read"],
            )
        )
        self._tools.try_register(
            ToolSpec(
                name="list_rag_indexes",
                description="List the names of all RAG indexes in the process-wide store.",
                parameters={},
                func=_list_rag_indexes,
                tags=["rag", "read"],
            )
        )
        self._tools.try_register(
            ToolSpec(
                name="text_complete",
                description="Run a non-conversational text completion against the default LLM provider.",
                parameters={"prompt": "str", "max_new_tokens": "int"},
                func=_text_complete,
                tags=["text", "read"],
            )
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        *,
        max_steps: Optional[int] = None,
        temperature: float = 0.0,
    ) -> AgentRunResult:
        """Run the ReAct loop and return the final answer.

        Args:
            query: The user query.
            max_steps: Optional override of the constructor's
                ``max_steps`` for this single run.
            temperature: Sampling temperature for the inner
                :class:`LLMProvider`.  Default ``0.0`` for
                deterministic CI runs.

        Returns:
            An :class:`AgentRunResult` with the transcript.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        steps: List[dict] = []
        iterations = 0
        parse_failures = 0
        last_text: str = ""
        budget = int(max_steps) if max_steps is not None else self._max_steps
        budget = min(budget, self._max_steps)
        history: List[str] = []
        tools_desc = json.dumps(self._tools.describe(), ensure_ascii=False)
        final_answer: Optional[str] = None
        ok = False

        for step in range(budget):
            iterations += 1
            prompt = self._build_prompt(query, history, tools_desc)
            try:
                text = self._llm_provider.generate(
                    prompt,
                    max_new_tokens=192,
                    do_sample=(temperature > 0.0),
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "LLM generate raised %s: %s", type(exc).__name__, exc,
                )
                return AgentRunResult(
                    query=query,
                    final_answer=f"agent error: {type(exc).__name__}: {exc}",
                    steps=steps,
                    iterations=iterations,
                    ok=False,
                )
            last_text = text

            final_match = FINAL_ANSWER_RE.search(text)
            if final_match:
                final_answer = final_match.group("answer").strip()
                steps.append({
                    "step": step, "thought": "",
                    "tool": None, "observation": "", "final": True,
                })
                ok = True
                break

            thought_match = THOUGHT_RE.search(text)
            thought = thought_match.group("thought").strip() if thought_match else ""

            action_match = ACTION_RE.search(text)
            if not action_match:
                parse_failures += 1
                steps.append({
                    "step": step, "thought": thought, "tool": None,
                    "observation": (
                        "could not parse an Action; please emit "
                        "Action: name(args) or Final Answer: ..."
                    ),
                    "final": False,
                })
                if parse_failures > self._max_parse_failures:
                    break
                continue
            parse_failures = 0
            tool_name = action_match.group("name")
            raw_args = action_match.group("args")
            spec = self._tools.try_get(tool_name)
            if spec is None:
                obs = f"unknown tool: {tool_name!r}"
                steps.append({
                    "step": step, "thought": thought, "tool": tool_name,
                    "observation": obs, "final": False,
                })
                history.append(
                    f"Thought: {thought}\nAction: {tool_name}({raw_args})\nObservation: {obs}"
                )
                continue
            kwargs, err = parse_action_args(raw_args, spec)
            if err is not None:
                obs = f"invalid action args: {err}"
                steps.append({
                    "step": step, "thought": thought, "tool": tool_name,
                    "observation": obs, "final": False,
                })
                history.append(
                    f"Thought: {thought}\nAction: {tool_name}({raw_args})\nObservation: {obs}"
                )
                continue
            result = self._tools.invoke(tool_name, **kwargs)
            obs_repr = (
                json.dumps(result.output, ensure_ascii=False, default=str)
                if result.ok
                else f"ERROR: {result.error}"
            )
            steps.append({
                "step": step, "thought": thought, "tool": tool_name,
                "args": kwargs, "observation": obs_repr, "final": False,
            })
            history.append(
                f"Thought: {thought}\nAction: {tool_name}({raw_args})\nObservation: {obs_repr}"
            )

        if final_answer is None:
            final_answer = last_text.strip() or "I could not produce a final answer."
        return AgentRunResult(
            query=query,
            final_answer=final_answer,
            steps=steps,
            iterations=iterations,
            ok=ok,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        query: str,
        history: Sequence[str],
        tools_desc: str,
    ) -> str:
        """Construct the full LLM prompt for one step.

        The prompt is plain-text ReAct format -- no chat
        template -- so the same code path works for
        instruct-tuned and base models alike.
        """
        parts: List[str] = [self._system_prompt, ""]
        parts.append(f"Available tools (JSON): {tools_desc}")
        parts.append("")
        if history:
            parts.append("Previous steps:")
            parts.extend(history)
            parts.append("")
        parts.append(f"Question: {query}")
        parts.append("")
        parts.append("Your next step:")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Process-wide default
# ---------------------------------------------------------------------------
_default_agent_bus: Optional[AgentBus] = None
_default_agent_lock: threading.Lock = threading.Lock()


def default_agent_bus() -> AgentBus:
    """Return the process-wide default :class:`AgentBus` (lazy-initialised)."""
    global _default_agent_bus
    with _default_agent_lock:
        if _default_agent_bus is None:
            _default_agent_bus = AgentBus()
        return _default_agent_bus


def reset_default_agent_bus() -> None:
    """Drop the cached default :class:`AgentBus` (test helper)."""
    global _default_agent_bus
    with _default_agent_lock:
        _default_agent_bus = None
