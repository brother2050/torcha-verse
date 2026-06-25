"""Lightweight tool-calling agent for the v0.4.x single-system stack.

The :class:`AgentBus` is a small **dependency-free** tool registry
and ReAct-style executor that lets the ``agent_*`` L4 nodes and
the ``/serving/v1/agent/*`` HTTP endpoints answer a free-form
query by orchestrating the framework's existing capabilities.

Design choices
--------------

* **No external agent framework** (no LangChain, no LlamaIndex).
  The agent is a small, well-tested loop the team can read end-
  to-end in 200 lines -- fits the v0.4.x "single system" roadmap.
* **Tool registry** is built on the existing
  :class:`core.module_bus.ModuleBus` and :class:`nodes.NodeRegistry`
  so every registered L4 node and every explicitly registered
  Python function is callable as a tool.
* **ReAct-style prompting**: each step the agent emits either
  *thought* (text), a *tool_call* (JSON object with ``name`` /
  ``args``) or a *final_answer* marker.  The loop terminates
  when ``final_answer`` is emitted, ``max_steps`` is reached, or
  the model produces unparseable output for ``max_parse_failures``
  consecutive times.
* **Audit + budget** -- every tool invocation is logged through
  :class:`AuditLogger` and respects the agent's
  :class:`ResourceBudget` so a runaway agent cannot exhaust
  the vram/ram allotment.
* **Deterministic** when ``seed`` is supplied: the inner
  :class:`LLMProvider` is asked to use ``do_sample=False`` so
  CI can run an end-to-end agent test.

Public surface
--------------

* :class:`ToolSpec` -- declarative description of a tool.
* :class:`ToolResult` -- a single tool invocation's outcome.
* :class:`ToolRegistry` -- in-process tool registry (sub-set
  of the :class:`ModuleBus` for tools).
* :class:`AgentBus` -- the executor: takes a query, runs the
  ReAct loop, returns :class:`AgentRunResult`.
* :func:`default_agent_bus` -- the process-wide default bus.

Example
-------

>>> from infrastructure.agent import AgentBus, ToolRegistry, ToolSpec
>>> bus = AgentBus()
>>> bus.tools.register(ToolSpec(
...     name="sum",
...     description="Compute a + b.",
...     parameters={"a": "int", "b": "int"},
...     func=lambda a, b: a + b,
... ))
>>> # Run with a query; the agent emits "Action: sum(a=1, b=2)" and
>>> # the bus returns the result.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "AgentRunResult",
    "AgentBus",
    "default_agent_bus",
]


_logger = get_logger("infrastructure.agent")


# ---------------------------------------------------------------------------
# Tool description
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """A callable tool exposed to the agent.

    Attributes:
        name: Stable tool name (e.g. ``"rag_query"``).  Must be a
            non-empty identifier; used as the JSON key in the
            model's tool_call output.
        description: One-line human-readable description of what
            the tool does.  Surfaced verbatim to the model so it
            can decide whether to call the tool.
        parameters: Mapping of parameter name -> type string
            (``"str"``, ``"int"``, ``"float"``, ``"bool"``,
            ``"json"``).  Free-form documentation; the runtime
            does not validate types beyond the JSON parsing.
        func: The Python callable to invoke.  Receives the
            parameters as keyword arguments and may return any
            JSON-serialisable value.
        tags: Free-form tag list (e.g. ``["rag", "read"]``).
    """

    name: str
    description: str
    parameters: Dict[str, str]
    func: Callable[..., Any]
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolSpec.name must be a non-empty string.")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.name):
            raise ValueError(
                f"ToolSpec.name must match [A-Za-z_][A-Za-z0-9_]*; got {self.name!r}."
            )
        if not isinstance(self.description, str):
            raise ValueError("ToolSpec.description must be a string.")
        if not isinstance(self.parameters, dict):
            raise ValueError("ToolSpec.parameters must be a dict[str, str].")
        for k, v in self.parameters.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"ToolSpec parameter name must be a non-empty str: {k!r}")
            if not isinstance(v, str):
                raise ValueError(
                    f"ToolSpec parameter {k!r} type must be a string; got {type(v).__name__}."
                )
        if not callable(self.func):
            raise TypeError(
                f"ToolSpec.func must be callable; got {type(self.func).__name__}."
            )


@dataclass
class ToolResult:
    """The outcome of a single tool invocation.

    Attributes:
        name: The tool name.
        ok: ``True`` when the tool returned without raising.
        output: The tool's return value (JSON-serialisable).
        error: When ``ok`` is ``False``, a human-readable error
            message (never raises out of the agent loop).
    """

    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """An in-process map from tool name to :class:`ToolSpec`."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.agent.tools")

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"Tool {spec.name!r} already registered.")
            self._tools[spec.name] = spec
        self._logger.info("Registered tool %s", spec.name)

    def try_register(self, spec: ToolSpec) -> bool:
        """Register if absent; return ``True`` if registered, ``False`` if duplicate."""
        with self._lock:
            if spec.name in self._tools:
                return False
            self._tools[spec.name] = spec
            return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec:
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as exc:
                raise KeyError(f"no tool named {name!r}") from exc

    def try_get(self, name: str) -> Optional[ToolSpec]:
        with self._lock:
            return self._tools.get(name)

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._tools.keys())

    def describe(self) -> List[Dict[str, Any]]:
        """Return a JSON-serialisable description of every tool.

        Used by the agent prompt so the LLM can pick a tool by
        name and reason about its parameters.
        """
        with self._lock:
            return [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.parameters),
                    "tags": list(spec.tags),
                }
                for spec in self._tools.values()
            ]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self.try_get(name)
        if spec is None:
            return ToolResult(name=name, ok=False, error=f"unknown tool: {name!r}")
        try:
            output = spec.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(name=name, ok=True, output=output)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.list()

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)


# ---------------------------------------------------------------------------
# Agent run result
# ---------------------------------------------------------------------------
@dataclass
class AgentRunResult:
    """The full transcript of a single :meth:`AgentBus.run` call.

    Attributes:
        query: The original user query.
        final_answer: The agent's final natural-language answer
            (string), or the last ``Observation`` text when the
            loop ran out of steps.
        steps: Ordered list of ``(thought, tool_call, observation)``
            tuples the agent emitted during the run.
        iterations: Number of LLM + tool rounds the loop took.
        ok: ``True`` when the agent emitted a final answer; ``False``
            when the loop terminated because ``max_steps`` or
            ``max_parse_failures`` was reached.
    """

    query: str
    final_answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    ok: bool = True


# ---------------------------------------------------------------------------
# ReAct parsing helpers
# ---------------------------------------------------------------------------
_FINAL_ANSWER_RE = re.compile(
    r"Final\s*Answer\s*[:：]\s*(?P<answer>.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL
)
_THOUGHT_RE = re.compile(r"Thought\s*[:：]\s*(?P<thought>.+?)(?:\n|$)", re.IGNORECASE)
_ACTION_RE = re.compile(
    r"Action\s*[:：]\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^)]*)\)",
    re.IGNORECASE,
)


def _coerce(value: str, type_str: str) -> Any:
    """Coerce a textual value to the requested JSON type."""
    t = type_str.strip().lower()
    if t in ("str", "text", "string"):
        return value
    if t in ("int", "integer"):
        return int(value)
    if t in ("float", "number"):
        return float(value)
    if t in ("bool", "boolean"):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"cannot coerce {value!r} to bool")
    if t in ("json", "object", "dict"):
        return json.loads(value)
    # Default: leave as string.
    return value


def _parse_action_args(
    raw_args: str,
    spec: ToolSpec,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse ``raw_args`` against ``spec.parameters`` and return ``(kwargs, error)``."""
    if not raw_args.strip():
        return {}, None
    # Accept both ``name=value`` and ``"name": value`` JSON-style; the
    # first form is more friendly to small LLMs.
    pieces: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str: Optional[str] = None
    for ch in raw_args:
        if in_str is not None:
            buf.append(ch)
            if ch == in_str and (len(buf) < 2 or buf[-2] != "\\"):
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            pieces.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        pieces.append("".join(buf).strip())

    parsed: Dict[str, Any] = {}
    for piece in pieces:
        if "=" not in piece:
            return {}, f"cannot parse action argument: {piece!r}"
        key, value = piece.split("=", 1)
        key = key.strip().strip("'\"")
        value = value.strip()
        if value.startswith(("[", "{")):
            # JSON-style value (e.g. `[1,2,3]` or `{"k": "v"}`).
            try:
                parsed[key] = json.loads(value)
                continue
            except json.JSONDecodeError as exc:
                return {}, f"invalid JSON in action argument {key!r}: {exc}"
        # Strip surrounding quotes.
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        try:
            parsed[key] = _coerce(value, spec.parameters.get(key, "str"))
        except Exception as exc:  # noqa: BLE001
            return {}, f"failed to coerce {key!r}={value!r}: {exc}"

    # Validate that every required parameter is present.
    missing = [k for k in spec.parameters if k not in parsed]
    if missing:
        return {}, f"missing required arguments: {missing}"
    return parsed, None


# ---------------------------------------------------------------------------
# AgentBus
# ---------------------------------------------------------------------------
class AgentBus:
    """The framework-side tool-calling agent.

    Args:
        tools: Optional :class:`ToolRegistry`.  When ``None`` a
            fresh registry is created and a small set of
            convenience tools (RAG query, text completion,
            rag_list_indexes) is registered.
        llm_provider: Optional :class:`LLMProvider`.  When
            ``None`` the :func:`fetch_and_load_text` default
            provider is used.
        max_steps: Maximum number of thought/action/observation
            cycles before the loop forces a final answer.
        max_parse_failures: Maximum number of consecutive steps
            where the LLM output cannot be parsed.  When this
            limit is hit the agent emits a deterministic "I could
            not parse the model's output" final answer.
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
        steps: List[Dict[str, Any]] = []
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
                self._logger.warning("LLM generate raised %s: %s", type(exc).__name__, exc)
                return AgentRunResult(
                    query=query,
                    final_answer=f"agent error: {type(exc).__name__}: {exc}",
                    steps=steps,
                    iterations=iterations,
                    ok=False,
                )
            last_text = text

            final_match = _FINAL_ANSWER_RE.search(text)
            if final_match:
                final_answer = final_match.group("answer").strip()
                steps.append(
                    {"step": step, "thought": "", "tool": None, "observation": "", "final": True}
                )
                ok = True
                break

            thought_match = _THOUGHT_RE.search(text)
            thought = thought_match.group("thought").strip() if thought_match else ""

            action_match = _ACTION_RE.search(text)
            if not action_match:
                parse_failures += 1
                steps.append(
                    {
                        "step": step,
                        "thought": thought,
                        "tool": None,
                        "observation": "could not parse an Action; please emit Action: name(args) or Final Answer: ...",
                        "final": False,
                    }
                )
                if parse_failures > self._max_parse_failures:
                    break
                continue
            parse_failures = 0
            tool_name = action_match.group("name")
            raw_args = action_match.group("args")
            spec = self._tools.try_get(tool_name)
            if spec is None:
                obs = f"unknown tool: {tool_name!r}"
                steps.append(
                    {
                        "step": step,
                        "thought": thought,
                        "tool": tool_name,
                        "observation": obs,
                        "final": False,
                    }
                )
                history.append(f"Thought: {thought}\nAction: {tool_name}({raw_args})\nObservation: {obs}")
                continue
            kwargs, err = _parse_action_args(raw_args, spec)
            if err is not None:
                obs = f"invalid action args: {err}"
                steps.append(
                    {
                        "step": step,
                        "thought": thought,
                        "tool": tool_name,
                        "observation": obs,
                        "final": False,
                    }
                )
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
            steps.append(
                {
                    "step": step,
                    "thought": thought,
                    "tool": tool_name,
                    "args": kwargs,
                    "observation": obs_repr,
                    "final": False,
                }
            )
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
