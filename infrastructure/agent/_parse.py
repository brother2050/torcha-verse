"""ReAct-style parsing helpers for :class:`AgentBus`.

The :class:`AgentBus` loop relies on a small set of pure
parsing functions to extract ``Thought`` / ``Action`` /
``Final Answer`` fields from the LLM's plain-text output:

* :data:`FINAL_ANSWER_RE` / :data:`THOUGHT_RE` / :data:`ACTION_RE`
  -- the three compiled regexes.
* :func:`coerce_value` -- convert a textual value to the
  declared JSON type.
* :func:`parse_action_args` -- parse the ``Action: name(args)``
  payload against a :class:`ToolSpec`.

These helpers are factored out so the bus loop can stay focused
on orchestration and so the parsing logic is unit-testable
without spinning up an :class:`AgentBus`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ._types import ToolSpec

__all__ = [
    "FINAL_ANSWER_RE",
    "THOUGHT_RE",
    "ACTION_RE",
    "coerce_value",
    "parse_action_args",
]


#: Regex that matches the ``Final Answer: <text>`` marker.
FINAL_ANSWER_RE = re.compile(
    r"Final\s*Answer\s*[:：]\s*(?P<answer>.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
#: Regex that matches the ``Thought: <text>`` marker.
THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(?P<thought>.+?)(?:\n|$)", re.IGNORECASE,
)
#: Regex that matches the ``Action: name(key=value, ...)`` payload.
ACTION_RE = re.compile(
    r"Action\s*[:：]\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^)]*)\)",
    re.IGNORECASE,
)


def coerce_value(value: str, type_str: str) -> Any:
    """Coerce a textual value to the requested JSON type.

    Supported type strings:

    * ``"str"`` / ``"text"`` / ``"string"`` -- pass-through.
    * ``"int"`` / ``"integer"`` -- :func:`int`.
    * ``"float"`` / ``"number"`` -- :func:`float`.
    * ``"bool"`` / ``"boolean"`` -- accepts ``true``/``false``/
      ``1``/``0``/``yes``/``no`` (case-insensitive).
    * ``"json"`` / ``"object"`` / ``"dict"`` -- :func:`json.loads`.

    Args:
        value:   The textual value to coerce.
        type_str: The declared JSON type.

    Returns:
        The coerced Python value.

    Raises:
        ValueError: When ``type_str`` is ``"bool"`` and the
            value is not recognised, or when ``type_str`` is
            ``"int"`` / ``"float"`` and the value cannot be
            parsed.
    """
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


def parse_action_args(
    raw_args: str,
    spec: ToolSpec,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse ``raw_args`` against ``spec.parameters``.

    Accepts both ``name=value`` (LLM-friendly) and
    ``"name": value`` (JSON-style) syntax.  A trailing error
    string is returned on failure (the function never raises
    out of the agent loop).

    Args:
        raw_args: The raw text inside the ``Action: name(...)``
            parentheses.
        spec:     The :class:`ToolSpec` whose parameters are
            used to coerce each value.

    Returns:
        A ``(kwargs, error)`` tuple.  When ``error`` is ``None``,
        ``kwargs`` is the parsed-and-coerced dict.  Otherwise
        ``kwargs`` is empty and ``error`` is a human-readable
        message.
    """
    if not raw_args.strip():
        return {}, None

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
            # JSON-style value (e.g. ``[1,2,3]`` or ``{"k": "v"}``).
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
            parsed[key] = coerce_value(value, spec.parameters.get(key, "str"))
        except Exception as exc:  # noqa: BLE001
            return {}, f"failed to coerce {key!r}={value!r}: {exc}"

    # Validate that every required parameter is present.
    missing = [k for k in spec.parameters if k not in parsed]
    if missing:
        return {}, f"missing required arguments: {missing}"
    return parsed, None
