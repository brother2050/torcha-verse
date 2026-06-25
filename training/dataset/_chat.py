"""Chat-style (conversational) dataset for the v0.6.x training stack.

The :class:`ChatDataset` is the workhorse dataset for
fine-tuning language models on multi-turn conversations.  It
recognises three input formats:

* **ShareGPT** -- each line is a JSON object with a
  ``"conversations"`` list, where each entry has ``"from"``
  (``"human"``/``"gpt"``) and ``"value"``.
* **OpenAI** -- each line is a JSON object with a ``"messages"``
  list, where each entry has ``"role"`` and ``"content"``.
* **Column-based** -- common in CSV exports: ``turn_0_role``,
  ``turn_0_content``, ``turn_1_role``, ``turn_1_content``,
  ...  The columns are gathered by suffix and ordered by the
  numeric prefix.

The file extension is auto-detected:

* ``.jsonl`` -- one JSON object per line.
* ``.csv`` -- a CSV file with the role/content columns.
* ``.parquet`` / ``.pq`` -- a Parquet table with the same
  schema.

This module depends on :mod:`._base` for :class:`BaseDataset`
and on :mod:`._readers` for the format-agnostic row readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ._base import BaseDataset, BaseTokenizer, PathLike
from ._readers import read_csv_rows, read_parquet_rows

__all__ = ["ChatDataset"]


class ChatDataset(BaseDataset):
    """Conversational dataset supporting ShareGPT and OpenAI formats.

    Args:
        file_path: Path to the JSONL/CSV/Parquet file.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        format: Conversation format (``"sharegpt"`` or
            ``"openai"``).  When ``None`` the format is
            auto-detected from the column names.
        system_prompt: Optional system prompt prepended to
            every conversation.
    """

    #: Mapping from ShareGPT ``"from"`` codes to canonical roles.
    _SHAREGPT_ROLE_MAP: Dict[str, str] = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    def __init__(
        self,
        file_path: PathLike,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        format: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.file_path: Path = Path(file_path).expanduser().resolve()
        self.format: Optional[str] = format.lower() if format else None
        self.system_prompt: Optional[str] = system_prompt
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load conversations from the configured file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".jsonl":
            raw_objects: List[Dict[str, Any]] = []
            with open(self.file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_objects.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif suffix == ".csv":
            raw_objects = read_csv_rows(self.file_path)
        elif suffix in (".parquet", ".pq"):
            raw_objects = read_parquet_rows(self.file_path)
        else:
            raise ValueError(
                f"ChatDataset does not support .{suffix} files; "
                "use .jsonl, .csv or .parquet."
            )

        for obj in raw_objects:
            messages = self._extract_messages(obj)
            if messages:
                self._examples.append(messages)

        self._logger.info(
            "Loaded %d conversations from %s.",
            len(self._examples), self.file_path,
        )

    def _extract_messages(self, obj: Dict[str, Any]) -> List[Dict[str, str]]:
        """Normalise a conversation object into a list of messages.

        Args:
            obj: The parsed JSON object.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dictionaries.

        Notes:
            CSV / Parquet rows often serialise a list of messages
            as a single JSON string under the ``"conversations"``
            / ``"messages"`` column.  When that happens we
            transparently decode the string back into a list --
            callers therefore do not have to pre-parse the JSON
            themselves.
        """
        messages: List[Dict[str, str]] = []

        # Prepend the system prompt if configured.
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # The row-shape used in CSV/Parquet files may have
        # "conversations" / "messages" as JSON-encoded strings;
        # if so, decode them transparently.
        if "conversations" in obj and isinstance(obj["conversations"], str):
            try:
                obj = dict(obj)
                obj["conversations"] = json.loads(obj["conversations"])
            except json.JSONDecodeError:
                return []
        if "messages" in obj and isinstance(obj["messages"], str):
            try:
                obj = dict(obj)
                obj["messages"] = json.loads(obj["messages"])
            except json.JSONDecodeError:
                return []

        if "conversations" in obj:
            # ShareGPT format.
            for turn in obj["conversations"]:
                role = self._SHAREGPT_ROLE_MAP.get(
                    turn.get("from", "").lower(), "user"
                )
                content = turn.get("value", "")
                messages.append({"role": role, "content": content})
        elif "messages" in obj:
            # OpenAI format.
            for turn in obj["messages"]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                messages.append({"role": role, "content": content})
        else:
            # Column-based format (common in CSV exports):
            # ``turn_0_role=user``, ``turn_0_content=hi``, ...
            # The columns are gathered by suffix and ordered by
            # the numeric prefix.
            turn_cols: Dict[int, Dict[str, str]] = {}
            for key, value in obj.items():
                if not isinstance(key, str):
                    continue
                # The column naming convention is
                # ``<prefix>_<idx>_<field>`` where ``prefix`` is
                # any non-numeric name (e.g. ``turn``) and
                # ``field`` is one of ``role`` / ``content``.
                # We split on underscores and look for the
                # first numeric chunk to find the turn index.
                parts = key.split("_")
                if len(parts) < 3:
                    continue
                # The field is always the last component.
                field = parts[-1]
                if field not in ("role", "content"):
                    continue
                # The turn index is the LAST numeric chunk
                # before the field.  Search from the right so
                # that a multi-word prefix (e.g. ``my_turn_0``)
                # still works.
                idx_str = ""
                for piece in reversed(parts[:-1]):
                    if piece.isdigit():
                        idx_str = piece
                        break
                if not idx_str:
                    continue
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                turn_cols.setdefault(idx, {})[field] = (
                    "" if value is None else str(value)
                )
            for idx in sorted(turn_cols):
                row = turn_cols[idx]
                if not row.get("content"):
                    continue
                role = row.get("role", "user")
                # Honour the ShareGPT role map (so callers can
                # use the same column names as their JSONL
                # export).
                role = self._SHAREGPT_ROLE_MAP.get(role.lower(), role)
                messages.append({"role": role, "content": row["content"]})
            if not messages:
                return []

        return messages

    # ------------------------------------------------------------------
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Render a conversation into a single prompt string.

        Args:
            messages: The conversation messages.

        Returns:
            The formatted prompt string.
        """
        parts: List[str] = []
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}] {msg['content']}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a tokenised conversation.

        The full conversation is tokenised and used as both the
        input and the labels (standard causal-LM training).
        """
        messages = self._examples[index]
        prompt = self._format_prompt(messages)
        input_ids = self._encode(prompt)
        attention_mask = self._make_attention_mask(input_ids)
        labels = list(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
