"""Plain-text dataset for the v0.6.x training stack.

The :class:`TextDataset` is the workhorse dataset for
language-modelling on plain text.  The file format is
auto-detected from the extension:

* ``.jsonl`` -- each line is a JSON object with a ``"text"``
  key (or a configurable ``text_field``).
* ``.csv`` -- a CSV file with a ``text`` column (or
  ``text_field``).
* ``.parquet`` / ``.pq`` -- a Parquet table with a ``text``
  column.  Optional ``pyarrow`` / ``pandas`` dependency.
* ``.txt`` (or any other) -- each line is treated as one text
  example.

The :attr:`block_size` option concatenates the entire corpus
and splits it into fixed-size token blocks -- the standard
setup for causal-LM pre-training.

This module depends on :mod:`._base` for :class:`BaseDataset` /
:func:`collate_fn` and on :mod:`._readers` for the
format-agnostic row readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ._base import BaseDataset, BaseTokenizer, PathLike
from ._readers import read_csv_rows, read_parquet_rows

__all__ = ["TextDataset"]


class TextDataset(BaseDataset):
    """Plain-text dataset supporting JSONL, CSV, and raw-text formats.

    Args:
        file_path: Path to the data file.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        text_field: Name of the field/column containing the text.
        block_size: When set, the entire corpus is concatenated
            and split into fixed-size blocks (useful for language
            modelling).
    """

    def __init__(
        self,
        file_path: PathLike,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        text_field: str = "text",
        block_size: Optional[int] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.file_path: Path = Path(file_path).expanduser().resolve()
        self.text_field: str = text_field
        self.block_size: Optional[int] = block_size
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load examples from the configured file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".jsonl":
            self._load_jsonl()
        elif suffix == ".csv":
            self._load_csv()
        elif suffix in (".parquet", ".pq"):
            self._load_parquet()
        else:
            self._load_text()

        # Optionally split into fixed-size blocks for LM training.
        if self.block_size is not None and self.block_size > 0:
            self._build_blocks()

        self._logger.info(
            "Loaded %d examples from %s.", len(self._examples), self.file_path,
        )

    def _load_jsonl(self) -> None:
        """Load a JSONL file (one JSON object per line)."""
        with open(self.file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(self.text_field, "")
                if text:
                    self._examples.append(text)

    def _load_csv(self) -> None:
        """Load a CSV file with a text column."""
        rows = read_csv_rows(self.file_path)
        if rows and self.text_field not in rows[0]:
            col = next(iter(rows[0].keys()))
        else:
            col = self.text_field
        for row in rows:
            text = row.get(col, "")
            if text:
                self._examples.append(text)

    def _load_text(self) -> None:
        """Load a raw text file (one example per line)."""
        with open(self.file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    self._examples.append(text)

    def _load_parquet(self) -> None:
        """Load a Parquet file with a text column.

        Parquet support is opt-in: the module imports ``pyarrow``
        / ``pandas`` lazily and falls back to a clear error when
        neither is installed.  Operators that need a
        zero-dependency build can drop the optional dependencies
        from ``requirements.txt`` without breaking the rest of
        the dataset stack.
        """
        rows = read_parquet_rows(self.file_path)
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get(self.text_field)
            if text is None and row:
                # Fall back to the first column when the
                # configured field is missing -- mirrors the CSV
                # branch.
                first_key = next(iter(row.keys()), None)
                if first_key is not None:
                    text = row.get(first_key)
            if text:
                self._examples.append(text)

    def _build_blocks(self) -> None:
        """Concatenate all text and split into fixed-size token blocks."""
        all_ids: List[int] = []
        for text in self._examples:
            ids = self._encode(text, add_special_tokens=False)
            all_ids.extend(ids)

        block = self.block_size or self.max_length
        self._examples = []  # type: ignore[assignment]
        self._blocks: List[List[int]] = []
        for i in range(0, len(all_ids), block):
            chunk = all_ids[i : i + block]
            if len(chunk) == block:
                self._blocks.append(chunk)
        self._logger.info(
            "Built %d fixed-size blocks of length %d.",
            len(self._blocks), block,
        )

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a tokenised text example.

        When ``block_size`` is set, returns pre-tokenised
        blocks; otherwise tokenises the example on the fly.
        """
        if self.block_size is not None and hasattr(self, "_blocks"):
            input_ids = self._blocks[index]
            attention_mask = [1] * len(input_ids)
            labels = list(input_ids)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        text = self._examples[index]
        input_ids = self._encode(text)
        attention_mask = self._make_attention_mask(input_ids)
        labels = list(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
