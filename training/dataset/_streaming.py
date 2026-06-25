"""Memory-efficient streaming dataset for the v0.6.x training stack.

The :class:`StreamingDataset` does not load the entire file
into memory.  Instead it scans the file once to record byte
offsets of each line and then seeks directly to the requested
line on access.  This allows random access to files that are
too large to fit in memory, which is the common case for
petabyte-scale text corpora.

Two flavours are supported:

* ``is_json=True`` -- each line is parsed as a JSON object and
  the configured ``text_field`` is extracted.
* ``is_json=False`` -- the raw line is used verbatim.

The byte-offset table is built once at construction time; it
sits in memory as a list of 64-bit integers (1 byte per line
on average), which is the right trade-off for the random-
access pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from infrastructure.logger import get_logger

from ._base import BaseTokenizer, PathLike, collate_fn

__all__ = ["StreamingDataset"]


class StreamingDataset(Dataset):
    """Memory-efficient streaming dataset for large line-delimited files.

    Unlike the in-memory datasets above, :class:`StreamingDataset`
    does not load the entire file into memory.  Instead it
    scans the file once to record byte offsets of each line
    and then seeks directly to the requested line on access.
    This allows random access to files that are too large to
    fit in memory.

    Args:
        file_path: Path to the line-delimited file (JSONL or
            text).
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        is_json: When ``True`` each line is parsed as JSON and
            the ``text_field`` is extracted; otherwise the raw
            line is used.
        text_field: Field name for JSON lines.
    """

    def __init__(
        self,
        file_path: PathLike,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        is_json: bool = True,
        text_field: str = "text",
    ) -> None:
        self.file_path: Path = Path(file_path).expanduser().resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        if tokenizer is None:
            # Fall back to the legacy ``TokenizerHub`` for
            # backward compatibility with v0.4.x code that
            # relied on its tiny character-level fallback.
            from training.tokenizer import TokenizerHub
            tokenizer = TokenizerHub().get_tokenizer(
                "text", vocab_size=256, max_length=max_length
            )
        self.tokenizer: BaseTokenizer = tokenizer
        self.max_length: int = max(1, int(max_length))
        self.is_json: bool = is_json
        self.text_field: str = text_field
        self.pad_token_id: int = getattr(self.tokenizer, "pad_token_id", 0)
        self._logger = get_logger(self.__class__.__name__)

        # Record the byte offset of each line.
        self._offsets: List[int] = []
        self._scan_file()
        self._logger.info(
            "Indexed %d lines in %s.", len(self._offsets), self.file_path
        )

    # ------------------------------------------------------------------
    def _scan_file(self) -> None:
        """Record the byte offset of the start of each line."""
        offset = 0
        with open(self.file_path, "rb") as handle:
            for line in handle:
                self._offsets.append(offset)
                offset += len(line)
        # Sentinel marking the end of the file.
        self._offsets.append(offset)

    def __len__(self) -> int:
        # The last entry is the EOF sentinel.
        return max(0, len(self._offsets) - 1)

    # ------------------------------------------------------------------
    def _read_line(self, index: int) -> str:
        """Read the ``index``-th line from the file via seeking.

        Args:
            index: Zero-based line index.

        Returns:
            The decoded line string (without trailing newline).
        """
        start = self._offsets[index]
        end = self._offsets[index + 1]
        with open(self.file_path, "rb") as handle:
            handle.seek(start)
            raw = handle.read(end - start)
        return raw.decode("utf-8").strip()

    def _encode(self, text: str) -> List[int]:
        """Tokenise ``text`` into a list of token ids."""
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=False,
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a tokenised example from the streamed file."""
        line = self._read_line(index)
        if not line:
            line = " "

        if self.is_json:
            try:
                obj = json.loads(line)
                text = obj.get(self.text_field, "")
            except json.JSONDecodeError:
                text = line
        else:
            text = line

        input_ids = self._encode(text)
        attention_mask = [
            1 if tid != self.pad_token_id else 0 for tid in input_ids
        ]
        labels = list(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def iter_examples(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over all examples sequentially (no random access).

        Yields:
            Tokenised example dictionaries.
        """
        for i in range(len(self)):
            yield self[i]

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of streaming examples."""
        return collate_fn(batch, pad_token_id=self.pad_token_id)
