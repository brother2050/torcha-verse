"""Dataset abstractions for training TorchaVerse models.

This module provides a family of :class:`torch.utils.data.Dataset`
subclasses that cover the common data formats used for training and
fine-tuning language and multimodal models:

* :class:`BaseDataset` -- shared functionality (tokenisation, padding,
  attention-mask generation, collation).
* :class:`TextDataset` -- plain-text corpora in JSONL, CSV, or raw-text
  format.
* :class:`ChatDataset` -- conversational data in ShareGPT or OpenAI
  message format.
* :class:`ImageTextDataset` -- image-caption pairs.
* :class:`StreamingDataset` -- memory-efficient iteration over large
  line-delimited files.

A module-level :func:`collate_fn` helper pads variable-length sequences
into a batched tensor and produces the corresponding attention mask.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Union, runtime_checkable

import torch
from torch.utils.data import Dataset

# TokenizerHub removed - use ModuleBus or inject tokenizer directly.
# A minimal ``BaseTokenizer`` protocol is provided here so that the
# dataset classes remain type-safe without depending on the old
# ``TokenizerHub`` abstraction.
from infrastructure.logger import get_logger

__all__ = [
    "BaseTokenizer",
    "BaseDataset",
    "TextDataset",
    "ChatDataset",
    "ImageTextDataset",
    "StreamingDataset",
    "collate_fn",
]


@runtime_checkable
class BaseTokenizer(Protocol):
    """Minimal tokenizer protocol used by TorchaVerse datasets.

    Implementations must expose ``encode``, ``decode``, ``pad_token_id``,
    ``bos_token_id`` and ``eos_token_id``.  The protocol is duck-typed
    so that any tokenizer (HuggingFace, tiktoken, custom) is accepted
    without a hard dependency.
    """

    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str, **kwargs: Any) -> List[int]: ...
    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...


#: Path-like type alias.
PathLike = Union[str, Path]


class _DefaultTokenizer:
    """A tiny character-level fallback tokenizer.

    Used when no external tokenizer is supplied.  It maps every Unicode
    code point to its ordinal and reserves ids 0/1/2 for PAD/BOS/EOS.
    This keeps the dataset classes import-safe even when the optional
    tokenizer dependency is not installed.
    """

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def encode(self, text: str, **_: Any) -> List[int]:
        return [self.bos_token_id] + [ord(c) + 3 for c in text] + [self.eos_token_id]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        return "".join(chr(int(i) - 3) for i in ids if int(i) >= 3)


# ---------------------------------------------------------------------------
# BaseDataset
# ---------------------------------------------------------------------------
class BaseDataset(Dataset):
    """Unified base class for all TorchaVerse datasets.

    Subclasses implement :meth:`_load` to populate ``self._examples``
    and :meth:`__getitem__` to return a single example.  The base class
    provides tokenisation, padding, attention-mask generation, and a
    default collation function.

    Args:
        tokenizer: A :class:`TextTokenizer` (or any object with
            ``encode``/``decode``).  When ``None`` a default tokenizer
            is obtained from the :class:`TokenizerHub`.
        max_length: Maximum sequence length.  Longer sequences are
            truncated.
        pad_token_id: Token id used for padding.  Defaults to the
            tokenizer's ``pad_token_id``.
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        pad_token_id: Optional[int] = None,
    ) -> None:
        # Fall back to a tiny built-in tokenizer so that ``BaseDataset``
        # and its subclasses can be instantiated without depending on the
        # legacy ``TokenizerHub`` (which has been removed).  Callers that
        # have a real tokenizer should pass it explicitly.
        self.tokenizer: BaseTokenizer = tokenizer or _DefaultTokenizer()
        self.max_length: int = max(1, int(max_length))
        self.pad_token_id: int = (
            pad_token_id
            if pad_token_id is not None
            else getattr(self.tokenizer, "pad_token_id", 0)
        )
        self.bos_token_id: int = getattr(self.tokenizer, "bos_token_id", 1)
        self.eos_token_id: int = getattr(self.tokenizer, "eos_token_id", 2)
        self._examples: List[Any] = []
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenised example.

        Subclasses should override this to produce the appropriate
        dictionary of tensors.
        """
        raise NotImplementedError("Subclasses must implement __getitem__.")

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------
    def _encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Tokenise ``text`` into a list of token ids.

        Args:
            text: The input string.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Optional override for truncation length.

        Returns:
            A list of integer token ids.
        """
        effective_max = max_length or self.max_length
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=effective_max,
            return_tensors=False,
        )
        # ``encode`` may return a list of lists (batch mode).
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _make_attention_mask(self, input_ids: Sequence[int]) -> List[int]:
        """Build a binary attention mask (1 for real tokens, 0 for pad).

        Args:
            input_ids: The token id sequence.

        Returns:
            A list of 0/1 integers of the same length.
        """
        return [1 if tid != self.pad_token_id else 0 for tid in input_ids]

    def _pad_sequence(
        self, input_ids: List[int], max_len: Optional[int] = None
    ) -> List[int]:
        """Right-pad ``input_ids`` to ``max_len`` with the pad token.

        Args:
            input_ids: The token id sequence.
            max_len: Target length.  Defaults to :attr:`max_length`.

        Returns:
            The padded sequence.
        """
        target = max_len or self.max_length
        if len(input_ids) >= target:
            return input_ids[:target]
        return input_ids + [self.pad_token_id] * (target - len(input_ids))

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a list of examples into a batched dictionary.

        Pads each tensor field to the longest sequence in the batch and
        stacks them.  Fields ending in ``_mask`` are padded with zeros;
        all other fields are padded with :attr:`pad_token_id`.

        Args:
            batch: A list of example dictionaries.

        Returns:
            A dictionary of batched tensors.
        """
        return collate_fn(batch, pad_token_id=self.pad_token_id)

    # ------------------------------------------------------------------
    # Loading hook
    # ------------------------------------------------------------------
    def _load(self, *args: Any, **kwargs: Any) -> None:
        """Populate ``self._examples`` from a data source.

        Subclasses override this to implement format-specific loading.
        """
        raise NotImplementedError("Subclasses must implement _load.")


# ---------------------------------------------------------------------------
# Module-level collate function
# ---------------------------------------------------------------------------
def collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Pad and stack a batch of example dictionaries.

    Each example is a mapping from field name to a 1-D :class:`torch.Tensor`
    (or list).  Fields are padded to the maximum length in the batch:

    * Fields whose name ends in ``_mask`` are padded with ``0``.
    * Fields named ``labels`` are padded with ``-100`` (ignored by loss).
    * All other fields are padded with ``pad_token_id``.

    Args:
        batch: A list of example dictionaries.
        pad_token_id: The padding token id for non-mask, non-label fields.

    Returns:
        A dictionary mapping field names to batched tensors of shape
        ``(batch_size, max_seq_len)``.
    """
    if not batch:
        return {}

    # Collect all field names present in the batch.
    fields: List[str] = []
    seen: set = set()
    for example in batch:
        for key in example:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    output: Dict[str, torch.Tensor] = {}
    for field in fields:
        # Determine the pad value for this field.
        if field.endswith("_mask"):
            pad_value = 0
        elif field == "labels":
            pad_value = -100
        else:
            pad_value = pad_token_id

        sequences: List[List[int]] = []
        for example in batch:
            value = example.get(field)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                sequences.append(value.tolist())
            elif isinstance(value, list):
                sequences.append(list(value))
            else:
                sequences.append([value])

        if not sequences:
            continue

        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        output[field] = torch.tensor(padded, dtype=torch.long)

    return output


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------
class TextDataset(BaseDataset):
    """Plain-text dataset supporting JSONL, CSV, and raw-text formats.

    The file format is auto-detected from the extension:

    * ``.jsonl`` -- each line is a JSON object with a ``"text"`` key
      (or a configurable ``text_field``).
    * ``.csv`` -- a CSV file with a ``text`` column (or ``text_field``).
    * ``.txt`` (or any other) -- each line is treated as one text
      example.

    Args:
        file_path: Path to the data file.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        text_field: Name of the field/column containing the text.
        block_size: When set, the entire corpus is concatenated and
            split into fixed-size blocks (useful for language modelling).
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
            "Loaded %d examples from %s.", len(self._examples), self.file_path
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
        with open(self.file_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and self.text_field not in reader.fieldnames:
                # Fall back to the first column.
                col = reader.fieldnames[0]
            else:
                col = self.text_field
            for row in reader:
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

        Parquet support is opt-in: the module imports ``pyarrow`` /
        ``pandas`` lazily and falls back to a clear error when
        neither is installed.  Operators that need a zero-dependency
        build can drop the optional dependencies from
        ``requirements.txt`` without breaking the rest of the
        dataset stack.
        """
        rows = _read_parquet_rows(self.file_path)
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get(self.text_field)
            if text is None and row:
                # Fall back to the first column when the configured
                # field is missing -- mirrors the CSV branch.
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
        self._logger.info("Built %d fixed-size blocks of length %d.", len(self._blocks), block)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a tokenised text example.

        When ``block_size`` is set, returns pre-tokenised blocks;
        otherwise tokenises the example on the fly.
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


# ---------------------------------------------------------------------------
# ChatDataset
# ---------------------------------------------------------------------------
class ChatDataset(BaseDataset):
    """Conversational dataset supporting ShareGPT and OpenAI formats.

    Two formats are recognised:

    * **ShareGPT** -- each line is a JSON object with a ``"conversations"``
      list, where each entry has ``"from"`` (``"human"``/``"gpt"``) and
      ``"value"``.
    * **OpenAI** -- each line is a JSON object with a ``"messages"`` list,
      where each entry has ``"role"`` and ``"content"``.

    Args:
        file_path: Path to the JSONL file.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        format: Conversation format (``"sharegpt"`` or ``"openai"``).
            When ``None`` the format is auto-detected.
        system_prompt: Optional system prompt prepended to every
            conversation.
    """

    #: Maps ShareGPT "from" values to role names.
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
        """Load conversations from the configured file.

        Recognised extensions:

        * ``.jsonl`` -- one JSON object per line.
        * ``.csv`` / ``.parquet`` -- each row is a record with either
          a ``"conversations"`` / ``"messages"`` column (JSON-encoded
          string) or with the column names suffixed by ``_role`` /
          ``_content`` for the role/content pairs.
        """
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
            raw_objects = _read_csv_rows(self.file_path)
        elif suffix in (".parquet", ".pq"):
            raw_objects = _read_parquet_rows(self.file_path)
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
            "Loaded %d conversations from %s.", len(self._examples), self.file_path
        )

    def _extract_messages(self, obj: Dict[str, Any]) -> List[Dict[str, str]]:
        """Normalise a conversation object into a list of messages.

        Args:
            obj: The parsed JSON object.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dictionaries.

        Notes:
            CSV / Parquet rows often serialise a list of messages
            as a single JSON string under the ``"conversations"`` /
            ``"messages"`` column.  When that happens we transparently
            decode the string back into a list -- callers therefore
            do not have to pre-parse the JSON themselves.
        """
        messages: List[Dict[str, str]] = []

        # Prepend the system prompt if configured.
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # The row-shape used in CSV/Parquet files may have
        # "conversations" / "messages" as JSON-encoded strings; if
        # so, decode them transparently.
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
            # Column-based format (common in CSV exports): e.g.
            # ``turn_0_role=user``, ``turn_0_content=hi``,
            # ``turn_1_role=assistant``, ``turn_1_content=hello``.
            # The columns are gathered by suffix and ordered by the
            # numeric prefix.
            turn_cols: Dict[int, Dict[str, str]] = {}
            for key, value in obj.items():
                if not isinstance(key, str):
                    continue
                # The column naming convention is
                # ``<prefix>_<idx>_<field>`` where ``prefix`` is
                # any non-numeric name (e.g. ``turn``) and
                # ``field`` is one of ``role`` / ``content``.
                # We split on underscores and look for the first
                # numeric chunk to find the turn index.
                parts = key.split("_")
                if len(parts) < 3:
                    continue
                # The field is always the last component.
                field = parts[-1]
                if field not in ("role", "content"):
                    continue
                # The turn index is the LAST numeric chunk
                # before the field.  Search from the right so that
                # a multi-word prefix (e.g. ``my_turn_0``) still
                # works.
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
                # Honour the ShareGPT role map (so callers can use
                # the same column names as their JSONL export).
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

        The full conversation is tokenised and used as both the input
        and the labels (standard causal-LM training).
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


# ---------------------------------------------------------------------------
# ImageTextDataset
# ---------------------------------------------------------------------------
class ImageTextDataset(BaseDataset):
    """Image-caption (image-text pair) dataset.

    Loads image paths and their associated captions from a JSONL file.
    Each line should be a JSON object with ``"image"`` (path) and
    ``"caption"`` (or ``"text"``) keys.

    Images are loaded lazily on access via :mod:`PIL` when available;
    otherwise the image path is returned and the caller is responsible
    for decoding.

    Args:
        file_path: Path to the JSONL metadata file.
        image_dir: Base directory for resolving relative image paths.
        tokenizer: Text tokenizer.
        max_length: Maximum caption length.
        caption_field: Name of the caption field in the JSON.
        image_field: Name of the image-path field in the JSON.
        load_images: When ``True`` load the image pixels on access
            (requires Pillow).
    """

    def __init__(
        self,
        file_path: PathLike,
        image_dir: Optional[PathLike] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        caption_field: str = "caption",
        image_field: str = "image",
        load_images: bool = False,
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.file_path: Path = Path(file_path).expanduser().resolve()
        self.image_dir: Optional[Path] = (
            Path(image_dir).expanduser().resolve() if image_dir else None
        )
        self.caption_field: str = caption_field
        self.image_field: str = image_field
        self.load_images: bool = load_images
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load image-caption pairs from the configured file.

        Recognised extensions:

        * ``.jsonl`` -- one JSON object per line.
        * ``.csv`` / ``.parquet`` -- one row per pair, with the
          configured ``image_field`` / ``caption_field`` columns.
        """
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
            raw_objects = _read_csv_rows(self.file_path)
        elif suffix in (".parquet", ".pq"):
            raw_objects = _read_parquet_rows(self.file_path)
        else:
            raise ValueError(
                f"ImageTextDataset does not support .{suffix} files; "
                "use .jsonl, .csv or .parquet."
            )

        for obj in raw_objects:
            image_path = obj.get(self.image_field, "")
            caption = obj.get(self.caption_field, obj.get("text", ""))
            if image_path and caption:
                self._examples.append(
                    {"image": image_path, "caption": caption}
                )

        self._logger.info(
            "Loaded %d image-caption pairs from %s.",
            len(self._examples),
            self.file_path,
        )

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve an image path relative to ``image_dir`` when needed.

        Args:
            image_path: The image path from the metadata.

        Returns:
            The resolved absolute :class:`~pathlib.Path`.
        """
        p = Path(image_path)
        if p.is_absolute() or self.image_dir is None:
            return p
        return (self.image_dir / image_path).resolve()

    def _load_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load an image as a tensor (requires Pillow).

        Args:
            image_path: Path to the image file.

        Returns:
            A ``torch.Tensor`` of shape ``(channels, height, width)``
            or ``None`` if Pillow is unavailable.
        """
        try:
            from PIL import Image
        except ImportError:
            return None

        img = Image.open(image_path).convert("RGB")
        import numpy as np

        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return arr

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an image-caption example.

        Returns a dictionary with tokenised ``input_ids``,
        ``attention_mask``, ``labels``, and the image ``path`` (and
        ``image`` tensor when ``load_images`` is ``True``).
        """
        pair = self._examples[index]
        caption = pair["caption"]
        input_ids = self._encode(caption)
        attention_mask = self._make_attention_mask(input_ids)
        labels = list(input_ids)

        result: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "path": pair["image"],
        }

        if self.load_images:
            image_path = self._resolve_image_path(pair["image"])
            image = self._load_image(image_path)
            if image is not None:
                result["image"] = image

        return result


# ---------------------------------------------------------------------------
# StreamingDataset
# ---------------------------------------------------------------------------
class StreamingDataset(Dataset):
    """Memory-efficient streaming dataset for large line-delimited files.

    Unlike the in-memory datasets above, :class:`StreamingDataset` does
    not load the entire file into memory.  Instead it scans the file
    once to record byte offsets of each line and then seeks directly to
    the requested line on access.  This allows random access to files
    that are too large to fit in memory.

    Args:
        file_path: Path to the line-delimited file (JSONL or text).
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        is_json: When ``True`` each line is parsed as JSON and the
            ``text_field`` is extracted; otherwise the raw line is used.
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

        self.tokenizer: BaseTokenizer = tokenizer or TokenizerHub().get_tokenizer(
            "text", vocab_size=256, max_length=max_length
        )
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
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]
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


# ---------------------------------------------------------------------------
# File-format helpers (shared by every dataset subclass)
# ---------------------------------------------------------------------------
def _read_csv_rows(file_path: PathLike) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of ``{column: value}`` dicts.

    Values are returned as plain strings (no automatic type
    coercion) so the dataset layer can decide what to do with
    them.  Empty cells become the empty string, never ``None``,
    to keep downstream ``row.get("text", "")`` lookups robust.
    """
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_parquet_rows(file_path: PathLike) -> List[Dict[str, Any]]:
    """Read a Parquet file and return a list of ``{column: value}`` dicts.

    Parquet support is opt-in: this helper tries ``pyarrow``
    first, then ``pandas``, and finally raises a clear error
    when neither optional dependency is available.  The dataset
    layer is therefore usable in zero-dependency environments
    *as long as* the operator avoids Parquet files.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    # pyarrow is the canonical, lightweight reader.
    try:
        import pyarrow.parquet as _pq  # type: ignore[import-not-found]

        table = _pq.read_table(p)
        columns = table.column_names
        out: List[Dict[str, Any]] = []
        for i in range(table.num_rows):
            row: Dict[str, Any] = {}
            for col in columns:
                value = table.column(col)[i].as_py()
                row[str(col)] = "" if value is None else value
            out.append(row)
        return out
    except ImportError:
        pass
    # pandas is the second-choice reader -- many production
    # environments already have it installed for data analysis.
    try:
        import pandas as _pd  # type: ignore[import-not-found]

        df = _pd.read_parquet(p)
        # ``df.to_dict("records")`` returns a list of plain dicts
        # with Python native types -- exactly what the dataset
        # layer expects.
        records: List[Dict[str, Any]] = df.to_dict("records")
        return [
            {str(k): ("" if v is None else v) for k, v in row.items()}
            for row in records
        ]
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires either `pyarrow` or `pandas`. "
            "Install one of them (e.g. `pip install pyarrow`) and retry."
        ) from exc
