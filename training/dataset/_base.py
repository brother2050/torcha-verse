"""The base classes for the v0.6.x :mod:`training.dataset` sub-package.

This module contains the *shared* abstractions every concrete
dataset class reuses:

* :class:`BaseTokenizer` -- the duck-typed tokenizer protocol
  (HuggingFace / tiktoken / custom are all accepted).
* :class:`_DefaultTokenizer` -- a tiny character-level fallback
  used when no tokenizer is supplied, so the dataset classes
  stay import-safe even without a real tokenizer.
* :class:`BaseDataset` -- the :class:`torch.utils.data.Dataset`
  base class that provides tokenisation helpers, padding,
  attention-mask generation, and a default collation method.
* :func:`collate_fn` -- the module-level batch-padding helper
  used by ``DataLoader``s.

The four concrete datasets (:class:`TextDataset`,
:class:`ChatDataset`, :class:`ImageTextDataset`,
:class:`StreamingDataset`) live in their own modules so each
stays under 200 lines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

import torch
from torch.utils.data import Dataset

from infrastructure.logger import get_logger

__all__ = [
    "BaseTokenizer",
    "PathLike",
    "BaseDataset",
    "collate_fn",
]


#: Path-like type alias used throughout the dataset sub-package.
PathLike = Union[str, "Path"]


@runtime_checkable
class BaseTokenizer(Protocol):
    """Minimal tokenizer protocol used by TorchaVerse datasets.

    Implementations must expose ``encode``, ``decode``,
    ``pad_token_id``, ``bos_token_id`` and ``eos_token_id``.  The
    protocol is duck-typed so that any tokenizer (HuggingFace,
    tiktoken, custom) is accepted without a hard dependency.
    """

    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str, **kwargs: Any) -> List[int]: ...
    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...


class _DefaultTokenizer:
    """A tiny character-level fallback tokenizer.

    Used when no external tokenizer is supplied.  It maps every
    Unicode code point to its ordinal and reserves ids 0/1/2 for
    PAD/BOS/EOS.  This keeps the dataset classes import-safe even
    when the optional tokenizer dependency is not installed.
    """

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def encode(self, text: str, **_: Any) -> List[int]:
        return [self.bos_token_id] + [ord(c) + 3 for c in text] + [self.eos_token_id]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        return "".join(chr(int(i) - 3) for i in ids if int(i) >= 3)


class BaseDataset(Dataset):
    """Unified base class for all TorchaVerse datasets.

    Subclasses implement :meth:`_load` to populate
    ``self._examples`` and :meth:`__getitem__` to return a single
    example.  The base class provides tokenisation, padding,
    attention-mask generation, and a default collation function.

    Args:
        tokenizer: A :class:`TextTokenizer` (or any object with
            ``encode``/``decode``).  When ``None`` a default
            tokenizer is used.
        max_length: Maximum sequence length.  Longer sequences
            are truncated.
        pad_token_id: Token id used for padding.  Defaults to
            the tokenizer's ``pad_token_id``.
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        pad_token_id: Optional[int] = None,
    ) -> None:
        # Fall back to a tiny built-in tokenizer so that
        # ``BaseDataset`` and its subclasses can be instantiated
        # without depending on the legacy ``TokenizerHub`` (which
        # has been removed).  Callers that have a real tokenizer
        # should pass it explicitly.
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

        Subclasses should override this to produce the
        appropriate dictionary of tensors.
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
    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a list of examples into a batched dictionary.

        Pads each tensor field to the longest sequence in the
        batch and stacks them.  Fields ending in ``_mask`` are
        padded with zeros; all other fields are padded with
        :attr:`pad_token_id`.

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

        Subclasses override this to implement format-specific
        loading.
        """
        raise NotImplementedError("Subclasses must implement _load.")


def collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Pad and stack a batch of example dictionaries.

    Each example is a mapping from field name to a 1-D
    :class:`torch.Tensor` (or list).  Fields are padded to the
    maximum length in the batch:

    * Fields whose name ends in ``_mask`` are padded with ``0``.
    * Fields named ``labels`` are padded with ``-100`` (ignored
      by loss).
    * All other fields are padded with ``pad_token_id``.

    Args:
        batch: A list of example dictionaries.
        pad_token_id: The padding token id for non-mask,
            non-label fields.

    Returns:
        A dictionary mapping field names to batched tensors of
        shape ``(batch_size, max_seq_len)``.
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
