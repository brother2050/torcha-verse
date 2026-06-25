"""Datasets for the v0.6.x training stack.

The :mod:`training.dataset` sub-package consolidates the four
concrete dataset classes that ship with TorchaVerse plus the
:class:`BaseDataset` abstract base.  All datasets accept an
optional :class:`BaseTokenizer` and produce a dictionary of
``input_ids`` / ``attention_mask`` / ``labels`` per item so
they work with the framework's :func:`collate_fn` out of the
box.

The v0.6.x refactor splits the previous single-file
``training/dataset.py`` (1063 lines) into six focused modules:

* :mod:`training.dataset._readers` -- format-agnostic file
  readers (JSONL / CSV / Parquet).
* :mod:`training.dataset._base` -- the :class:`BaseTokenizer`
  protocol, the :class:`BaseDataset` base class, and the
  :func:`collate_fn` helper.
* :mod:`training.dataset._text` -- :class:`TextDataset`
  (plain-text / JSONL / CSV / Parquet / block-mode).
* :mod:`training.dataset._chat` -- :class:`ChatDataset`
  (ShareGPT / OpenAI / column-based).
* :mod:`training.dataset._image_text` --
  :class:`ImageTextDataset` (image-caption pairs).
* :mod:`training.dataset._streaming` -- :class:`StreamingDataset`
  (offset-indexed random access to large files).

The public API is unchanged --
``from training.dataset import TextDataset, ChatDataset,
ImageTextDataset, StreamingDataset, BaseDataset,
collate_fn`` keeps working.
"""

from __future__ import annotations

from ._base import BaseDataset, BaseTokenizer, PathLike, collate_fn
from ._chat import ChatDataset
from ._image_text import ImageTextDataset
from ._streaming import StreamingDataset
from ._text import TextDataset

__all__ = [
    "BaseDataset",
    "BaseTokenizer",
    "PathLike",
    "collate_fn",
    "TextDataset",
    "ChatDataset",
    "ImageTextDataset",
    "StreamingDataset",
]
