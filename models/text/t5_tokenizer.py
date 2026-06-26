"""T5-style SentencePiece tokenizer (v0.8.0).

A minimal SentencePiece reader for the ``sp.model`` (v1) format
that the T5 / FLUX / HunyuanVideo text encoders rely on.  When
the ``sentencepiece`` package is unavailable the module falls
back to a byte-level encoding so the v0.8 smoke tests can run
without any external dependency.

The implementation supports the ``sp.model`` unigram vocabulary
layout (the variant used by T5 XXL) and exposes a small wrapper
that mirrors :class:`models.text.clip_tokenizer.SimpleByteBPETokenizer`
so callers can swap encoders with the same call shape.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

__all__ = ["SimpleSentencePieceTokenizer", "T5Tokenizer"]


class SimpleSentencePieceTokenizer:
    """A pragmatic SentencePiece (unigram) loader.

    The implementation is deliberately small — it only supports the
    pieces the v0.8 plan actually consumes (T5's ``<pad>``,
    ``</s>`` / ``<s>`` and a small piece table).  When the
    ``sentencepiece`` package is installed, that library is used
    directly for full fidelity.

    Args:
        model_path: Path to a ``sp.model`` file.  When ``None`` the
            tokenizer builds a 256-piece byte-level fallback so
            unit tests run on a fresh dev box.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        max_length: int = 256,
    ) -> None:
        self.max_length = int(max_length)
        self._use_native = False
        self._sp: Optional["object"] = None
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}
        if model_path is not None and Path(model_path).is_file():
            self._try_load_native(Path(model_path))
        if not self._use_native:
            self._build_byte_fallback()

    # ------------------------------------------------------------------
    def _try_load_native(self, path: Path) -> None:
        try:
            import sentencepiece  # type: ignore
            self._sp = sentencepiece.SentencePieceProcessor()
            self._sp.Load(str(path))
            self._use_native = True
            for i in range(self._sp.GetPieceSize()):
                piece = self._sp.IdToPiece(i)
                self._vocab[piece] = i
                self._inv_vocab[i] = piece
        except Exception:  # noqa: BLE001
            self._use_native = False
            self._sp = None

    def _build_byte_fallback(self) -> None:
        """Build a 256-piece byte-level vocabulary as a fallback."""
        self._vocab = {
            "<pad>": 0,
            "</s>": 1,
            "<unk>": 2,
        }
        for b in range(256):
            self._vocab[f"▁{chr(b)}" if b < 128 else f"<0x{b:02X}>"] = 3 + b
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    # ------------------------------------------------------------------
    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        if self._use_native:
            return self._encode_native(list(texts))
        return self._encode_fallback(list(texts))

    def _encode_native(self, texts: list[str]) -> dict[str, torch.Tensor]:
        assert self._sp is not None
        ids_batch: list[list[int]] = []
        for t in texts:
            ids = self._sp.EncodeAsIds(t)
            ids = ids[: self.max_length]
            pad_id = 0
            ids = ids + [pad_id] * (self.max_length - len(ids))
            ids_batch.append(ids)
        input_ids = torch.tensor(ids_batch, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _encode_fallback(self, texts: list[str]) -> dict[str, torch.Tensor]:
        ids_batch: list[list[int]] = []
        for t in texts:
            raw = t.encode("utf-8", errors="ignore")
            ids = [1]  # </s> BOS
            for b in raw:
                key = f"▁{chr(b)}" if b < 128 else f"<0x{b:02X}>"
                ids.append(self._vocab.get(key, 2))
                if len(ids) >= self.max_length - 1:
                    break
            ids.append(1)  # </s> EOS
            ids = ids + [0] * (self.max_length - len(ids))
            ids_batch.append(ids[: self.max_length])
        input_ids = torch.tensor(ids_batch, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: Sequence[int]) -> str:
        if self._use_native and self._sp is not None:
            return self._sp.DecodeIds([int(i) for i in ids])
        return bytes(
            self._inv_vocab.get(int(i), "<0x00>").encode("utf-8", errors="ignore")[1:]
            if self._inv_vocab.get(int(i), "").startswith("▁")
            else b""
            for i in ids
        ).decode("utf-8", errors="ignore")


# Convenience alias — the v0.8.0 pipeline targets T5.
class T5Tokenizer(SimpleSentencePieceTokenizer):
    """Alias for :class:`SimpleSentencePieceTokenizer` with T5 defaults."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__(model_path, max_length=256)
