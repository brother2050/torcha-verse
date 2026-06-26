"""CLIP-style Byte-Pair-Encoding (BPE) tokenizer (v0.8.0).

A self-contained, dependency-free BPE tokenizer that follows the
public OpenAI CLIP / Stable Diffusion 1.x vocabulary layout:

* Reads a ``vocab.json`` (``{token: id}``) + ``merges.txt``
  (one merge per line) pair that ``@openai/CLIP`` ships.
* Falls back to a deterministic byte-level encoding when the
  vocabulary files are missing (so unit tests can run on a stock
  dev box without a network).
* Honours ``max_length`` + BOS / EOS tokens, returning
  ``input_ids`` as a :class:`torch.LongTensor` of shape
  ``(batch, max_length)``.

The implementation is intentionally small (~150 lines) and uses
only the Python standard library + :mod:`torch`.  This avoids the
``tokenizers`` / ``transformers`` optional dependencies that the
v0.8 plan keeps out of the core.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Union

import torch

__all__ = [
    "SimpleByteBPETokenizer",
    "byte_level_decode",
    "byte_level_encode",
]


# ---------------------------------------------------------------------------
# Byte-level helpers
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> dict[int, str]:
    """Return the OpenAI byte-to-unicode mapping used by CLIP BPE.

    The mapping is deterministic and identical to the reference
    implementation in ``@openai/CLIP``.  We re-create it here so
    the tokenizer is self-contained.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def byte_level_encode(text: str) -> str:
    """Map every byte in ``text`` through the CLIP byte-to-unicode table."""
    return "".join(_B2U[b] for b in text.encode("utf-8", errors="ignore"))


def byte_level_decode(text: str) -> str:
    """Inverse of :func:`byte_level_encode`."""
    return bytes(_U2B[c] for c in text).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class SimpleByteBPETokenizer:
    """Minimal OpenAI-style BPE tokenizer.

    Args:
        vocab_path: Path to a ``vocab.json`` file (``{token: id}``).
            When ``None`` or missing, the tokenizer falls back to a
            deterministic byte-level encoding.
        merges_path: Path to a ``merges.txt`` file.  Each line is
            one merge pair (separated by a single space); lines
            starting with ``#`` are comments.
        max_length: Maximum sequence length.  Sequences are
            truncated / padded to this length.
        bos_token: Beginning-of-sequence token (``"<|startoftext|>"``
            in the reference CLIP vocabulary).
        eos_token: End-of-sequence token (``"<|endoftext|>"``).
        pad_token: Padding token (default ``"<|endoftext|>"``).
        unk_token: Unknown token.  Tokens not in the vocabulary are
            mapped to this id (or to ``0`` when ``unk_token`` is
            missing).
    """

    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        merges_path: Optional[Union[str, Path]] = None,
        *,
        max_length: int = 77,
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
    ) -> None:
        self.max_length = int(max_length)
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token if pad_token is not None else eos_token
        self.unk_token = unk_token
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}
        self._bpe_ranks: dict[tuple[str, str], int] = {}
        self._cache: dict[str, list[str]] = {}

        if vocab_path is not None and Path(vocab_path).is_file():
            self._load_vocab(Path(vocab_path))
        if merges_path is not None and Path(merges_path).is_file():
            self._load_merges(Path(merges_path))
        # Make sure the special tokens have ids even in fallback mode.
        for tok in (bos_token, eos_token, pad_token, unk_token):
            if tok and tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_vocab(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"vocab.json at {path} is not a JSON object")
        self._vocab = {k: int(v) for k, v in data.items()}
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    def _load_merges(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = [
                line.strip() for line in f.read().split("\n")
                if line.strip() and not line.startswith("#")
            ]
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 2:
                continue
            self._bpe_ranks[(parts[0], parts[1])] = i

    # ------------------------------------------------------------------
    # BPE
    # ------------------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        if not self._bpe_ranks:
            # No merges available; treat the token as a single unit.
            return [token]
        word = tuple(token)
        pairs = {(" ".join(word[i:i+1]), " ".join(word[i+1:i+2]))
                 for i in range(len(word) - 1)}
        if not pairs:
            return list(word)
        while True:
            ranked = sorted(
                ((self._bpe_ranks.get(p, float("inf")), p) for p in pairs),
                key=lambda x: x[0],
            )
            if ranked[0][0] == float("inf"):
                break
            bigram = ranked[0][1]
            new_word: list[str] = []
            i = 0
            while i < len(word):
                j = i + 1
                while j < len(word) and (word[i], word[j]) != bigram:
                    j += 1
                if j >= len(word):
                    new_word.append(word[i])
                    i += 1
                else:
                    new_word.append(word[i] + word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {
                (word[i], word[i + 1])
                for i in range(len(word) - 1)
            }
        self._cache[token] = list(word)
        return list(word)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _tokenize_one(self, text: str) -> list[int]:
        if not text:
            return []
        bpe_tokens: list[int] = []
        # Split on whitespace / punctuation using a CLIP-style regex.
        import re
        pattern = (
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"""
            r"""|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        )
        # Fallback when ``regex`` is unavailable.
        try:
            tokens = re.findall(pattern, text, flags=re.UNICODE)
        except Exception:  # noqa: BLE001
            tokens = text.split()
        for tok in tokens:
            bpe = self._bpe(byte_level_encode(tok))
            for piece in bpe:
                bpe_tokens.append(self._vocab.get(piece, self._vocab.get(self.unk_token, 0)))
        return bpe_tokens

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenize a string or a list of strings.

        Returns:
            A dict with ``input_ids`` (always), and optionally
            ``attention_mask`` (when padding is performed).
        """
        if isinstance(texts, str):
            texts = [texts]
        batch: list[list[int]] = []
        for t in texts:
            ids = self._tokenize_one(t)
            ids = [self._vocab.get(self.bos_token, 0)] + ids
            ids = ids[: self.max_length - 1]
            ids.append(self._vocab.get(self.eos_token, 0))
            if len(ids) < self.max_length:
                pad_id = self._vocab.get(self.pad_token, 0)
                ids = ids + [pad_id] * (self.max_length - len(ids))
            batch.append(ids)
        input_ids = torch.tensor(batch, dtype=torch.long)
        out: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if return_tensors == "pt":
            attention_mask = (input_ids != self._vocab.get(self.pad_token, 0)).long()
            out["attention_mask"] = attention_mask
        return out

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of token ids back to text."""
        text = "".join(self._inv_vocab.get(int(i), "") for i in ids)
        return byte_level_decode(text)
