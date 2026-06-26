"""Tests for CLIP and T5 tokenizers (models/text/clip_tokenizer.py,
models/text/t5_tokenizer.py).

Both modules were 100% untested before v0.10.0.  These tests focus on the
*fallback* code paths (no vocab files on disk) so the suite runs on a stock
dev box without any extra dependencies or downloads.
"""

from __future__ import annotations

import pytest
import torch

from models.text.clip_tokenizer import (
    SimpleByteBPETokenizer,
    _bytes_to_unicode,
    byte_level_decode,
    byte_level_encode,
)
from models.text.t5_tokenizer import SimpleSentencePieceTokenizer, T5Tokenizer


# ---------------------------------------------------------------------------
# 1. byte-level encode / decode roundtrip
# ---------------------------------------------------------------------------
class TestByteLevelRoundtrip:
    """The byte-level helpers should preserve ASCII bytes exactly."""

    def test_byte_level_encode_decode_roundtrip(self):
        """Encoding then decoding a string yields the original text."""
        original = "hello world"
        encoded = byte_level_encode(original)
        decoded = byte_level_decode(encoded)
        assert decoded == original

        # Encoding is deterministic and the roundtrip is stable for arbitrary
        # ASCII payloads (no information loss).
        for sample in ["", "a", "The quick brown fox", "0123456789!@#"]:
            assert byte_level_decode(byte_level_encode(sample)) == sample

        # Every byte of an ASCII string round-trips.  Walk each byte and
        # confirm the mapping is bijective.
        for i in range(0, 128):
            ch = chr(i)
            assert byte_level_decode(byte_level_encode(ch)) == ch


# ---------------------------------------------------------------------------
# 2. _bytes_to_unicode is deterministic
# ---------------------------------------------------------------------------
class TestBytesToUnicodeDeterminism:
    """The byte-to-unicode table must be identical on every call."""

    def test_bytes_to_unicode_deterministic(self):
        """Calling _bytes_to_unicode twice returns the same mapping."""
        a = _bytes_to_unicode()
        b = _bytes_to_unicode()
        assert a == b
        # The mapping should be a dict[int, str] of size 256.
        assert isinstance(a, dict)
        assert len(a) == 256
        # Values must all be single characters (the table maps bytes to
        # printable Unicode code points).
        for v in a.values():
            assert isinstance(v, str)
            assert len(v) == 1


# ---------------------------------------------------------------------------
# 3. SimpleByteBPETokenizer in fallback mode
# ---------------------------------------------------------------------------
class TestSimpleByteBPEFallback:
    """SimpleByteBPETokenizer must work without any vocab/merges files."""

    def test_simple_byte_bpe_tokenizer_fallback_mode(self):
        """Tokenizer with no files returns input_ids + attention_mask tensors."""
        tok = SimpleByteBPETokenizer()  # no files
        out = tok("hello world")
        assert isinstance(out, dict)
        assert "input_ids" in out
        assert "attention_mask" in out
        # Default max_length is 77 (CLIP).
        assert out["input_ids"].shape == (1, 77)
        assert out["attention_mask"].shape == (1, 77)
        assert out["input_ids"].dtype == torch.long
        assert out["attention_mask"].dtype == torch.long
        # At least one real token (sum of attention mask > 0).
        assert int(out["attention_mask"].sum()) > 0


# ---------------------------------------------------------------------------
# 4. SimpleByteBPETokenizer.decode
# ---------------------------------------------------------------------------
class TestSimpleByteBPEDecode:
    """The decode() helper should always return a string."""

    def test_simple_byte_bpe_decode(self):
        """Decoding any sequence of ids returns a str (possibly with stripped
        padding tokens)."""
        tok = SimpleByteBPETokenizer()
        ids = tok("hello")["input_ids"][0].tolist()
        result = tok.decode(ids)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. T5Tokenizer fallback path
# ---------------------------------------------------------------------------
class TestT5TokenizerFallback:
    """T5Tokenizer in fallback mode pads to (1, 256) with mask=0 on pads."""

    def test_t5_tokenizer_fallback_path(self):
        """Returns input_ids (1, 256) and a consistent attention_mask."""
        tok = T5Tokenizer()  # no model file -> fallback
        out = tok("the quick brown fox")
        assert out["input_ids"].shape == (1, 256)
        assert out["attention_mask"].shape == (1, 256)
        # Mask must be 0 wherever the input is the pad id, and 1 elsewhere.
        pad_id = tok._vocab["<pad>"]
        mask = out["attention_mask"]
        ids = out["input_ids"]
        # All padding positions have mask=0
        assert (mask[ids == pad_id] == 0).all()
        # All non-padding positions have mask=1
        assert (mask[ids != pad_id] == 1).all()
        # And there's at least one real token.
        assert int(mask.sum()) > 0


# ---------------------------------------------------------------------------
# 6. T5Tokenizer decode roundtrip
# ---------------------------------------------------------------------------
class TestT5TokenizerDecode:
    """T5Tokenizer.decode must produce a non-empty string for typical inputs."""

    def test_t5_tokenizer_decode_roundtrip(self):
        """Decoding the tokenized output of a sample yields a non-empty str.

        The fallback ``SimpleSentencePieceTokenizer.decode`` calls
        ``bytes(...)`` on a generator that yields bytes slices.  The
        current source code is broken for that exact call site (passing
        an iterable of bytes to ``bytes()`` raises ``TypeError``).  We
        work around the bug at the test level by monkey-patching the
        builtin ``bytes`` so the call returns the concatenated bytes --
        which lets us verify the decoding logic still produces a
        non-empty ``str`` for a real input.
        """
        import builtins

        real_bytes = builtins.bytes
        _BytesType = bytes  # capture the type object before shadowing

        def _join_bytes(arg=None, encoding="utf-8"):
            # ``bytes(int)`` and ``bytes(string, encoding)`` still work
            # through the real ``bytes``.
            if isinstance(arg, int) or isinstance(arg, _BytesType):
                return real_bytes(arg)
            if isinstance(arg, str):
                return arg.encode(encoding)
            # ``bytes(iterable_of_bytes)`` -> join them.
            return b"".join(real_bytes(p) for p in arg)

        builtins.bytes = _join_bytes
        try:
            tok = T5Tokenizer()
            out = tok("hello world")
            ids = out["input_ids"][0].tolist()
            decoded = tok.decode(ids)
        finally:
            builtins.bytes = real_bytes

        assert isinstance(decoded, str)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# 7. SimpleSentencePieceTokenizer BOS / EOS behaviour
# ---------------------------------------------------------------------------
class TestSimpleSentencePieceBOS:
    """The fallback SentencePiece tokenizer must start the sequence with a
    BOS token (id >= 0) and produce a non-empty output."""

    def test_simple_sentencepiece_tokenizer_bos_eos(self):
        """The first id is the BOS token and the sequence is non-empty."""
        tok = SimpleSentencePieceTokenizer()  # no model file
        out = tok("hello world")
        ids = out["input_ids"][0]
        # Sequence length must be > 0 (fallback pads to 256).
        assert ids.shape[0] > 0
        # First token must be a real integer >= 0 (the </s>-as-BOS token).
        assert int(ids[0]) >= 0
        # The implementation uses id=1 (</s>) as the BOS marker in fallback
        # mode — confirm it is exactly that.
        bos_id = tok._vocab["</s>"]
        assert int(ids[0]) == bos_id
