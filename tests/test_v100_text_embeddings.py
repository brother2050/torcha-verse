"""Tests for token and positional embedding layers
(models/text/embeddings.py).

Both ``TokenEmbedding`` and ``PositionalEmbedding`` were 100% untested before
v0.10.0.  The three tests below cover the scaling-by-sqrt(hidden_size) recipe
of :class:`TokenEmbedding`, the ``padding_idx`` zero-row invariant, and the
``seq_len`` validation of :class:`PositionalEmbedding`.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.text.embeddings import PositionalEmbedding, TokenEmbedding


# ---------------------------------------------------------------------------
# 1. TokenEmbedding scales by sqrt(hidden_size)
# ---------------------------------------------------------------------------
class TestTokenEmbeddingScale:
    """Forward output must equal ``weight[ids] * sqrt(hidden_size)``."""

    def test_token_embedding_scale_by_sqrt(self):
        """Output shape and value match the manual scaling recipe."""
        vocab_size = 100
        hidden_size = 64
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
        emb.eval()  # disable dropout / init randomness side-effects
        torch.manual_seed(0)
        ids = torch.randint(0, vocab_size, (2, 10))
        out = emb(ids)
        # Shape must be (batch, seq, hidden).
        assert out.shape == (2, 10, hidden_size)
        # Manual computation: look up weights, scale by sqrt(hidden_size).
        manual = emb.weight[ids] * math.sqrt(hidden_size)
        assert torch.allclose(out, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. TokenEmbedding.padding_idx keeps the row at zero
# ---------------------------------------------------------------------------
class TestTokenEmbeddingPadding:
    """The padding row stays exactly zero both at init and after a forward."""

    def test_token_embedding_padding_idx_zero(self):
        """weight[padding_idx] is exactly zero and the forward output for
        any padding id is also zero."""
        emb = TokenEmbedding(vocab_size=100, hidden_size=64, padding_idx=0)
        emb.eval()
        # Row 0 is exactly zero.
        assert torch.all(emb.weight[0] == 0)
        # All other rows are not all zero (sanity check — the layer was
        # actually initialised with random normal samples).
        assert not torch.all(emb.weight[1] == 0)
        # Forward a batch that includes id 0; the corresponding output
        # positions are zero.
        ids = torch.tensor([[0, 5, 0], [3, 0, 7]])
        out = emb(ids)
        out_for_padding = out[ids == 0]
        assert torch.all(out_for_padding == 0)
        # And the non-padding positions are NOT all zero.
        out_for_real = out[ids != 0]
        assert not torch.all(out_for_real == 0)


# ---------------------------------------------------------------------------
# 3. PositionalEmbedding validates seq_len against max_seq_len
# ---------------------------------------------------------------------------
class TestPositionalEmbeddingSeqLen:
    """seq_len must be <= max_seq_len or a ValueError is raised."""

    def test_positional_embedding_seq_len_exceeds_max(self):
        """seq_len=11 (with max_seq_len=10) raises ValueError; seq_len=10 ok."""
        emb = PositionalEmbedding(max_seq_len=10, hidden_size=8)
        emb.eval()
        # seq_len > max_seq_len must raise ValueError with a clear message.
        with pytest.raises(ValueError) as excinfo:
            emb(seq_len=11)
        # The error message should mention the offending length and limit.
        msg = str(excinfo.value)
        assert "11" in msg
        assert "10" in msg
        # seq_len == max_seq_len must succeed.
        out = emb(seq_len=10)
        # The implementation returns (seq_len, hidden_size) without a leading
        # batch dimension, which is what ``TransformerDecoder`` expects.
        assert out.shape == (10, 8)
