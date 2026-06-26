"""Tests for ``models/text/attention.py``.

This module hosts three attention variants (:class:`MultiHeadAttention`,
:class:`GroupedQueryAttention`, :class:`MultiQueryAttention`) plus two
pure-function helpers (:func:`apply_rotary_pos_emb`, :func:`repeat_kv`).
All of them were 100% untested before v0.10.0.  The five tests below
focus on:

* rotary-position-embedding correctness,
* the KV-head repetition helper (``n_rep > 1`` and ``n_rep == 1``),
* :class:`MultiHeadAttention` output shapes with and without a mask,
* the :class:`GroupedQueryAttention` KV-cache path (regression guard).
"""

from __future__ import annotations

import math

import pytest
import torch

from models.text.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# 1. apply_rotary_pos_emb correctness
# ---------------------------------------------------------------------------
class TestApplyRotaryPosEmb:
    """``apply_rotary_pos_emb`` must match the manual half-rotation recipe."""

    def test_apply_rotary_pos_emb_correctness(self):
        """Manual rotation equals apply_rotary_pos_emb(q, k, cos, sin)."""
        torch.manual_seed(0)
        B, H, T, D = 2, 4, 7, 8  # head_dim must be even for half-rotation
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)  # k is required by the signature
        # cos / sin tables, shape (1, T, D) — broadcastable to (B, H, T, D).
        cos = torch.randn(1, T, D)
        sin = torch.randn(1, T, D)

        # Manual half-rotation that mirrors the implementation:
        #   rotate(x) = concat(-x[..., half:], x[..., :half])
        #   x' = x * cos + rotate(x) * sin
        def manual_rotate(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_manual = q * cos + manual_rotate(q) * sin
        k_manual = k * cos + manual_rotate(k) * sin

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_out, q_manual, atol=1e-6)
        assert torch.allclose(k_out, k_manual, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. repeat_kv with n_rep > 1
# ---------------------------------------------------------------------------
class TestRepeatKVWithRep:
    """``repeat_kv`` must expand (B, H_kv, T, D) -> (B, H_kv * n_rep, T, D)
    with ``n_rep`` identical copies of each head."""

    def test_repeat_kv_with_rep_gt_one(self):
        """Output has H_kv * n_rep heads, each group of n_rep heads equal."""
        torch.manual_seed(0)
        B, H_kv, T, D = 3, 2, 5, 4
        n_rep = 4
        x = torch.randn(B, H_kv, T, D)
        out = repeat_kv(x, n_rep=n_rep)
        # Shape is (B, H_kv * n_rep, T, D).
        assert out.shape == (B, H_kv * n_rep, T, D)
        # For each source head, the n_rep copies must be identical.
        for h in range(H_kv):
            group = out[:, h * n_rep : (h + 1) * n_rep]  # (B, n_rep, T, D)
            for i in range(1, n_rep):
                assert torch.allclose(group[:, 0], group[:, i], atol=1e-6)
            # And the group must equal the original head.
            assert torch.allclose(group[:, 0], x[:, h], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. repeat_kv with n_rep == 1 is the identity
# ---------------------------------------------------------------------------
class TestRepeatKVIdentity:
    """``repeat_kv(x, 1)`` returns ``x`` itself."""

    def test_repeat_kv_with_rep_one_is_identity(self):
        """When n_rep == 1 the function must return the input tensor."""
        x = torch.randn(2, 2, 5, 4)
        out = repeat_kv(x, n_rep=1)
        # Output shape is unchanged.
        assert out.shape == x.shape
        # Output is element-wise equal to the input.
        assert torch.allclose(out, x, atol=1e-6)
        # And (as an identity optimisation) is the same Python object.
        assert out is x


# ---------------------------------------------------------------------------
# 4. MultiHeadAttention forward shape
# ---------------------------------------------------------------------------
class TestMultiHeadAttentionForward:
    """``MultiHeadAttention(hidden_size, num_heads)`` preserves the
    (batch, seq_len, hidden_size) shape with and without an attention mask."""

    def test_multi_head_attention_forward_shape(self):
        """Output shape is (batch, seq_len, hidden_size) in both cases."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=64, num_heads=4)
        attn.eval()
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == (2, 10, 64)

        # With an attention mask shaped (batch, seq_q, seq_k) that masks
        # out the second half of the keys.  SDPA requires the mask to be
        # bool or float (not long).  The attention shape is
        # (batch, heads, seq_q, seq_k) so a (1, 1, 10, 10) mask
        # broadcasts cleanly.
        attn.eval()
        mask = torch.zeros(1, 1, 10, 10, dtype=torch.float)
        mask[..., 5:] = float("-inf")
        out_masked, _ = attn(x, attention_mask=mask)
        assert out_masked.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# 5. GroupedQueryAttention KV-cache
# ---------------------------------------------------------------------------
class TestGroupedQueryAttentionKVCache:
    """Two forward passes that share the KV cache must concatenate the
    sequence dimension and return only the new tokens' outputs."""

    def test_grouped_query_attention_kv_cache(self):
        """Pass 1 caches the K/V for the first 5 tokens; pass 2 reuses
        them to attend over 5 + 3 = 8 positions while emitting 3 new
        outputs."""
        torch.manual_seed(0)
        attn = GroupedQueryAttention(hidden_size=64, num_heads=4, num_kv_heads=2)
        attn.eval()

        # Pass 1: process 5 tokens; ask the module to return the cache.
        x1 = torch.randn(2, 5, 64)
        out1, kv_cache = attn(x1, use_cache=True)
        assert out1.shape == (2, 5, 64)
        assert kv_cache is not None
        past_k, past_v = kv_cache
        # The cache stores one K/V per token (T = 5) and per kv-head.
        assert past_k.shape == (2, 2, 5, 16)  # (B, H_kv, T, head_dim)
        assert past_v.shape == (2, 2, 5, 16)

        # Pass 2: process 3 new tokens, supplying the cache from pass 1.
        x2 = torch.randn(2, 3, 64)
        out2, new_kv = attn(x2, kv_cache=(past_k, past_v), use_cache=True)
        # Output covers only the 3 new tokens.
        assert out2.shape == (2, 3, 64)
        # The updated cache spans the full 5 + 3 = 8 positions.
        assert new_kv is not None
        new_k, new_v = new_kv
        assert new_k.shape == (2, 2, 8, 16)
        assert new_v.shape == (2, 2, 8, 16)
