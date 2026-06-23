"""Attention mechanisms for Transformer models.

This module provides three attention variants commonly used in modern
large language models:

* :class:`MultiHeadAttention` -- standard multi-head self-attention
  (MHA).
* :class:`GroupedQueryAttention` -- Grouped-Query Attention (GQA) where
  the number of key/value heads is smaller than the number of query
  heads, sharing KV projections across query groups.
* :class:`MultiQueryAttention` -- Multi-Query Attention (MQA) where a
  single key/value head is shared across all query heads.

All variants support:

* Flash Attention via
  :func:`torch.nn.functional.scaled_dot_product_attention` when available.
* Rotary position embeddings applied to the query and key tensors.
* An optional KV cache for autoregressive generation.

References:
    Vaswani et al., "Attention Is All You Need" (2017).
    Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints" (2023).
    Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"
    (2019).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "apply_rotary_pos_emb",
    "repeat_kv",
]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        k: Key tensor of shape ``(batch, num_kv_heads, seq_len, head_dim)``.
        cos: Cosine table broadcastable to ``q``.
        sin: Sine table broadcastable to ``q``.

    Returns:
        A tuple ``(q_rotated, k_rotated)``.
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate(q) * sin)
    k_embed = (k * cos) + (rotate(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads.

    Args:
        hidden_states: Tensor of shape ``(batch, num_kv_heads, seq_len, head_dim)``.
        n_rep: Number of repetitions (``num_heads // num_kv_heads``).

    Returns:
        Tensor of shape ``(batch, num_kv_heads * n_rep, seq_len, head_dim)``.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class _AttentionBase(nn.Module):
    """Shared functionality for the attention variants."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim or hidden_size // num_heads
        self.num_kv_reps: int = num_heads // num_kv_heads
        self.scaling: float = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p: float = dropout

        # Query projection always produces ``num_heads`` heads.
        self.q_proj: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        # Key / value projections produce ``num_kv_heads`` heads.
        self.k_proj: nn.Linear = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    # ------------------------------------------------------------------
    def _shape(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """Reshape ``(batch, seq, dim)`` -> ``(batch, heads, seq, head_dim)``."""
        return x.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, heads, seq, head_dim)`` -> ``(batch, seq, dim)``."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    @staticmethod
    def _has_flash_attention() -> bool:
        """Return ``True`` if SDPA (Flash Attention) is available."""
        return hasattr(F, "scaled_dot_product_attention")


class MultiHeadAttention(_AttentionBase):
    """Standard multi-head self-attention (MHA).

    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (defaults to ``hidden_size // num_heads``).
        bias: Whether to use bias in the projections.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run multi-head attention.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask of shape
                ``(batch, 1, seq_len, seq_len)`` (or broadcastable).
            position_embeddings: Optional ``(cos, sin)`` tuple for RoPE.
            kv_cache: Optional ``(past_key, past_value)`` tuple.
            use_cache: Whether to return the updated KV cache.

        Returns:
            A tuple ``(output, new_kv_cache)``.
        """
        return _attention_forward(self, hidden_states, attention_mask, position_embeddings, kv_cache, use_cache)


class GroupedQueryAttention(_AttentionBase):
    """Grouped-Query Attention (GQA).

    With ``num_kv_heads < num_heads`` the key/value projections are
    shared across groups of query heads, reducing the KV cache size and
    memory bandwidth.

    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (must divide ``num_heads``).
        head_dim: Dimension per head.
        bias: Whether to use bias in the projections.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run grouped-query attention.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask.
            position_embeddings: Optional ``(cos, sin)`` for RoPE.
            kv_cache: Optional ``(past_key, past_value)`` tuple.
            use_cache: Whether to return the updated KV cache.

        Returns:
            A tuple ``(output, new_kv_cache)``.
        """
        return _attention_forward(self, hidden_states, attention_mask, position_embeddings, kv_cache, use_cache)


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query Attention (MQA).

    A special case of GQA with ``num_kv_heads = 1``: a single key/value
    head is shared across all query heads.

    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of query heads.
        head_dim: Dimension per head.
        bias: Whether to use bias in the projections.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            bias=bias,
            dropout=dropout,
        )


# ---------------------------------------------------------------------------
# Shared forward implementation
# ---------------------------------------------------------------------------
def _attention_forward(
    module: _AttentionBase,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Shared attention forward pass used by all variants.

    Args:
        module: The attention module (providing projections and config).
        hidden_states: ``(batch, seq_len, hidden_size)``.
        attention_mask: Optional additive mask.
        position_embeddings: Optional ``(cos, sin)`` for RoPE.
        kv_cache: Optional ``(past_key, past_value)``.
        use_cache: Whether to return the new KV cache.

    Returns:
        ``(output, new_kv_cache)``.
    """
    batch_size, seq_len, _ = hidden_states.shape

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = module._shape(query_states, seq_len, batch_size)
    key_states = module._shape(key_states, seq_len, batch_size)
    value_states = module._shape(value_states, seq_len, batch_size)

    # Apply rotary position embeddings if provided.
    if position_embeddings is not None:
        cos, sin = position_embeddings
        # Ensure cos/sin cover the current sequence length.
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Concatenate with the KV cache.
    if kv_cache is not None:
        past_key, past_value = kv_cache
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    new_kv_cache = (key_states, value_states) if use_cache else None

    # Repeat KV heads to match the number of query heads (GQA / MQA).
    key_states = repeat_kv(key_states, module.num_kv_reps)
    value_states = repeat_kv(value_states, module.num_kv_reps)

    # Compute attention.
    if module._has_flash_attention():
        # SDPA handles the causal mask internally when ``is_causal=True``;
        # here we rely on the explicit ``attention_mask`` for flexibility.
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=module.dropout_p if module.training else 0.0,
            is_causal=False,
        )
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * module.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)
        if module.dropout_p > 0.0 and module.training:
            attn_weights = F.dropout(attn_weights, p=module.dropout_p)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)
    attn_output = module.o_proj(attn_output)

    return attn_output, new_kv_cache
