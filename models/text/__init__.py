"""Text models for TorchaVerse.

This sub-package contains decoder-only Transformer language models,
attention mechanisms, embeddings, and Mixture-of-Experts variants.
"""

from __future__ import annotations

from .attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from .embeddings import PositionalEmbedding, TokenEmbedding
from .moe import (
    Expert,
    MoELayer,
    MoETransformerBlock,
    MoETransformerDecoder,
    Router,
)
from .transformer import TransformerBlock, TransformerDecoder

__all__ = [
    # attention
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "apply_rotary_pos_emb",
    "repeat_kv",
    # embeddings
    "TokenEmbedding",
    "PositionalEmbedding",
    # transformer
    "TransformerBlock",
    "TransformerDecoder",
    # moe
    "Expert",
    "Router",
    "MoELayer",
    "MoETransformerBlock",
    "MoETransformerDecoder",
]
