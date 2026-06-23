"""Embedding layers for Transformer models.

This module provides token and positional embedding layers used by the
decoder-only language models.  Two positional strategies are supported:

* **Learned absolute** positional embeddings (:class:`PositionalEmbedding`).
* **Rotary** position embeddings applied to the query/key tensors inside
  the attention layers (handled by :class:`RotaryPositionEmbedding` in
  the ``components`` sub-package).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.components.rope import RotaryPositionEmbedding

__all__ = ["TokenEmbedding", "PositionalEmbedding"]


class TokenEmbedding(nn.Module):
    """Standard token embedding layer.

    Maps discrete token ids to dense vectors.  Embeddings are scaled by
    ``sqrt(hidden_size)`` following the original Transformer recipe.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the embedding vectors.
        padding_idx: Optional padding token id whose embedding is kept
            at zero.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}.")
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.padding_idx: Optional[int] = padding_idx
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings.

        Args:
            input_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Embedded tensor of shape ``(batch, seq_len, hidden_size)``
            scaled by ``sqrt(hidden_size)``.
        """
        embeds = self.embedding(input_ids)
        return embeds * math.sqrt(self.hidden_size)

    @property
    def weight(self) -> torch.Tensor:
        """The embedding weight matrix."""
        return self.embedding.weight


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embedding.

    Args:
        max_seq_len: Maximum supported sequence length.
        hidden_size: Embedding dimension.
    """

    def __init__(self, max_seq_len: int, hidden_size: int) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")
        self.max_seq_len: int = max_seq_len
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return positional embeddings for the first ``seq_len`` positions.

        Args:
            seq_len: Number of positions to return.
            device: Device for the position indices.

        Returns:
            Positional embeddings of shape ``(seq_len, hidden_size)``.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}."
            )
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)

    @property
    def weight(self) -> torch.Tensor:
        """The embedding weight matrix."""
        return self.embedding.weight
