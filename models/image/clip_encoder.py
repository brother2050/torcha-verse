"""CLIP text encoder.

This module implements a CLIP-style text encoder that converts token
ids into dense text embeddings.  It is used as the text-conditioning
branch of text-to-image diffusion models (e.g. Stable Diffusion).

The encoder uses causal self-attention (as in the original CLIP text
transformer) and produces both a sequence of token embeddings and a
pooled representation (taken from the ``[CLS]`` token or via mean
pooling).

Reference:
    Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision" (2021).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

__all__ = ["CLIPTextEncoder", "CLIPEncoderLayer"]


def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Build an additive causal mask of shape ``(seq_len, seq_len)``."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class CLIPEncoderLayer(nn.Module):
    """A single CLIP text-encoder layer.

    Uses a pre-LN architecture with causal multi-head self-attention
    and a quick-GeLU MLP.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of the MLP intermediate size to ``hidden_size``.
        layernorm_eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.scale: float = self.head_dim ** -0.5

        self.self_attn: nn.Module = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.layer_norm1: nn.LayerNorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.layer_norm2: nn.LayerNorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)

        intermediate_size = int(hidden_size * mlp_ratio)
        self.fc1: nn.Linear = nn.Linear(hidden_size, intermediate_size)
        self.fc2: nn.Linear = nn.Linear(intermediate_size, hidden_size)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Quick-GeLU MLP."""
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the CLIP encoder layer.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            causal_attention_mask: Additive causal mask of shape
                ``(1, 1, seq_len, seq_len)``.
            attention_mask: Optional padding mask ``(batch, seq_len)``.

        Returns:
            Output hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_attention_mask,
            key_padding_mask=attention_mask,
            need_weights=False,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self._mlp(hidden_states)
        return hidden_states


class CLIPTextEncoder(BaseModel):
    """CLIP text encoder.

    Converts token ids into text embeddings and a pooled representation.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimension.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (for positional embeddings).
        mlp_ratio: MLP intermediate-size ratio.
        layernorm_eps: LayerNorm epsilon.
        pool_mode: How to produce the pooled output: ``"cls"`` (use the
            first token) or ``"mean"`` (mean over tokens).
        config: Optional configuration dictionary (used by the
            :class:`ModelRegistry`).  When provided, any missing
            argument is filled from it.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 77,
        mlp_ratio: float = 4.0,
        layernorm_eps: float = 1e-5,
        pool_mode: str = "cls",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            vocab_size = config.get("vocab_size", vocab_size)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            max_seq_len = config.get("max_seq_len", max_seq_len)
            mlp_ratio = config.get("mlp_ratio", mlp_ratio)
            layernorm_eps = config.get("layernorm_eps", layernorm_eps)
            pool_mode = config.get("pool_mode", pool_mode)

        super().__init__(config=config)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        if pool_mode not in ("cls", "mean"):
            raise ValueError(f"pool_mode must be 'cls' or 'mean', got {pool_mode!r}.")

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.max_seq_len: int = max_seq_len
        self.pool_mode: str = pool_mode

        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding: nn.Embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers: nn.ModuleList = nn.ModuleList([
            CLIPEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                layernorm_eps=layernorm_eps,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialise weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def build_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the additive causal attention mask."""
        return _build_causal_mask(seq_len, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token ids into text embeddings.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask ``(batch, seq_len)``
                with ``1`` for valid tokens and ``0`` for padding.  When
                provided, padded positions are zeroed in the output.
            position_ids: Optional explicit position ids.
            output_hidden_states: When ``True`` also returns the list of
                per-layer hidden states.

        Returns:
            A tuple ``(last_hidden_state, pooled_output)``.  When
            ``output_hidden_states`` is ``True`` a third element (the
            list of hidden states) is appended.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        causal_mask = self.build_attention_mask(seq_len, device)

        # Convert padding mask to key_padding_mask format (True = ignore).
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        all_hidden_states: list = []
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(
                hidden_states,
                causal_attention_mask=causal_mask,
                attention_mask=key_padding_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)

        # Zero out padded positions.
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        # Pooled output.
        if self.pool_mode == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, pooled, all_hidden_states
        return hidden_states, pooled

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text embeddings from token ids.

        For a text encoder, "generation" simply means encoding the
        input tokens and returning the last hidden states (the sequence
        of text embeddings used to condition a diffusion model).

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask.

        Returns:
            Last hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        self.eval()
        hidden_states, _ = self.forward(input_ids, attention_mask=attention_mask)
        return hidden_states
