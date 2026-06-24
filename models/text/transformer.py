"""Decoder-only Transformer language model.

This module implements a modern decoder-only Transformer (LLaMA-style)
with the following features:

* Pre-normalisation with RMSNorm (or LayerNorm).
* Grouped-Query Attention (GQA) with rotary position embeddings.
* SwiGLU feed-forward blocks.
* KV-cache support for efficient autoregressive generation.
* Top-k / top-p (nucleus) sampling.

The model inherits from :class:`BaseModel` and is therefore
registerable with the :class:`ModelRegistry`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.components.rope import RotaryPositionEmbedding
from models.components.rmsnorm import RMSNorm
from models.components.swiglu import SwiGLU
from .attention import GroupedQueryAttention
from .embeddings import TokenEmbedding

__all__ = ["TransformerBlock", "TransformerDecoder"]


def _get_norm(norm_type: str, hidden_size: int, eps: float = 1e-6) -> nn.Module:
    """Return a normalisation layer based on ``norm_type``."""
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}. Use 'rmsnorm' or 'layernorm'.")


def _get_activation(
    activation: str,
    hidden_size: int,
    intermediate_size: int,
) -> nn.Module:
    """Return an MLP block based on ``activation``."""
    if activation == "swiglu":
        return SwiGLU(hidden_size, intermediate_size, bias=False)
    if activation == "geglu":
        return _GeGLU(hidden_size, intermediate_size, bias=False)
    if activation == "mlp":
        return _MLP(hidden_size, intermediate_size)
    raise ValueError(f"Unknown activation: {activation!r}.")


class _GeGLU(nn.Module):
    """GELU-gated feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class _MLP(nn.Module):
    """Standard MLP feed-forward block (GELU)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """A single Transformer decoder layer.

    Pre-norm architecture::

        h = x + Attn(norm1(x))
        out = h + MLP(norm2(h))

    Args:
        hidden_size: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        intermediate_size: MLP intermediate dimension.
        max_seq_len: Maximum sequence length (for RoPE).
        rope_theta: RoPE base frequency.
        norm_type: ``"rmsnorm"`` or ``"layernorm"``.
        activation: ``"swiglu"``, ``"geglu"``, or ``"mlp"``.
        attention_dropout: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_layernorm: nn.Module = _get_norm(norm_type, hidden_size)
        self.self_attn: GroupedQueryAttention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
        )
        self.post_attention_layernorm: nn.Module = _get_norm(norm_type, hidden_size)
        self.mlp: nn.Module = _get_activation(activation, hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the transformer block.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask.
            position_embeddings: Optional ``(cos, sin)`` for RoPE.
            kv_cache: Optional ``(past_key, past_value)``.
            use_cache: Whether to return the updated KV cache.

        Returns:
            ``(output, new_kv_cache)``.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class TransformerDecoder(BaseModel):
    """Decoder-only Transformer language model.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        intermediate_size: MLP intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        norm_type: Normalisation type (``"rmsnorm"`` or ``"layernorm"``).
        activation: MLP activation (``"swiglu"``, ``"geglu"``, ``"mlp"``).
        tie_word_embeddings: Whether to tie the LM head to the token
            embedding weights.
        attention_dropout: Attention dropout probability.
        config: Optional configuration dictionary (used by the
            :class:`ModelRegistry`).  When provided, any missing
            argument is filled from it.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 11008,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Allow the registry to pass a config dict.
        if config is not None:
            vocab_size = config.get("vocab_size", vocab_size)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            num_kv_heads = config.get("num_kv_heads", num_kv_heads)
            intermediate_size = config.get("intermediate_size", intermediate_size)
            max_seq_len = config.get("max_seq_len", max_seq_len)
            rope_theta = config.get("rope_theta", rope_theta)
            norm_type = config.get("norm_type", norm_type)
            activation = config.get("activation", activation)
            tie_word_embeddings = config.get("tie_word_embeddings", tie_word_embeddings)
            attention_dropout = config.get("attention_dropout", attention_dropout)

        super().__init__(config=config)

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = hidden_size // num_heads
        self.max_seq_len: int = max_seq_len
        self.tie_word_embeddings: bool = tie_word_embeddings

        self.embed_tokens: TokenEmbedding = TokenEmbedding(vocab_size, hidden_size)
        self.rotary_emb: RotaryPositionEmbedding = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        self.layers: nn.ModuleList = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                norm_type=norm_type,
                activation=activation,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm: nn.Module = _get_norm(norm_type, hidden_size)
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        # Initialise weights.
        self.apply(self._init_weights)
        # Re-scale residual projections.
        for name, param in self.named_parameters():
            if name.endswith("w3.weight") or name.endswith("o_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise weights with a normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def _build_causal_mask(
        self,
        seq_len: int,
        past_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build a causal attention mask (additive).

        Args:
            seq_len: Current query length.
            past_len: Number of cached tokens.
            device: Target device.
            attention_mask: Optional padding mask of shape
                ``(batch, total_len)``.

        Returns:
            An additive mask of shape ``(batch, 1, seq_len, total_len)``
            or ``None`` when no masking is needed.
        """
        total_len = seq_len + past_len
        # Causal mask: query position i can attend to key positions <= i + past_len.
        causal = torch.triu(
            torch.full((seq_len, total_len), float("-inf"), device=device),
            diagonal=1 + past_len,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, total)

        if attention_mask is not None:
            # attention_mask: (batch, total_len) with 1 for valid, 0 for pad.
            pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * float("-inf")
            mask = mask + pad_mask
        return mask

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the forward pass and return the logits.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.  Either
                this or ``inputs_embeds`` must be provided.
            attention_mask: Optional padding mask of shape
                ``(batch, seq_len)`` (1 = valid, 0 = pad).
            kv_cache: Optional list of ``(key, value)`` tuples, one per
                layer, for incremental decoding.
            use_cache: Whether to populate and return the KV cache.
            inputs_embeds: Optional precomputed embeddings of shape
                ``(batch, seq_len, hidden_size)``.  When provided,
                ``input_ids`` is ignored.

        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            device = hidden_states.device
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            hidden_states = self.embed_tokens(input_ids)

        # Precompute RoPE cos/sin.
        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]
        total_len = seq_len + past_len
        cos, sin = self.rotary_emb.get_cos_sin(total_len, device)
        position_embeddings = (cos, sin)

        # Build the causal mask.
        attn_mask = self._build_causal_mask(seq_len, past_len, device, attention_mask)

        new_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if use_cache:
            self._last_kv_cache = new_cache
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token ids of shape ``(batch, seq_len)``.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (``1.0`` = no scaling).
            top_k: If ``> 0`` keep only the top-k tokens before sampling.
            top_p: Nucleus sampling threshold (``1.0`` = disabled).
            eos_token_id: Optional end-of-sequence token id that stops
                generation.

        Returns:
            Generated token ids of shape ``(batch, seq_len + max_tokens)``.
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids

        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        use_cache = True

        for step in range(max_tokens):
            if step == 0:
                # Prefill: process the whole prompt.
                logits = self.forward(
                    generated, kv_cache=kv_cache, use_cache=use_cache
                )
                next_logits = logits[:, -1, :]
            else:
                # Decode: process only the last token.
                last_token = generated[:, -1:].contiguous()
                logits = self.forward(
                    last_token, kv_cache=kv_cache, use_cache=use_cache
                )
                next_logits = logits[:, -1, :]

            # Apply temperature.
            if temperature > 0:
                next_logits = next_logits / temperature
                # Top-k filtering.
                if top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, top_k)
                    min_values = values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < min_values, float("-inf")
                    )
                # Top-p (nucleus) filtering.
                if top_p < 1.0:
                    next_logits = _top_p_filter(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding.
                next_token = torch.argmax(next_logits, dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Logits of shape ``(batch, vocab_size)``.
        top_p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold.
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right so the first token above the threshold is kept.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # Scatter back to the original indices.
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits
