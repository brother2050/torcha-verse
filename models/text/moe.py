"""Mixture-of-Experts (MoE) Transformer language model.

This module implements a sparse Mixture-of-Experts Transformer where
the feed-forward block of (optionally some) layers is replaced by a
mixture of expert MLPs with a top-k gating router.

Key components:

* :class:`Expert` -- a single MLP expert.
* :class:`Router` -- top-k gating router with load-balancing loss.
* :class:`MoELayer` -- the MoE feed-forward layer.
* :class:`MoETransformerBlock` -- a transformer block using MoE.
* :class:`MoETransformerDecoder` -- the full MoE model.

The model's :meth:`forward` returns ``(logits, aux_loss)`` where
``aux_loss`` is the load-balancing auxiliary loss that encourages
uniform expert utilisation.

References:
    Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer" (2017).
    Fedus et al., "Switch Transformers: Scaling to Trillion Parameter
    Models with Simple and Efficient Sparsity" (2021).
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
from .transformer import _get_norm, _top_p_filter

__all__ = [
    "Expert",
    "Router",
    "MoELayer",
    "MoETransformerBlock",
    "MoETransformerDecoder",
]


class Expert(nn.Module):
    """A single expert (an MLP feed-forward block).

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: MLP intermediate dimension.
        activation: ``"swiglu"`` or ``"mlp"``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if activation == "swiglu":
            self.mlp: nn.Module = SwiGLU(hidden_size, intermediate_size, bias=False)
        elif activation == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown activation: {activation!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the expert MLP.

        Args:
            x: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.mlp(x)


class Router(nn.Module):
    """Top-k gating router for Mixture-of-Experts.

    Computes a gating logits vector for each token and selects the
    top-k experts.  A load-balancing auxiliary loss is computed to
    encourage uniform expert utilisation.

    Args:
        hidden_size: Input dimension.
        num_experts: Number of experts.
        top_k: Number of experts selected per token.
        router_jitter: Optional noise added to the logits during training
            to encourage exploration.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts}).")
        self.hidden_size: int = hidden_size
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.router_jitter: float = router_jitter
        self.gate: nn.Linear = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and the load-balancing loss.

        Args:
            x: Input of shape ``(batch, seq_len, hidden_size)`` or
                ``(num_tokens, hidden_size)``.

        Returns:
            A tuple ``(dispatch_mask, routing_weights, aux_loss)``:

            * ``dispatch_mask``: ``(num_tokens, num_experts)`` float mask
              with ``1`` for selected experts.
            * ``routing_weights``: ``(num_tokens, num_experts)`` softmax
              weights (zero for non-selected experts).
            * ``aux_loss``: Scalar load-balancing loss.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        logits = self.gate(x_flat)
        if self.training and self.router_jitter > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(
                -self.router_jitter, self.router_jitter
            )

        routing_weights = F.softmax(logits, dim=-1)

        # Top-k selection.
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Normalise the selected weights.
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Build a sparse (num_tokens, num_experts) weight matrix.
        sparse_weights = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        sparse_weights.scatter_(1, top_k_indices, top_k_weights)

        # Binary dispatch mask.
        dispatch_mask = (sparse_weights > 0).to(x.dtype)

        # Load-balancing loss (Switch Transformer style).
        aux_loss = self._load_balancing_loss(routing_weights, dispatch_mask)

        # Restore the batch dimension for the mask/weights.
        if len(orig_shape) > 2:
            batch, seq_len = orig_shape[0], orig_shape[1]
            dispatch_mask = dispatch_mask.view(batch, seq_len, self.num_experts)
            sparse_weights = sparse_weights.view(batch, seq_len, self.num_experts)

        return dispatch_mask, sparse_weights, aux_loss

    def _load_balancing_loss(
        self,
        routing_weights: torch.Tensor,
        dispatch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the load-balancing auxiliary loss.

        ``L = num_experts * sum_i (f_i * P_i)``

        where ``f_i`` is the fraction of tokens dispatched to expert ``i``
        and ``P_i`` is the average routing probability for expert ``i``.

        Args:
            routing_weights: ``(num_tokens, num_experts)`` softmax weights.
            dispatch_mask: ``(num_tokens, num_experts)`` binary mask.

        Returns:
            Scalar auxiliary loss.
        """
        num_tokens = routing_weights.shape[0]
        # Fraction of tokens assigned to each expert.
        tokens_per_expert = dispatch_mask.sum(dim=0)
        f = tokens_per_expert / max(num_tokens, 1)
        # Average routing probability per expert.
        p = routing_weights.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * p)
        return aux_loss


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Per-expert intermediate dimension.
        num_experts: Number of experts.
        top_k: Number of experts activated per token.
        activation: Expert activation (``"swiglu"`` or ``"mlp"``).
        router_jitter: Router noise during training.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "swiglu",
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.router: Router = Router(hidden_size, num_experts, top_k, router_jitter)
        self.experts: nn.ModuleList = nn.ModuleList([
            Expert(hidden_size, intermediate_size, activation)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the MoE layer.

        Args:
            x: Input of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            A tuple ``(output, aux_loss)``.
        """
        batch, seq_len, hidden = x.shape
        dispatch_mask, routing_weights, aux_loss = self.router(x)
        # dispatch_mask / routing_weights: (batch, seq, num_experts)

        x_flat = x.reshape(-1, hidden)  # (num_tokens, hidden)
        mask_flat = dispatch_mask.reshape(-1, self.num_experts)  # (num_tokens, num_experts)
        weights_flat = routing_weights.reshape(-1, self.num_experts)

        num_tokens = x_flat.shape[0]
        output = torch.zeros_like(x_flat)

        # Dispatch tokens to their selected experts.
        for expert_idx in range(self.num_experts):
            token_indices = torch.nonzero(mask_flat[:, expert_idx], as_tuple=False).squeeze(-1)
            if token_indices.numel() == 0:
                continue
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            w = weights_flat[token_indices, expert_idx].unsqueeze(-1)
            output[token_indices] += expert_output * w

        output = output.view(batch, seq_len, hidden)
        return output, aux_loss


class MoETransformerBlock(nn.Module):
    """A Transformer block with a Mixture-of-Experts feed-forward layer.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        intermediate_size: Per-expert intermediate dimension.
        num_experts: Number of experts.
        top_k: Experts per token.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        norm_type: Normalisation type.
        activation: Expert activation.
        use_moe: When ``False`` the block uses a dense SwiGLU instead.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        use_moe: bool = True,
        router_jitter: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_moe: bool = use_moe
        self.input_layernorm: nn.Module = _get_norm(norm_type, hidden_size)
        self.self_attn: GroupedQueryAttention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
        )
        self.post_attention_layernorm: nn.Module = _get_norm(norm_type, hidden_size)
        if use_moe:
            self.mlp: nn.Module = MoELayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                router_jitter=router_jitter,
            )
        else:
            from .transformer import _get_activation
            self.mlp = _get_activation(activation, hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the MoE transformer block.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional additive mask.
            position_embeddings: Optional ``(cos, sin)`` for RoPE.
            kv_cache: Optional KV cache.
            use_cache: Whether to return the updated KV cache.

        Returns:
            ``(output, aux_loss, new_kv_cache)``.
        """
        aux_loss = hidden_states.new_zeros(1).squeeze()

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
        if self.use_moe:
            mlp_output, aux_loss = self.mlp(hidden_states)
        else:
            mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, aux_loss, new_kv_cache


class MoETransformerDecoder(BaseModel):
    """Decoder-only Transformer with Mixture-of-Experts layers.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        intermediate_size: Per-expert intermediate dimension.
        num_experts: Number of experts per MoE layer.
        top_k: Number of experts activated per token.
        moe_layer_freq: Every ``moe_layer_freq``-th layer uses MoE
            (others use a dense MLP).  ``1`` means every layer is MoE.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        norm_type: Normalisation type.
        activation: Expert activation.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 11008,
        num_experts: int = 8,
        top_k: int = 2,
        moe_layer_freq: int = 1,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        tie_word_embeddings: bool = False,
        router_jitter: float = 0.0,
        attention_dropout: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            vocab_size = config.get("vocab_size", vocab_size)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            num_kv_heads = config.get("num_kv_heads", num_kv_heads)
            intermediate_size = config.get("intermediate_size", intermediate_size)
            num_experts = config.get("num_experts", num_experts)
            top_k = config.get("top_k", top_k)
            moe_layer_freq = config.get("moe_layer_freq", moe_layer_freq)
            max_seq_len = config.get("max_seq_len", max_seq_len)
            rope_theta = config.get("rope_theta", rope_theta)
            norm_type = config.get("norm_type", norm_type)
            activation = config.get("activation", activation)
            tie_word_embeddings = config.get("tie_word_embeddings", tie_word_embeddings)
            router_jitter = config.get("router_jitter", router_jitter)
            attention_dropout = config.get("attention_dropout", attention_dropout)

        super().__init__(config=config)

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.max_seq_len: int = max_seq_len
        self.moe_layer_freq: int = moe_layer_freq

        self.embed_tokens: TokenEmbedding = TokenEmbedding(vocab_size, hidden_size)
        self.rotary_emb: RotaryPositionEmbedding = RotaryPositionEmbedding(
            dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta
        )
        self.layers: nn.ModuleList = nn.ModuleList([
            MoETransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                norm_type=norm_type,
                activation=activation,
                use_moe=((layer_idx + 1) % moe_layer_freq == 0),
                router_jitter=router_jitter,
                attention_dropout=attention_dropout,
            )
            for layer_idx in range(num_layers)
        ])
        self.norm: nn.Module = _get_norm(norm_type, hidden_size)
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise weights."""
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
        """Build a causal attention mask (additive)."""
        total_len = seq_len + past_len
        causal = torch.triu(
            torch.full((seq_len, total_len), float("-inf"), device=device),
            diagonal=1 + past_len,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * float("-inf")
            mask = mask + pad_mask
        return mask

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask.
            kv_cache: Optional KV cache.
            use_cache: Whether to return the KV cache.

        Returns:
            A tuple ``(logits, aux_loss)``.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)

        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]
        total_len = seq_len + past_len
        cos, sin = self.rotary_emb.get_cos_sin(total_len, device)
        position_embeddings = (cos, sin)

        attn_mask = self._build_causal_mask(seq_len, past_len, device, attention_mask)

        total_aux_loss = hidden_states.new_zeros(1).squeeze()
        new_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None
            hidden_states, aux_loss, new_kv = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                new_cache.append(new_kv)

        # Average the auxiliary loss across layers.
        total_aux_loss = total_aux_loss / max(self.num_layers, 1)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if use_cache:
            self._last_kv_cache = new_cache
        return logits, total_aux_loss

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
            temperature: Sampling temperature.
            top_k: Top-k filtering (``0`` = disabled).
            top_p: Nucleus sampling threshold (``1.0`` = disabled).
            eos_token_id: Optional EOS token id.

        Returns:
            Generated token ids of shape ``(batch, seq_len + max_tokens)``.
        """
        self.eval()
        generated = input_ids
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        use_cache = True

        for step in range(max_tokens):
            if step == 0:
                logits, _ = self.forward(generated, kv_cache=kv_cache, use_cache=use_cache)
                next_logits = logits[:, -1, :]
            else:
                last_token = generated[:, -1:].contiguous()
                logits, _ = self.forward(last_token, kv_cache=kv_cache, use_cache=use_cache)
                next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, top_k)
                    min_values = values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < min_values, float("-inf")
                    )
                if top_p < 1.0:
                    next_logits = _top_p_filter(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
