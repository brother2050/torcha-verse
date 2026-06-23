"""Diffusion Transformer (DiT).

This module implements a Diffusion Transformer (DiT) in the style of
SD3 / Sora: the noisy latent is split into patches, projected into
tokens, and processed by a stack of Transformer blocks that use
adaptive LayerNorm (adaLN-Zero) for timestep conditioning and
cross-attention for text conditioning.

Key components:

* :class:`PatchEmbed` -- image-to-patch embedding.
* :class:`DiTBlock` -- a single DiT block with adaLN-Zero.
* :class:`DiT` -- the full Diffusion Transformer.

Reference:
    Peebles & Xie, "Scalable Diffusion Models with Transformers" (2022).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel
from models.components.rope import RotaryPositionEmbedding
from models.components.rmsnorm import RMSNorm
from .unet import TimestepEmbedding

__all__ = ["PatchEmbed", "DiTBlock", "DiT"]


class PatchEmbed(nn.Module):
    """Convert an image (or latent) into a sequence of patch tokens.

    Args:
        patch_size: Size of each square patch.
        in_channels: Number of input channels.
        hidden_size: Output (token) dimension.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Embed the image into patch tokens.

        Args:
            x: Image tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            A tuple ``(tokens, h_patches, w_patches)`` where ``tokens``
            has shape ``(batch, num_patches, hidden_size)``.
        """
        x = self.proj(x)  # (batch, hidden, H/p, W/p)
        batch, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch, h*w, hidden)
        return x, h, w


class DiTBlock(nn.Module):
    """A single DiT block with adaptive LayerNorm-Zero conditioning.

    The timestep embedding produces six modulation vectors (shift, scale,
    gate) for the attention and MLP sub-layers.  The gates are
    initialised to zero so the block is initially an identity function
    (adaLN-Zero), which stabilises training.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (GQA).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        norm_eps: Normalisation epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        context_dim: int = 0,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads or num_heads
        self.head_dim: int = hidden_size // num_heads
        self.context_dim: int = context_dim

        # adaLN modulation: produces 6 * hidden_size parameters.
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.norm1: nn.Module = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2: nn.Module = RMSNorm(hidden_size, eps=norm_eps)

        # Self-attention.
        self.attn_q: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_k: nn.Linear = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.attn_v: nn.Linear = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.attn_out: nn.Linear = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Optional cross-attention.
        self.cross_attn_norm: Optional[nn.Module] = None
        self.cross_q: Optional[nn.Linear] = None
        self.cross_k: Optional[nn.Linear] = None
        self.cross_v: Optional[nn.Linear] = None
        self.cross_out: Optional[nn.Linear] = None
        self.cross_gate: Optional[nn.Linear] = None
        if context_dim > 0:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.cross_q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
            self.cross_k = nn.Linear(context_dim, self.num_kv_heads * self.head_dim, bias=False)
            self.cross_v = nn.Linear(context_dim, self.num_kv_heads * self.head_dim, bias=False)
            self.cross_out = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
            self.cross_gate = nn.Linear(hidden_size, hidden_size)

        # MLP.
        self.mlp_fc1: nn.Linear = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.mlp_fc2: nn.Linear = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

        self.scale: float = self.head_dim ** -0.5

    # ------------------------------------------------------------------
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run multi-head self-attention (GQA)."""
        batch, seq_len, _ = x.shape
        q = self.attn_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Repeat KV heads.
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.attn_out(attn)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run cross-attention to ``context``."""
        batch, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        q = self.cross_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(context).view(batch, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(context).view(batch, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.cross_out(attn)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the DiT block.

        Args:
            x: Patch tokens ``(batch, seq_len, hidden_size)``.
            temb: Timestep embedding ``(batch, hidden_size)``.
            context: Optional text embeddings ``(batch, ctx_len, context_dim)``.

        Returns:
            Output tokens ``(batch, seq_len, hidden_size)``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(temb).chunk(6, dim=-1)
        )

        # Self-attention with adaLN.
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self._self_attention(h)
        x = x + gate_msa.unsqueeze(1) * h

        # Cross-attention (gated).
        if self.cross_attn_norm is not None and context is not None:
            h = self.cross_attn_norm(x)
            h = self._cross_attention(h, context)
            x = x + torch.sigmoid(self.cross_gate(temb)).unsqueeze(1) * h

        # MLP with adaLN.
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = F.gelu(self.mlp_fc1(h))
        h = self.mlp_fc2(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiT(BaseModel):
    """Diffusion Transformer.

    Args:
        input_size: Spatial size of the input latent (assumed square).
        patch_size: Patch size.
        in_channels: Number of input latent channels.
        hidden_size: Model dimension.
        num_layers: Number of DiT blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        context_dim: int = 4096,
        mlp_ratio: float = 4.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            input_size = config.get("input_size", input_size)
            patch_size = config.get("patch_size", patch_size)
            in_channels = config.get("in_channels", in_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            num_kv_heads = config.get("num_kv_heads", num_kv_heads)
            context_dim = config.get("context_dim", context_dim)
            mlp_ratio = config.get("mlp_ratio", mlp_ratio)

        super().__init__(config=config)

        self.input_size: int = input_size
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads or num_heads
        self.context_dim: int = context_dim

        self.patch_embed: PatchEmbed = PatchEmbed(patch_size, in_channels, hidden_size)
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep embedding.
        self.time_embed: TimestepEmbedding = TimestepEmbedding(hidden_size)

        self.blocks: nn.ModuleList = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                context_dim=context_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

        # Final adaLN + output projection.
        self.final_norm: nn.Module = RMSNorm(hidden_size)
        self.final_adaLN: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_linear: nn.Linear = nn.Linear(
            hidden_size, patch_size * patch_size * in_channels
        )
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the noise added to ``x``.

        Args:
            x: Noisy latent of shape ``(batch, in_channels, H, W)``.
            timesteps: Diffusion timesteps ``(batch,)``.
            encoder_hidden_states: Text embeddings
                ``(batch, seq_len, context_dim)``.

        Returns:
            Noise prediction of shape ``(batch, in_channels, H, W)``.
        """
        batch, _, h, w = x.shape
        temb = self.time_embed(timesteps)

        x, ph, pw = self.patch_embed(x)
        x = x + self.pos_embed[:, : x.shape[1], :]

        for block in self.blocks:
            x = block(x, temb, encoder_hidden_states)

        # Final adaLN.
        shift, scale = self.final_adaLN(temb).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)

        # Unpatchify.
        x = x.reshape(batch, ph, pw, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch, self.in_channels, ph * self.patch_size, pw * self.patch_size)
        return x

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, ...],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a simple DDPM-style denoising loop.

        Args:
            shape: Shape of the latent to generate
                ``(batch, in_channels, H, W)``.
            encoder_hidden_states: Optional text conditioning.
            num_steps: Number of denoising steps.

        Returns:
            Denoised latent tensor.
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(num_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            noise_pred = self.forward(x, t, encoder_hidden_states)
            alpha = 1.0 - i / num_steps
            x = (x - (1 - alpha) * noise_pred) / max(alpha, 1e-4)
        return x
