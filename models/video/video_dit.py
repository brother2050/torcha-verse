"""Video Diffusion Transformer (VideoDiT).

This module implements a Diffusion Transformer for video generation
(Sora-style).  The noisy video latent is split into spatiotemporal
patches, projected into tokens, and processed by a stack of
Transformer blocks that combine:

* adaptive LayerNorm-Zero (adaLN) for timestep conditioning,
* spatiotemporal self-attention,
* cross-attention for text conditioning,
* explicit temporal attention via a :class:`MotionModule`.

Key components:

* :class:`SpatioTemporalPatchEmbed` -- 3-D patch embedding.
* :class:`VideoDiTBlock` -- a single video DiT block.
* :class:`VideoDiT` -- the full video diffusion Transformer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.components.rmsnorm import RMSNorm
from models.image.unet import TimestepEmbedding
from .motion_module import MotionModule

__all__ = ["SpatioTemporalPatchEmbed", "VideoDiTBlock", "VideoDiT"]


class SpatioTemporalPatchEmbed(nn.Module):
    """Embed a video latent into a sequence of spatiotemporal tokens.

    Uses a 3-D convolution with kernel/stride equal to ``patch_size``
    to split the ``(T, H, W)`` volume into non-overlapping patches.

    Args:
        patch_size: Tuple ``(t, h, w)`` patch sizes.
        in_channels: Number of input channels.
        hidden_size: Output (token) dimension.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        in_channels: int = 4,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size: Tuple[int, int, int] = tuple(patch_size)  # type: ignore[assignment]
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.proj: nn.Conv3d = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """Embed the video into patch tokens.

        Args:
            x: Video latent of shape ``(batch, in_channels, T, H, W)``.

        Returns:
            ``(tokens, t_patches, h_patches, w_patches)`` where
            ``tokens`` has shape ``(batch, num_patches, hidden_size)``.
        """
        x = self.proj(x)  # (batch, hidden, T/pt, H/ph, W/pw)
        batch, _, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch, t*h*w, hidden)
        return x, t, h, w


class VideoDiTBlock(nn.Module):
    """A single video DiT block.

    Combines adaLN-Zero conditioned self-attention, cross-attention,
    and a temporal motion module.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_frames: Number of frames (for the motion module).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        norm_eps: Normalisation epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        num_frames: int = 16,
        context_dim: int = 0,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.context_dim: int = context_dim

        # adaLN modulation (6 * hidden_size).
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.norm1: nn.Module = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2: nn.Module = RMSNorm(hidden_size, eps=norm_eps)

        # Spatiotemporal self-attention.
        self.attn_q: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_k: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_v: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_out: nn.Linear = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Temporal motion module.
        self.motion_module: MotionModule = MotionModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_frames=num_frames,
            num_layers=1,
        )

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
            self.cross_k = nn.Linear(context_dim, num_heads * self.head_dim, bias=False)
            self.cross_v = nn.Linear(context_dim, num_heads * self.head_dim, bias=False)
            self.cross_out = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
            self.cross_gate = nn.Linear(hidden_size, hidden_size)

        # MLP.
        self.mlp_fc1: nn.Linear = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.mlp_fc2: nn.Linear = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

    # ------------------------------------------------------------------
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run multi-head self-attention."""
        batch, seq_len, _ = x.shape
        q = self.attn_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.attn_out(attn)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run cross-attention to ``context``."""
        batch, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        q = self.cross_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(context).view(batch, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(context).view(batch, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.cross_out(attn)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        shape_3d: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Run the video DiT block.

        Args:
            x: Patch tokens ``(batch, seq_len, hidden_size)``.
            temb: Timestep embedding ``(batch, hidden_size)``.
            context: Optional text embeddings.
            shape_3d: ``(t_patches, h_patches, w_patches)`` for the
                motion module reshape.

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

        # Temporal motion module (operates on a 5-D tensor).
        if shape_3d is not None:
            t_p, h_p, w_p = shape_3d
            batch = x.shape[0]
            x_5d = x.reshape(batch, t_p, h_p, w_p, self.hidden_size)
            x_5d = x_5d.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            x_5d = self.motion_module(x_5d)
            x = x_5d.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.hidden_size)

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


class VideoDiT(BaseModel):
    """Video Diffusion Transformer.

    Args:
        in_channels: Number of input latent channels.
        latent_channels: Number of latent channels (alias of in_channels).
        hidden_size: Model dimension.
        num_layers: Number of DiT blocks.
        num_heads: Number of attention heads.
        patch_size: Tuple ``(t, h, w)`` patch sizes.
        num_frames: Expected number of latent frames.
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: Optional[int] = None,
        hidden_size: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        num_frames: int = 16,
        context_dim: int = 4096,
        mlp_ratio: float = 4.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            latent_channels = config.get("latent_channels", latent_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            patch_size = config.get("patch_size", patch_size)
            num_frames = config.get("num_frames", num_frames)
            context_dim = config.get("context_dim", context_dim)
            mlp_ratio = config.get("mlp_ratio", mlp_ratio)

        if latent_channels is not None:
            in_channels = latent_channels

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.patch_size: Tuple[int, int, int] = tuple(patch_size)  # type: ignore[assignment]
        self.num_frames: int = num_frames
        self.context_dim: int = context_dim

        self.patch_embed: SpatioTemporalPatchEmbed = SpatioTemporalPatchEmbed(
            patch_size=self.patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        # Time embedding.
        self.time_embed: TimestepEmbedding = TimestepEmbedding(hidden_size)

        self.blocks: nn.ModuleList = nn.ModuleList([
            VideoDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_frames=num_frames,
                context_dim=context_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

        # Final adaLN + output.
        self.final_norm: nn.Module = RMSNorm(hidden_size)
        self.final_adaLN: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.final_linear: nn.Linear = nn.Linear(hidden_size, patch_volume * in_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        # Placeholder positional embedding (created lazily).
        self.pos_embed: Optional[nn.Parameter] = None

    # ------------------------------------------------------------------
    def _get_pos_embed(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Get or create the positional embedding for ``num_patches`` tokens."""
        if self.pos_embed is None or self.pos_embed.shape[1] < num_patches:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.hidden_size, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed[:, :num_patches]

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the noise added to the video latent ``x``.

        Args:
            x: Noisy video latent of shape ``(batch, in_channels, T, H, W)``.
            timesteps: Diffusion timesteps ``(batch,)``.
            encoder_hidden_states: Text embeddings
                ``(batch, seq_len, context_dim)``.

        Returns:
            Noise prediction of shape ``(batch, in_channels, T, H, W)``.
        """
        batch = x.shape[0]
        temb = self.time_embed(timesteps)

        x, t_p, h_p, w_p = self.patch_embed(x)
        pos = self._get_pos_embed(x.shape[1], x.device)
        x = x + pos

        for block in self.blocks:
            x = block(x, temb, encoder_hidden_states, shape_3d=(t_p, h_p, w_p))

        shift, scale = self.final_adaLN(temb).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)

        # Unpatchify.
        pt, ph, pw = self.patch_size
        x = x.reshape(batch, t_p, h_p, w_p, pt, ph, pw, self.in_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(
            batch,
            self.in_channels,
            t_p * pt,
            h_p * ph,
            w_p * pw,
        )
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
                ``(batch, in_channels, T, H, W)``.
            encoder_hidden_states: Optional text conditioning.
            num_steps: Number of denoising steps.

        Returns:
            Denoised video latent tensor.
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
