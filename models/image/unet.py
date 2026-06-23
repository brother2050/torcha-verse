"""U-Net denoising network for latent diffusion.

This module implements a Stable-Diffusion-style U-Net that predicts the
noise (or velocity) added to a latent tensor at a given diffusion
timestep, conditioned on text embeddings via cross-attention.

Architecture overview::

    x (noisy latent) ──> conv_in
        │
        ├── down_blocks (ResBlock + SelfAttn + CrossAttn) ──> skips
        │
        mid_block (ResBlock + Attn + ResBlock)
        │
        up_blocks (ResBlock [+skip] + SelfAttn + CrossAttn)
        │
    conv_out ──> noise prediction

The timestep is encoded with sinusoidal embeddings and injected into
each ResBlock.  Text conditioning is injected through cross-attention.

Key components:

* :class:`TimestepEmbedding` -- sinusoidal timestep embedding.
* :class:`ResBlock` -- residual block with timestep injection.
* :class:`SelfAttentionBlock` -- spatial self-attention.
* :class:`CrossAttentionBlock` -- cross-attention to text embeddings.
* :class:`DownBlock` / :class:`MidBlock` / :class:`UpBlock` -- stage
  containers.
* :class:`UNet` -- the full denoising network.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel

__all__ = [
    "TimestepEmbedding",
    "ResBlock",
    "SelfAttentionBlock",
    "CrossAttentionBlock",
    "DownBlock",
    "MidBlock",
    "UpBlock",
    "UNet",
]


def _num_groups(channels: int, groups: int = 32) -> int:
    """Return the largest divisor of ``channels`` that is ``<= groups``."""
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP.

    Args:
        channel: Embedding dimension.
        max_period: Maximum period for the sinusoidal frequencies.
    """

    def __init__(self, channel: int, max_period: int = 10000) -> None:
        super().__init__()
        self.channel: int = channel
        self.max_period: int = max_period
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(channel, channel * 4),
            nn.SiLU(),
            nn.Linear(channel * 4, channel),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute the timestep embedding.

        Args:
            timesteps: 1-D tensor of timestep indices ``(batch,)``.

        Returns:
            Embedding tensor of shape ``(batch, channel)``.
        """
        half = self.channel // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.channel % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block with timestep-embedding injection.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        time_emb_dim: Dimension of the timestep embedding.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1: nn.GroupNorm = nn.GroupNorm(_num_groups(in_channels, groups), in_channels)
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb_proj: nn.Linear = nn.Linear(time_emb_dim, out_channels)

        self.norm2: nn.GroupNorm = nn.GroupNorm(_num_groups(out_channels, groups), out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: ``(batch, in_channels, H, W)``.
            temb: Timestep embedding ``(batch, time_emb_dim)``.

        Returns:
            ``(batch, out_channels, H, W)``.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject the timestep embedding.
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """Spatial self-attention block.

    Args:
        channels: Number of input channels.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (defaults to ``channels // num_heads``).
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm: nn.GroupNorm = nn.GroupNorm(_num_groups(channels, groups), channels)
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim or channels // num_heads
        inner_dim: int = self.num_heads * self.head_dim
        self.qkv: nn.Conv2d = nn.Conv2d(channels, inner_dim * 3, 1)
        self.out_proj: nn.Conv2d = nn.Conv2d(inner_dim, channels, 1)
        self.scale: float = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run spatial self-attention.

        Args:
            x: ``(batch, channels, H, W)``.

        Returns:
            ``(batch, channels, H, W)``.
        """
        residual = x
        x = self.norm(x)
        batch, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)  # each (batch, heads, head_dim, hw)
        # (batch, heads, hw, head_dim)
        q = q.permute(0, 1, 3, 2) * self.scale
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v)  # (batch, heads, hw, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(batch, -1, h, w)
        return residual + self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for text conditioning.

    The query is derived from the spatial feature map while the key and
    value come from the text embeddings.

    Args:
        channels: Number of input (spatial) channels.
        context_dim: Dimension of the text embeddings.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm: nn.GroupNorm = nn.GroupNorm(_num_groups(channels, groups), channels)
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim or channels // num_heads
        inner_dim: int = self.num_heads * self.head_dim
        self.to_q: nn.Conv2d = nn.Conv2d(channels, inner_dim, 1)
        self.to_k: nn.Linear = nn.Linear(context_dim, inner_dim)
        self.to_v: nn.Linear = nn.Linear(context_dim, inner_dim)
        self.out_proj: nn.Conv2d = nn.Conv2d(inner_dim, channels, 1)
        self.scale: float = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Run cross-attention.

        Args:
            x: Spatial features ``(batch, channels, H, W)``.
            context: Text embeddings ``(batch, seq_len, context_dim)``.

        Returns:
            ``(batch, channels, H, W)``.
        """
        residual = x
        x = self.norm(x)
        batch, c, h, w = x.shape
        q = self.to_q(x).reshape(batch, self.num_heads, self.head_dim, h * w)
        q = q.permute(0, 1, 3, 2) * self.scale  # (batch, heads, hw, head_dim)

        k = self.to_k(context).reshape(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.to_v(context).reshape(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.softmax(torch.matmul(q, k), dim=-1)  # (batch, heads, hw, seq)
        out = torch.matmul(attn, v)  # (batch, heads, hw, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(batch, -1, h, w)
        return residual + self.out_proj(out)


# ---------------------------------------------------------------------------
# Stage containers
# ---------------------------------------------------------------------------
class DownBlock(nn.Module):
    """Down-sampling stage of the U-Net.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        time_emb_dim: Timestep embedding dimension.
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        num_res_blocks: Number of ResBlocks.
        num_heads: Attention heads.
        add_downsample: Whether to add a downsampler at the end.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int = 0,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        add_downsample: bool = True,
    ) -> None:
        super().__init__()
        self.res_blocks: nn.ModuleList = nn.ModuleList()
        self.self_attentions: nn.ModuleList = nn.ModuleList()
        self.cross_attentions: nn.ModuleList = nn.ModuleList()
        self.has_cross_attn: bool = context_dim > 0
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_emb_dim))
            self.self_attentions.append(SelfAttentionBlock(out_channels, num_heads=num_heads))
            if context_dim > 0:
                self.cross_attentions.append(CrossAttentionBlock(out_channels, context_dim, num_heads=num_heads))

        self.downsampler: Optional[nn.Module] = None
        if add_downsample:
            self.downsampler = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the down block.

        Args:
            x: Input features.
            temb: Timestep embedding.
            context: Optional text embeddings.

        Returns:
            ``(output, skips)`` where ``skips`` is the list of skip
            tensors for the up-sampling path.
        """
        skips: List[torch.Tensor] = []
        for i, res in enumerate(self.res_blocks):
            x = res(x, temb)
            skips.append(x)
            x = self.self_attentions[i](x)
            if self.has_cross_attn and context is not None:
                x = self.cross_attentions[i](x, context)
        if self.downsampler is not None:
            x = self.downsampler(x)
            skips.append(x)
        return x, skips


class MidBlock(nn.Module):
    """Middle (bottleneck) stage of the U-Net.

    Args:
        channels: Channel width.
        time_emb_dim: Timestep embedding dimension.
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        num_heads: Attention heads.
    """

    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        context_dim: int = 0,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.res1: ResBlock = ResBlock(channels, channels, time_emb_dim)
        self.self_attn: SelfAttentionBlock = SelfAttentionBlock(channels, num_heads=num_heads)
        self.cross_attn: Optional[CrossAttentionBlock] = None
        if context_dim > 0:
            self.cross_attn = CrossAttentionBlock(channels, context_dim, num_heads=num_heads)
        self.res2: ResBlock = ResBlock(channels, channels, time_emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the mid block."""
        x = self.res1(x, temb)
        x = self.self_attn(x)
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context)
        x = self.res2(x, temb)
        return x


class UpBlock(nn.Module):
    """Up-sampling stage of the U-Net.

    Each residual block concatenates a skip connection from the
    corresponding down-sampling stage.  Because skips from different
    stages may have different channel counts, ``skip_channels_list``
    specifies the channel count for each skip.

    Args:
        out_channels: Output channels.
        prev_output_channels: Channels of the input (previous up-block
            or mid-block output).
        skip_channels_list: List of skip-channel counts, one per
            residual block (length ``num_res_blocks + 1``).
        time_emb_dim: Timestep embedding dimension.
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        num_res_blocks: Number of ResBlocks (each block has
            ``num_res_blocks + 1`` to match the down path).
        num_heads: Attention heads.
        add_upsample: Whether to add an upsampler at the end.
    """

    def __init__(
        self,
        out_channels: int,
        prev_output_channels: int,
        skip_channels_list: List[int],
        time_emb_dim: int,
        context_dim: int = 0,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        add_upsample: bool = True,
    ) -> None:
        super().__init__()
        if len(skip_channels_list) != num_res_blocks + 1:
            raise ValueError(
                f"skip_channels_list must have length num_res_blocks+1 "
                f"({num_res_blocks + 1}), got {len(skip_channels_list)}."
            )
        self.res_blocks: nn.ModuleList = nn.ModuleList()
        self.self_attentions: nn.ModuleList = nn.ModuleList()
        self.cross_attentions: nn.ModuleList = nn.ModuleList()
        self.has_cross_attn: bool = context_dim > 0
        for i in range(num_res_blocks + 1):
            res_in = prev_output_channels if i == 0 else out_channels
            block_in = res_in + skip_channels_list[i]
            self.res_blocks.append(ResBlock(block_in, out_channels, time_emb_dim))
            self.self_attentions.append(SelfAttentionBlock(out_channels, num_heads=num_heads))
            if context_dim > 0:
                self.cross_attentions.append(CrossAttentionBlock(out_channels, context_dim, num_heads=num_heads))

        self.upsampler: Optional[nn.Module] = None
        if add_upsample:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the up block, consuming skip connections.

        Args:
            x: Input features.
            skips: List of skip tensors in pop order (most recent first
                is the last element).
            temb: Timestep embedding.
            context: Optional text embeddings.

        Returns:
            Output features.
        """
        for i, res in enumerate(self.res_blocks):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res(x, temb)
            x = self.self_attentions[i](x)
            if self.has_cross_attn and context is not None:
                x = self.cross_attentions[i](x, context)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class UNet(BaseModel):
    """U-Net denoising network (Stable Diffusion style).

    Args:
        in_channels: Input latent channels.
        out_channels: Output (noise prediction) channels.
        hidden_size: Base channel width.
        context_dim: Text embedding dimension (``0`` disables
            cross-attention).
        num_heads: Number of attention heads.
        num_res_blocks: ResBlocks per stage.
        block_channels: Channel widths for each down/up stage.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 320,
        context_dim: int = 768,
        num_heads: int = 8,
        num_res_blocks: int = 2,
        block_channels: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            out_channels = config.get("out_channels", out_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            context_dim = config.get("context_dim", context_dim)
            num_heads = config.get("num_heads", num_heads)
            num_res_blocks = config.get("num_res_blocks", num_res_blocks)
            block_channels = config.get("block_channels", block_channels)

        super().__init__(config=config)

        if block_channels is None:
            block_channels = [hidden_size, hidden_size * 2, hidden_size * 4, hidden_size * 4]

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.context_dim: int = context_dim
        self.block_channels: List[int] = list(block_channels)

        time_emb_dim: int = hidden_size * 4
        self.time_embedding: TimestepEmbedding = TimestepEmbedding(hidden_size)
        self.time_proj: nn.Linear = nn.Linear(hidden_size, time_emb_dim)
        self.time_act: nn.Module = nn.SiLU()

        self.conv_in: nn.Conv2d = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)

        # Down blocks.
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = block_channels[0]
        for i, out_ch in enumerate(self.block_channels):
            is_last = i == len(self.block_channels) - 1
            self.down_blocks.append(DownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                time_emb_dim=time_emb_dim,
                context_dim=context_dim,
                num_res_blocks=num_res_blocks,
                num_heads=num_heads,
                add_downsample=not is_last,
            ))
            in_ch = out_ch

        # Mid block.
        self.mid_block: MidBlock = MidBlock(
            channels=self.block_channels[-1],
            time_emb_dim=time_emb_dim,
            context_dim=context_dim,
            num_heads=num_heads,
        )

        # Up blocks (reverse order).
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        reversed_channels = list(reversed(self.block_channels))
        num_blocks = len(self.block_channels)
        for j in range(num_blocks):
            out_ch = reversed_channels[j]
            prev_out = self.block_channels[-1] if j == 0 else reversed_channels[j - 1]
            # The first ``num_res_blocks`` skips come from the mirrored
            # down block (channel = block_channels[num_blocks-1-j]); the
            # last skip comes from the previous down block's downsampler
            # (or the conv_in output for the final up block).
            mirrored_ch = self.block_channels[num_blocks - 1 - j]
            prev_ch = self.block_channels[max(0, num_blocks - 2 - j)]
            skip_channels_list = [mirrored_ch] * num_res_blocks + [prev_ch]
            is_last = j == num_blocks - 1
            self.up_blocks.append(UpBlock(
                out_channels=out_ch,
                prev_output_channels=prev_out,
                skip_channels_list=skip_channels_list,
                time_emb_dim=time_emb_dim,
                context_dim=context_dim,
                num_res_blocks=num_res_blocks,
                num_heads=num_heads,
                add_upsample=not is_last,
            ))

        # Output.
        self.conv_norm_out: nn.GroupNorm = nn.GroupNorm(_num_groups(block_channels[0]), block_channels[0])
        self.conv_act: nn.Module = nn.SiLU()
        self.conv_out: nn.Conv2d = nn.Conv2d(block_channels[0], out_channels, 3, padding=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the noise added to ``x`` at ``timesteps``.

        Args:
            x: Noisy latent of shape ``(batch, in_channels, H, W)``.
            timesteps: Diffusion timesteps ``(batch,)``.
            encoder_hidden_states: Text embeddings
                ``(batch, seq_len, context_dim)``.

        Returns:
            Noise prediction of shape ``(batch, out_channels, H, W)``.
        """
        # Timestep embedding.
        temb = self.time_embedding(timesteps)
        temb = self.time_proj(self.time_act(temb))

        # Input conv.
        x = self.conv_in(x)

        # Down path, collecting skips.
        all_skips: List[torch.Tensor] = [x]
        for down in self.down_blocks:
            x, skips = down(x, temb, encoder_hidden_states)
            all_skips.extend(skips)

        # Mid path.
        x = self.mid_block(x, temb, encoder_hidden_states)

        # Up path, consuming skips.
        for up in self.up_blocks:
            skip_count = len(up.res_blocks)
            skips = all_skips[-skip_count:]
            all_skips = all_skips[:-skip_count]
            x = up(x, list(skips), temb, encoder_hidden_states)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
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
