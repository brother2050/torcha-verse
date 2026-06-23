"""Temporal attention (motion) module for video models.

This module implements a motion module that performs cross-frame
temporal attention, allowing video models to capture motion dynamics
across the temporal dimension.

Key components:

* :class:`TemporalAttention` -- cross-frame temporal self-attention.
* :class:`MotionModule` -- a stack of temporal attention blocks with
  residual connections, operating on video tensors of shape
  ``(batch, channels, time, height, width)``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalAttention", "MotionModule"]


def _num_groups(channels: int, groups: int = 32) -> int:
    """Return the largest divisor of ``channels`` that is ``<= groups``.

    ``nn.GroupNorm`` requires ``channels`` to be divisible by the number
    of groups.  When the channel count is small or not a multiple of
    ``groups`` we fall back to the largest valid divisor.
    """
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class TemporalAttention(nn.Module):
    """Cross-frame temporal self-attention.

    For each spatial position, attends across the temporal dimension.
    The input ``(batch, channels, time, height, width)`` is reshaped so
    that the attention sequence is the time axis.

    Args:
        hidden_size: Channel dimension.
        num_heads: Number of attention heads.
        num_frames: Expected number of frames (used for positional info).
        head_dim: Dimension per head (defaults to ``hidden_size // num_heads``).
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_frames: int = 16,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim or hidden_size // num_heads
        self.num_frames: int = num_frames
        self.scale: float = self.head_dim ** -0.5

        inner_dim: int = self.num_heads * self.head_dim
        self.to_q: nn.Linear = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k: nn.Linear = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v: nn.Linear = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_out: nn.Linear = nn.Linear(inner_dim, hidden_size)
        self.dropout_p: float = dropout

        # Learnable temporal positional embedding.
        self.temporal_pos_emb: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_frames, 1, hidden_size)
        )
        nn.init.trunc_normal_(self.temporal_pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run cross-frame temporal attention.

        Args:
            x: Video tensor of shape ``(batch, channels, time, height, width)``.

        Returns:
            Output video tensor of the same shape.
        """
        batch, channels, time, height, width = x.shape
        if time > self.num_frames:
            # Slice the positional embedding if needed.
            pos_emb = self.temporal_pos_emb[:, :time]
        else:
            pos_emb = self.temporal_pos_emb[:, :time]

        # Reshape: (batch, channels, time, height, width)
        # -> (batch, height*width, time, channels)
        x = x.permute(0, 3, 4, 2, 1).reshape(batch, height * width, time, channels)
        x = x + pos_emb.transpose(1, 2)  # (1, time, 1, channels) broadcast

        # Merge batch and spatial dims for the attention computation.
        x = x.reshape(batch * height * width, time, channels)

        q = self.to_q(x).reshape(-1, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).reshape(-1, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).reshape(-1, time, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(-1, time, self.num_heads * self.head_dim)
        out = self.to_out(attn)

        # Reshape back to (batch, channels, time, height, width).
        out = out.reshape(batch, height, width, time, channels)
        out = out.permute(0, 4, 3, 1, 2).contiguous()
        return out


class MotionModule(nn.Module):
    """Motion module: a stack of temporal attention blocks.

    Applies temporal self-attention with residual connections to inject
    motion modelling into a video backbone.

    Args:
        hidden_size: Channel dimension.
        num_heads: Number of attention heads.
        num_frames: Expected number of frames.
        num_layers: Number of temporal attention blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_frames: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_frames: int = num_frames
        self.num_layers: int = num_layers

        self.norms: nn.ModuleList = nn.ModuleList([
            nn.GroupNorm(_num_groups(hidden_size, 32), hidden_size) for _ in range(num_layers)
        ])
        self.temporal_attns: nn.ModuleList = nn.ModuleList([
            TemporalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_frames=num_frames,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the motion module.

        Args:
            x: Video tensor of shape ``(batch, channels, time, height, width)``.

        Returns:
            Output video tensor of the same shape.
        """
        for norm, attn in zip(self.norms, self.temporal_attns):
            x = x + attn(norm(x))
        return x
