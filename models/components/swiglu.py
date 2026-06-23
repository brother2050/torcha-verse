"""SwiGLU (Swish-Gated Linear Unit) activation / feed-forward block.

SwiGLU is the gated activation used in the feed-forward networks of
modern large language models (LLaMA, PaLM).  It projects the input
through two parallel linear layers, applies a Swish (SiLU) gate to one
of them, multiplies them element-wise, and projects the result back to
the original dimension.

Reference:
    Shazeer, "GLU Variants Improve Transformer" (2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit feed-forward block.

    The forward pass is::

        h = silu(w1(x)) * w2(x)   # gated activation
        y = w3(h)                  # down-projection back to ``dim``

    where ``w1`` and ``w2`` are the two gating projections and ``w3`` is
    the output projection.

    Args:
        dim: Input and output dimension.
        hidden_dim: Inner (hidden) dimension of the block.
        bias: Whether to include bias terms in the linear layers.
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}.")
        self.dim: int = dim
        self.hidden_dim: int = hidden_dim
        # Gate projection (Swish applied to this).
        self.w1: nn.Linear = nn.Linear(dim, hidden_dim, bias=bias)
        # Up projection (multiplied with the gated value).
        self.w2: nn.Linear = nn.Linear(dim, hidden_dim, bias=bias)
        # Down projection back to the original dimension.
        self.w3: nn.Linear = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SwiGLU gated feed-forward output.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim)``.
        """
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.w3(gate * up)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"
