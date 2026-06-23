"""Root Mean Square Normalisation (RMSNorm).

RMSNorm is a simplified variant of LayerNorm that only computes the
root-mean-square normalisation without subtracting the mean.  It is
computationally cheaper than LayerNorm and has become the default
normalisation in modern large language models (e.g. LLaMA, Mistral,
Qwen).

Reference:
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    For an input tensor ``x`` of shape ``(..., dim)`` the output is::

        y = x * rsqrt(mean(x^2, -1) + eps) * weight

    where ``weight`` is a learnable scale vector of shape ``(dim,)``.

    Args:
        dim: The size of the normalised (last) dimension.
        eps: Small constant added inside the square root for numerical
            stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        self.dim: int = dim
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    # ------------------------------------------------------------------
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the RMS normalisation (without the learnable scale)."""
        # Compute in float32 for numerical stability then cast back.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation followed by the learnable scale.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalised tensor of the same shape as ``x``.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
