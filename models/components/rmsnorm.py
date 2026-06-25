"""Root Mean Square Normalisation (RMSNorm).

This module is a thin compatibility wrapper around :class:`torch.nn.RMSNorm`
(introduced in PyTorch 2.4).  On older PyTorch versions (< 2.4) it falls
back to a hand-rolled pure-torch implementation with the same public
surface (``dim`` / ``eps`` attributes, ``extra_repr``), so the rest of
the framework can use it uniformly across torch versions.

Reference:
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class _RMSNormFallback(nn.Module):
    """Pure-torch RMSNorm implementation (PyTorch < 2.4 fallback).

    Computes ``y = x / sqrt(mean(x**2) + eps) * weight`` along the
    last ``dim`` dimension.  Public attributes match the native
    :class:`torch.nn.RMSNorm` so callers don't have to branch.
    """

    __constants__ = ["dim", "eps"]

    def __init__(self, dim: int, eps: float | None = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        self.dim: int = int(dim)
        self.eps: float = float(eps) if eps is not None else 1e-5
        # ``weight`` shape == (dim,) so the C++ native and the
        # fallback expose the same parameter.
        self.weight: nn.Parameter = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` shape: (..., dim)  -- normalise the last dim.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


#: Native :class:`torch.nn.RMSNorm` (PyTorch 2.4+) when available,
#: otherwise the hand-rolled pure-torch fallback above.
_native_rmsnorm = getattr(nn, "RMSNorm", None)


if _native_rmsnorm is not None:
    class RMSNorm(_native_rmsnorm):  # type: ignore[misc, valid-type]
        """RMSNorm -- thin subclass of :class:`torch.nn.RMSNorm` (torch >= 2.4).

        Adds the ``dim`` integer attribute (PyTorch stores it as a
        ``normalized_shape`` tuple) for backward compatibility with
        downstream code, and an ``extra_repr`` consistent with the
        rest of the framework.

        Args:
            dim: The size of the normalised (last) dimension.
            eps: Small constant added inside the square root for
                numerical stability.  ``None`` falls back to the
                PyTorch default (``1e-5``).
        """

        def __init__(self, dim: int, eps: float | None = None) -> None:
            if dim <= 0:
                raise ValueError(f"dim must be a positive integer, got {dim}.")
            # PyTorch's native RMSNorm uses normalized_shape (int or tuple).
            super().__init__(normalized_shape=dim, eps=eps, elementwise_affine=True)
            self.dim: int = int(dim)
            # ``eps`` may be ``None`` on the underlying module when the
            # user did not pass one; expose the resolved value for
            # diagnostics.
            self.eps: float = self.eps if self.eps is not None else 1e-5

        def extra_repr(self) -> str:
            return f"dim={self.dim}, eps={self.eps}"
else:
    # PyTorch < 2.4 -- the native class is not present; alias the
    # fallback so the public ``RMSNorm`` name works the same way.
    RMSNorm = _RMSNormFallback  # type: ignore[assignment,misc]
