"""Root Mean Square Normalisation (RMSNorm).

This module is a thin compatibility wrapper around :class:`torch.nn.RMSNorm`
(introduced in PyTorch 2.4).  It exists to:

* Keep the public API stable for downstream code that reads
  ``module.dim`` and ``module.eps`` attributes and uses the class as a
  drop-in replacement for the previous hand-rolled implementation.
* Defer to the PyTorch-native kernel, which is implemented in C++/CUDA
  and is both faster and numerically more stable than the previous
  Python implementation.

Reference:
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.RMSNorm):
    """RMSNorm -- thin subclass of :class:`torch.nn.RMSNorm`.

    Adds the ``dim`` integer attribute (PyTorch stores it as a
    ``normalized_shape`` tuple) for backward compatibility with
    downstream code, and an ``extra_repr`` consistent with the rest of
    the framework.

    Args:
        dim: The size of the normalised (last) dimension.
        eps: Small constant added inside the square root for numerical
            stability.  ``None`` falls back to the PyTorch default
            (``1e-5``).
    """

    def __init__(self, dim: int, eps: float | None = None) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        # PyTorch's native RMSNorm uses normalized_shape (int or tuple).
        super().__init__(normalized_shape=dim, eps=eps, elementwise_affine=True)
        self.dim: int = dim
        # ``eps`` may be ``None`` on the underlying module when the user
        # did not pass one; expose the resolved value for diagnostics.
        self.eps: float = self.eps if self.eps is not None else 1e-5

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
