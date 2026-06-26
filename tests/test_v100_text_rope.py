"""v1.00 -- Tests for the Rotary Position Embedding (components/rope.py).

Covers the scaling branches and the forward path:

* ``rotate_half`` -- element-wise correctness.
* ``_compute_inv_freq`` -- ntk-aware rescaling; linear is a no-op at the
  ``inv_freq`` level (it instead interpolates the cos/sin tables).
* ``get_cos_sin`` -- the returned tables satisfy ``cos^2 + sin^2 == 1``.

All tests are CPU-only.
"""
from __future__ import annotations

import torch

from models.components.rope import (
    RotaryPositionEmbedding,
    rotate_half,
)


# ===========================================================================
# 1. rotate_half
# ===========================================================================
class TestRotateHalf:
    """``rotate_half`` swaps the two halves of the last dim with a sign flip."""

    def test_rotate_half_correctness(self) -> None:
        """``rotate_half(x) == concat(-x[..., half:], x[..., :half])``."""
        torch.manual_seed(0)
        x = torch.randn(2, 4, 7, 32)
        out = rotate_half(x)

        assert out.shape == x.shape
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        expected = torch.cat((-x2, x1), dim=-1)
        assert torch.allclose(out, expected)


# ===========================================================================
# 2. _compute_inv_freq -- ntk-aware
# ===========================================================================
class TestComputeInvFreq:
    """``_compute_inv_freq`` under the various scaling strategies."""

    def test_compute_inv_freq_ntk_aware_changes(self) -> None:
        """NTK-aware scaling changes ``inv_freq`` relative to the default."""
        # The constructor accepts a dict (not a plain string).
        base = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        scaled = RotaryPositionEmbedding(
            dim=64,
            max_seq_len=128,
            rope_scaling={"type": "ntk-aware", "factor": 2.0},
        )
        # NTK-aware rescales the base frequency, so the inv_freq must differ.
        assert not torch.allclose(base.inv_freq, scaled.inv_freq, atol=1e-6)

    def test_compute_inv_freq_linear_interpolation(self) -> None:
        """Linear scaling does not modify ``inv_freq`` (it only resamples the
        cos/sin tables downstream).  ``_compute_inv_freq`` therefore returns
        the same buffer as the unscaled base.
        """
        base = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        linear = RotaryPositionEmbedding(
            dim=64,
            max_seq_len=128,
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        # Linear scaling is a no-op at the inv_freq level.
        assert torch.allclose(base.inv_freq, linear.inv_freq, atol=1e-6)


# ===========================================================================
# 3. get_cos_sin -- the cos/sin tables
# ===========================================================================
class TestGetCosSin:
    """``get_cos_sin`` returns a (cos, sin) pair of the right shape."""

    def test_get_cos_sin_for_seq_len(self) -> None:
        """The returned cos/sin satisfy ``cos^2 + sin^2 == 1`` on the
        rotated half-dim positions.
        """
        torch.manual_seed(0)
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        cos, sin = rope.get_cos_sin(seq_len=64)
        assert isinstance(cos, torch.Tensor)
        assert isinstance(sin, torch.Tensor)
        # ``get_cos_sin`` returns a (seq_len, head_dim) pair.
        assert cos.shape == (64, 64)
        assert sin.shape == (64, 64)

        # The implementation duplicates ``freqs`` along the last dim, so
        # ``cos[:, k::2]`` and ``sin[:, k::2]`` share the same underlying
        # frequency.  Checking either column should give ``cos^2 + sin^2 = 1``.
        c = cos[:, 0::2]
        s = sin[:, 0::2]
        assert c.shape == s.shape
        assert torch.allclose(c * c + s * s, torch.ones_like(c), atol=1e-5)
