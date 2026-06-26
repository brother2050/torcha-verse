"""Deeper coverage of :class:`VideoVAE` and :class:`VideoDiT` internals.

Exercises less-obvious code paths in the v0.9.5 video models that
are not yet covered by the existing test suite:

* :class:`ResBlock3D` -- the shortcut-conv branch (in_ch != out_ch).
* :class:`VideoVAE.reparameterize` -- the eval-mode no-noise branch.
* :class:`VideoVAE.generate` -- random-latent generation by shape.
* :class:`SpatioTemporalPatchEmbed` -- 3-D patch tokenisation
  output shape contract.

4 tests; all CPU-only.
"""
from __future__ import annotations

import pytest
import torch

from models.video.video_dit import SpatioTemporalPatchEmbed
from models.video.video_vae import ResBlock3D, VideoVAE


# ---------------------------------------------------------------------------
# Section 1 -- ResBlock3D (1 test)
# ---------------------------------------------------------------------------
class TestResBlock3D:
    """Tests for :class:`ResBlock3D` (3-D residual block)."""

    def test_video_vae_resblock_with_shortcut(self):
        """``ResBlock3D(in_ch, out_ch)`` with ``in_ch != out_ch`` uses
        the learned shortcut (1x1 conv) and returns a tensor of
        shape ``(B, out_ch, T, H, W)``."""
        in_ch = 8
        out_ch = 16
        block = ResBlock3D(in_channels=in_ch, out_channels=out_ch)
        block.eval()
        # T=4, H=8, W=8 spatial volume.
        x = torch.randn(1, in_ch, 4, 8, 8)
        with torch.no_grad():
            y = block(x)
        # Output has the *out* channel count and the same volume.
        assert y.shape == (1, out_ch, 4, 8, 8)
        # The shortcut is a learned 1x1 conv (not Identity) when the
        # channel counts differ.  We verify the type.
        assert isinstance(block.shortcut, torch.nn.Conv3d)


# ---------------------------------------------------------------------------
# Section 2 -- VideoVAE (2 tests)
# ---------------------------------------------------------------------------
class TestVideoVAE:
    """Tests for :class:`VideoVAE` (spatiotemporal VAE)."""

    def test_video_vae_reparameterize_eval_branch(self):
        """In ``eval()`` mode :meth:`VideoVAE.reparameterize` returns
        the mean (no noise) -- the KL term is then deterministic."""
        vae = VideoVAE(
            in_channels=3, latent_channels=4,
            hidden_size=16, num_down_blocks=1,
        )
        vae.eval()
        # Two deterministic latent vectors.
        mu = torch.zeros(1, 4, 4, 4, 4)
        logvar = torch.zeros(1, 4, 4, 4, 4)  # std=1, but unused in eval.
        with torch.no_grad():
            z = vae.reparameterize(mu, logvar)
        # In eval mode the reparameterisation is deterministic
        # (returns the mean).
        assert torch.equal(z, mu)
        # And the result is the mean regardless of logvar.
        logvar2 = torch.full_like(logvar, 5.0)
        with torch.no_grad():
            z2 = vae.reparameterize(mu, logvar2)
        assert torch.equal(z2, mu)

    def test_video_vae_generate_with_shape(self):
        """``VideoVAE.generate(shape=...)`` produces a tensor of
        that exact shape (decoded random latents)."""
        vae = VideoVAE(
            in_channels=3, latent_channels=4,
            hidden_size=16, num_down_blocks=1,
        )
        vae.eval()
        # The latent shape (B, C, T, H, W) is fed to decode() which
        # upsamples back to (B, 3, T*2, H*2, W*2).
        latent_shape = (1, 4, 4, 8, 8)
        with torch.no_grad():
            video = vae.generate(shape=latent_shape)
        # Decoder doubles each spatial dim (num_up_blocks == 1).
        assert video.shape == (1, 3, 4, 16, 16)
        # Output is finite.
        assert torch.isfinite(video).all()


# ---------------------------------------------------------------------------
# Section 3 -- SpatioTemporalPatchEmbed (1 test)
# ---------------------------------------------------------------------------
class TestSpatioTemporalPatchEmbed:
    """Tests for :class:`SpatioTemporalPatchEmbed` (3-D patch embed)."""

    def test_video_dit_patch_embed_output_shape(self):
        """``SpatioTemporalPatchEmbed(in_channels=3, hidden_size=64,
        patch_size=(2, 4, 4))`` flattens a
        ``(B=1, C=3, T=4, H=8, W=8)`` video into a sequence of
        ``(T/2) * (H/4) * (W/4) == 2 * 2 * 2 == 8`` tokens, each of
        dim 64.

        The :class:`nn.Conv3d` kernel/stride is ``patch_size``;
        the conv output has shape ``(B, embed_dim, T/2, H/4, W/4)``
        which is then flattened to ``(B, T/2 * H/4 * W/4, embed_dim)``.
        """
        embed = SpatioTemporalPatchEmbed(
            patch_size=(2, 4, 4),
            in_channels=3, hidden_size=64,
        )
        embed.eval()
        # (B=1, C=3, T=4, H=8, W=8) -- 3-D video latent volume.
        x = torch.randn(1, 3, 4, 8, 8)
        with torch.no_grad():
            tokens, t_p, h_p, w_p = embed(x)
        # Each of the three spatial dims is patched:
        #   T: 4 / 2 == 2 patches.
        #   H: 8 / 4 == 2 patches.
        #   W: 8 / 4 == 2 patches.
        # Total sequence length: 2 * 2 * 2 == 8.
        expected_seq_len = (4 // 2) * (8 // 4) * (8 // 4)
        assert expected_seq_len == 8
        assert t_p == 2 and h_p == 2 and w_p == 2
        assert tokens.shape == (1, expected_seq_len, 64)
