"""v1.0.0 deeper HunyuanVideo adapter tests.

Exercises the v0.9.5 ``papers.adapters.hunyuan_video`` adapter on
behaviours that are not covered by the v0.9.5 HunyuanVideo smoke
suite:

* :func:`papers.adapters.hunyuan_video._materialise_per_block_map` --
  per-block ``{i}`` expansion across a key-rewrite table.
* :meth:`HunyuanVideoSampler.from_pretrained` -- friendly
  ``FileNotFoundError`` on a missing path.
* :meth:`HunyuanVideoVAE.from_pretrained` -- the same error path.
* :class:`HunyuanVideoSampler` -- parameter count after attaching
  a real :class:`HunyuanVideoVAE`.

4 tests; all CPU-only.
"""
from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn

from papers.adapters.hunyuan_video import (
    HUNYUAN_VIDEO_KEY_MAP,
    HUNYUAN_VIDEO_VAE_KEY_MAP,
    HunyuanVideoConfig,
    HunyuanVideoSampler,
    HunyuanVideoVAE,
    _materialise_per_block_map,
)


def _sampler_num_parameters(sampler: HunyuanVideoSampler) -> int:
    """Count all :class:`nn.Module` parameters reachable through the
    sampler's :attr:`dit`, :attr:`vae`, and :attr:`text_encoder`
    fields.

    The skeleton adapter does not currently expose a
    :meth:`HunyuanVideoSampler.num_parameters` helper, so the
    test defines one inline that walks the three well-known
    slots and ignores ``None`` entries.
    """
    total = 0
    for child in (sampler.dit, sampler.vae, sampler.text_encoder):
        if isinstance(child, nn.Module):
            total += sum(p.numel() for p in child.parameters())
    return total


# ---------------------------------------------------------------------------
# Section 1 -- _materialise_per_block_map (1 test)
# ---------------------------------------------------------------------------
class TestHunyuanVideoPerBlockMap:
    """Per-block ``{i}`` placeholder expansion."""

    def test_hunyuan_video_materialise_per_block_map(self):
        """``_materialise_per_block_map(num_blocks=3)`` expands every
        ``{i}`` placeholder to ``{0, 1, 2}`` for both the upstream
        and the local key."""
        expanded: Dict[str, str] = _materialise_per_block_map(
            HUNYUAN_VIDEO_KEY_MAP, num_layers=3,
        )
        # The map must be a dict and not empty.
        assert isinstance(expanded, dict)
        assert len(expanded) > 0
        # The upstream rules that mention ``double_blocks.{i}`` /
        # ``single_blocks.{i}`` must be expanded three times.
        n_double = sum(
            1 for k in expanded if k.startswith("double_blocks.")
        )
        n_single = sum(
            1 for k in expanded if k.startswith("single_blocks.")
        )
        # The upstream table has 18 ``double_blocks.{i}.*`` rules and
        # 9 ``single_blocks.{i}.*`` rules -- with num_layers=3 that
        # yields 54 + 27 = 81 per-block entries.  We assert the
        # *lower* bound so the test stays robust against future
        # upstream table edits.
        assert n_double >= 18 * 3
        assert n_single >= 9 * 3
        # And the per-block expansion is exactly {0, 1, 2}.
        block_indices = set()
        for k in expanded:
            if k.startswith("double_blocks."):
                idx = k.split(".")[1]
                block_indices.add(int(idx))
        assert block_indices == {0, 1, 2}, block_indices


# ---------------------------------------------------------------------------
# Section 2 -- from_pretrained error paths (2 tests)
# ---------------------------------------------------------------------------
class TestHunyuanVideoFromPretrainedErrors:
    """``from_pretrained`` raises a friendly error on missing paths."""

    def test_hunyuan_video_sampler_from_pretrained_raises_on_missing_path(self):
        """``HunyuanVideoSampler.from_pretrained('/nonexistent/...')``
        raises :class:`FileNotFoundError` (or a subclass) with a
        message that mentions HunyuanVideo."""
        with pytest.raises(FileNotFoundError) as exc_info:
            HunyuanVideoSampler.from_pretrained(
                "/nonexistent/hunyuan_video/dir",
            )
        # The error message should mention HunyuanVideo so the user
        # can debug the cause.
        assert "HunyuanVideo" in str(exc_info.value) or "hunyuan" in str(
            exc_info.value,
        ).lower()

    def test_hunyuan_video_vae_from_pretrained_raises_on_missing_path(self):
        """``HunyuanVideoVAE.from_pretrained('/nonexistent/...')``
        raises :class:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError) as exc_info:
            HunyuanVideoVAE.from_pretrained("/nonexistent/vae/path")
        assert "HunyuanVideoVAE" in str(exc_info.value) or "vae" in str(
            exc_info.value,
        ).lower()


# ---------------------------------------------------------------------------
# Section 3 -- Sampler parameter count (1 test)
# ---------------------------------------------------------------------------
class TestHunyuanVideoSamplerParameters:
    """``HunyuanVideoSampler`` exposes a positive parameter count."""

    def test_hunyuan_video_sampler_num_parameters_positive(self):
        """``HunyuanVideoSampler(HunyuanVideoConfig.tiny())`` has a
        positive parameter count once a real :class:`HunyuanVideoVAE`
        is attached.  The tiny config's VAE has well over 1K real
        parameters."""
        cfg = HunyuanVideoConfig.tiny()
        sampler = HunyuanVideoSampler(cfg)
        # By default the skeleton's dit / vae / text_encoder slots are
        # all ``None``; attach a real VAE so the parameter count is
        # non-zero.
        sampler.vae = HunyuanVideoVAE(
            in_channels=3, latent_channels=4, base_ch=64,
        )
        n = _sampler_num_parameters(sampler)
        assert isinstance(n, int)
        assert n > 0, f"expected positive parameter count, got {n}"
        # Tiny config has real parameters (at least 1K).
        assert n >= 1000, (
            f"expected the tiny config to expose at least 1000 "
            f"real parameters, got {n}"
        )
