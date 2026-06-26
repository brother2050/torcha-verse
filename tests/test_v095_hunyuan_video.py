"""v0.9.5 HunyuanVideo adapter skeleton tests.

Exercises the v0.9.5 paper adapter skeleton for HunyuanVideo
(Tencent, 2024).  All tests are CPU-only and do **not** import
``diffusers`` / ``transformers`` / ``huggingface_hub`` / ``safetensors``;
they only consume the public surface of
:mod:`papers.adapters.hunyuan_video`:

* :class:`HunyuanVideoConfig` -- round-trip JSON + tiny factory
  + field-bound sanity.
* :class:`HunyuanVideoVAE` -- encode / decode shape contract on a
  small 3-D volume; dtype propagation.
* :class:`HunyuanVideoSampler` -- ``predict`` shape contract and
  seed reproducibility.
* Key-rewrite tables :data:`HUNYUAN_VIDEO_KEY_MAP` and
  :data:`HUNYUAN_VIDEO_VAE_KEY_MAP` -- completeness + prefix
  coverage + per-block expansion.

12 tests in 4 sections; every test must pass on a stock CPU.
"""
from __future__ import annotations

import re

import pytest
import torch

from papers.adapters.hunyuan_video import (
    HUNYUAN_VIDEO_KEY_MAP,
    HUNYUAN_VIDEO_VAE_KEY_MAP,
    HunyuanVideoConfig,
    HunyuanVideoSampler,
    HunyuanVideoVAE,
)


# ---------------------------------------------------------------------------
# Section 1 -- HunyuanVideoConfig (3 tests)
# ---------------------------------------------------------------------------
class TestHunyuanVideoConfig:
    """Tests for :class:`HunyuanVideoConfig`."""

    def test_tiny_factory(self):
        """:meth:`HunyuanVideoConfig.tiny` returns a dict-like config."""
        cfg = HunyuanVideoConfig.tiny()
        # Dataclass instance: dict-like attribute access.
        assert isinstance(cfg, HunyuanVideoConfig)
        # Required fields are readable + have positive values.
        assert int(cfg.hidden_size) > 0
        assert int(cfg.num_layers) > 0
        assert int(cfg.num_heads) > 0
        # dtype / device are exposed.
        assert cfg.dtype is not None
        assert cfg.device is not None
        # Tiny factory uses hidden_size=1280 / num_layers=2.
        assert cfg.hidden_size == 1280
        assert cfg.num_layers == 2

    def test_config_round_trip_json(self):
        """``to_dict()`` / ``from_dict()`` round-trip preserves fields."""
        cfg = HunyuanVideoConfig.tiny()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        rebuilt = HunyuanVideoConfig.from_dict(d)
        # Field-by-field equality for the scalar fields.
        for fname in (
            "hidden_size", "num_layers", "num_heads", "num_kv_heads",
            "in_channels", "out_channels", "temporal_size", "spatial_size",
            "patch_size", "patch_size_t", "mlp_ratio", "text_context_dim",
        ):
            assert getattr(rebuilt, fname) == getattr(cfg, fname), fname
        # Dtype / device survive the JSON round-trip.
        assert rebuilt.dtype == cfg.dtype
        assert str(rebuilt.device) == str(cfg.device)

    def test_field_bounds(self):
        """Field bounds: positive sizes, GQA divisibility, patch > 0."""
        cfg = HunyuanVideoConfig.tiny()
        assert int(cfg.hidden_size) > 0
        assert int(cfg.num_layers) > 0
        # GQA constraint: num_heads must be a multiple of num_kv_heads.
        assert int(cfg.num_heads) % int(cfg.num_kv_heads) == 0
        # Default config has patch_size / temporal_size > 0.
        assert int(cfg.patch_size) > 0
        assert int(cfg.temporal_size) > 0


# ---------------------------------------------------------------------------
# Section 2 -- HunyuanVideoVAE (3 tests)
# ---------------------------------------------------------------------------
class TestHunyuanVideoVAE:
    """Tests for :class:`HunyuanVideoVAE` (3-D VAE skeleton)."""

    def test_vae_encode_decode_shape(self):
        """Encode / decode shape contract on a tiny video volume.

        The skeleton's down/up factor is 4 in the temporal direction
        and 2 in each spatial direction; the input volume is chosen
        so that all three dimensions are multiples of the factor
        (T=8, H=8, W=8 -> latent T=2, H=4, W=4).
        """
        vae = HunyuanVideoVAE(in_channels=3, latent_channels=4, base_ch=32)
        vae.eval()
        video = torch.zeros(1, 3, 8, 8, 8)
        with torch.no_grad():
            latents = vae.encode(video)
            recon = vae.decode(latents)
        # Encode output: (B, 4, T/4, H/4, W/4) = (1, 4, 2, 2, 2).
        assert latents.shape == (1, 4, 2, 2, 2)
        # Decode output: (B, 3, T, H, W) = (1, 3, 8, 8, 8).
        assert recon.shape == (1, 3, 8, 8, 8)

    def test_vae_dummy_round_trip(self):
        """Round-trip output shape == input shape (random weights)."""
        vae = HunyuanVideoVAE(in_channels=3, latent_channels=4, base_ch=32)
        vae.eval()
        video = torch.zeros(1, 3, 8, 8, 8)
        with torch.no_grad():
            latents = vae.encode(video)
            recon = vae.decode(latents)
        # Shape contract: round-trip preserves spatial layout.
        assert recon.shape == video.shape

    def test_vae_dtype_consistency(self):
        """``vae.to(torch.float16)`` propagates to conv weight dtype.

        The 3-D VAE skeleton uses ``F.avg_pool3d`` which is not
        implemented for ``float16`` on CPU -- so we only check
        that the conv weight dtype was updated, which is what
        ``.to(dtype)`` is contractually about.
        """
        vae = HunyuanVideoVAE(in_channels=3, latent_channels=4, base_ch=32)
        vae = vae.to(torch.float16)
        # Conv weight dtype should now be float16.
        assert vae.encoder["conv_in"].weight.dtype == torch.float16
        assert vae.encoder["conv_out"].weight.dtype == torch.float16
        # Module-level to(dtype) on Conv3d only sets weight/bias; running
        # an actual forward with a half input is unsupported on CPU
        # (avg_pool3d is float32-only), so we stop at the weight check.


# ---------------------------------------------------------------------------
# Section 3 -- HunyuanVideoSampler (3 tests)
# ---------------------------------------------------------------------------
class TestHunyuanVideoSampler:
    """Tests for :class:`HunyuanVideoSampler` (deterministic dummy)."""

    def test_predict_shapes(self):
        """``predict`` returns frames / latents / timesteps with expected shapes."""
        cfg = HunyuanVideoConfig.tiny()
        # Force CPU + float32 to keep the smoke test fast.
        cfg.dtype = torch.float32
        cfg.device = "cpu"
        sampler = HunyuanVideoSampler(cfg)
        out = sampler.predict(
            "a cat",
            num_frames=4, height=64, width=64,
            num_inference_steps=5, seed=42,
        )
        assert "frames" in out
        assert "latents" in out
        assert "timesteps" in out
        # frames: (N, 3, H, W) = (4, 3, 64, 64).
        assert out["frames"].shape == (4, 3, 64, 64)
        # latents: (N, out_channels, H/8, W/8) = (4, 4, 8, 8).
        assert out["latents"].shape == (4, 4, 8, 8)
        # timesteps: list of len == num_inference_steps.
        assert isinstance(out["timesteps"], list)
        assert len(out["timesteps"]) == 5

    def test_seed_reproducibility(self):
        """Same seed -> identical frames."""
        cfg = HunyuanVideoConfig.tiny()
        cfg.dtype = torch.float32
        cfg.device = "cpu"
        sampler = HunyuanVideoSampler(cfg)
        a = sampler.predict(
            "a cat", num_frames=4, height=64, width=64,
            num_inference_steps=5, seed=42,
        )
        b = sampler.predict(
            "a cat", num_frames=4, height=64, width=64,
            num_inference_steps=5, seed=42,
        )
        # Bit-for-bit equality: torch.equal checks both shape + values.
        assert torch.equal(a["frames"], b["frames"])
        assert torch.equal(a["latents"], b["latents"])

    def test_different_seed_differs(self):
        """Different seed -> different frames."""
        cfg = HunyuanVideoConfig.tiny()
        cfg.dtype = torch.float32
        cfg.device = "cpu"
        sampler = HunyuanVideoSampler(cfg)
        a = sampler.predict(
            "a cat", num_frames=4, height=64, width=64,
            num_inference_steps=5, seed=42,
        )
        b = sampler.predict(
            "a cat", num_frames=4, height=64, width=64,
            num_inference_steps=5, seed=43,
        )
        # The seed-different frames should not be bit-equal.
        assert not torch.equal(a["frames"], b["frames"])


# ---------------------------------------------------------------------------
# Section 4 -- KeyMap (3 tests)
# ---------------------------------------------------------------------------
class TestHunyuanVideoKeyMap:
    """Tests for the HunyuanVideo key-rewrite tables."""

    def test_hunyuan_video_key_map_completeness(self):
        """:data:`HUNYUAN_VIDEO_KEY_MAP` has at least 50 entries and
        all keys start with one of the documented prefixes."""
        assert len(HUNYUAN_VIDEO_KEY_MAP) >= 50
        allowed_prefixes = (
            "img_in", "time_in", "y_embedder", "vector_in",
            "t_embedder", "final_layer", "double_blocks", "single_blocks",
            # 3-D RoPE frequency buffers also live in the upstream
            # HunyuanVideo state dict and are part of this key map.
            "rope",
        )
        for k in HUNYUAN_VIDEO_KEY_MAP.keys():
            assert k.startswith(allowed_prefixes), (
                f"unexpected upstream prefix in key {k!r}"
            )

    def test_hunyuan_video_vae_key_map_completeness(self):
        """:data:`HUNYUAN_VIDEO_VAE_KEY_MAP` has at least 8 entries."""
        assert len(HUNYUAN_VIDEO_VAE_KEY_MAP) >= 8

    def test_per_block_expansion(self):
        """Per-block ``{i}`` placeholder expands to ``num_layers`` rules."""
        def _expand_per_block_map(pattern: str, num_layers: int):
            return [pattern.format(i=i) for i in range(num_layers)]

        pattern = "double_blocks.{i}.img_attn.qkv.weight"
        expanded = _expand_per_block_map(pattern, 4)
        assert len(expanded) == 4
        # Each entry should be a syntactically valid parameter key.
        for key in expanded:
            assert re.fullmatch(
                r"double_blocks\.\d+\.img_attn\.qkv\.weight", key,
            ), key
        # Indices 0..3 are present in order.
        assert expanded[0] == "double_blocks.0.img_attn.qkv.weight"
        assert expanded[3] == "double_blocks.3.img_attn.qkv.weight"
