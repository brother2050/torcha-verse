"""v0.9.0 ControlNet + IP-Adapter adapter skeleton tests.

Exercises the project-internal ControlNet and IP-Adapter adapter
skeletons.  All tests are CPU-only and do **not** import
``diffusers`` / ``transformers`` / ``huggingface_hub`` / ``safetensors``;
they only consume the public surface of
:mod:`papers.adapters.controlnet` and
:mod:`papers.adapters.ip_adapter`.

12 tests in 2 sections (6 + 6); every test must pass on a stock CPU.
"""
from __future__ import annotations

import pytest
import torch

from papers.adapters.controlnet import (
    CONTROLNET_KEY_MAP,
    ControlNetAdapter,
    ControlNetConfig,
)
from papers.adapters.ip_adapter import (
    IP_ADAPTER_KEY_MAP,
    IPAdapter,
    IPAdapterConfig,
    MockImageEncoder,
)


# ---------------------------------------------------------------------------
# Section A -- ControlNet (6 tests)
# ---------------------------------------------------------------------------
class TestControlNet:
    """Tests for the ControlNet adapter skeleton."""

    def test_controlnet_config_defaults(self):
        """:class:`ControlNetConfig` defaults are readable."""
        cfg = ControlNetConfig()
        assert int(cfg.in_channels) > 0
        assert int(cfg.hint_channels) > 0
        assert int(cfg.num_layers) > 0
        # block_out_channels has length == num_layers (Tiny preset is 4).
        assert len(cfg.block_out_channels) == cfg.num_layers
        assert all(int(c) > 0 for c in cfg.block_out_channels)
        # downsample flag is a bool.
        assert isinstance(cfg.downsample, bool)

    def test_controlnet_config_overrides(self):
        """Explicit field overrides are respected."""
        # The ``__post_init__`` check enforces
        # ``len(block_out_channels) == num_layers`` -- so we must
        # pass them in a consistent way.
        cfg = ControlNetConfig(
            in_channels=1, num_layers=2,
            block_out_channels=(32, 64),
        )
        # The override values land on the dataclass.
        assert cfg.in_channels == 1
        assert cfg.num_layers == 2
        assert tuple(cfg.block_out_channels) == (32, 64)
        # A 4-layer config also works.
        cfg2 = ControlNetConfig(
            in_channels=1, num_layers=4,
            block_out_channels=(32, 64, 128, 256),
        )
        assert cfg2.in_channels == 1
        assert cfg2.num_layers == 4
        assert tuple(cfg2.block_out_channels) == (32, 64, 128, 256)

    def test_controlnet_adapter_forward_shape(self):
        """``ControlNetAdapter.forward`` returns ``num_layers + 1`` residuals."""
        cfg = ControlNetConfig(num_layers=2, block_out_channels=(32, 64))
        adapter = ControlNetAdapter(cfg).eval()
        latent = torch.randn(1, 4, 8, 8)
        control_image = torch.randn(1, 3, 8, 8)
        t = torch.tensor([500])
        ehs = torch.randn(1, 24, 64)
        with torch.no_grad():
            residuals = adapter(latent, control_image, t, ehs)
        # num_layers=2 -> 2 per-block + 1 mid-block = 3 residuals.
        assert len(residuals) == cfg.num_layers + 1
        # Each per-block residual should be 4-D (B, C, H, W).
        for r in residuals:
            assert r.dim() == 4
            assert r.shape[0] == 1

    def test_controlnet_key_map_count(self):
        """:data:`CONTROLNET_KEY_MAP` has at least 20 entries."""
        assert len(CONTROLNET_KEY_MAP) >= 20

    def test_controlnet_key_map_no_collision(self):
        """No duplicate values in :data:`CONTROLNET_KEY_MAP`."""
        values = list(CONTROLNET_KEY_MAP.values())
        assert len(values) == len(set(values))
        # Also: no duplicate keys (this is a property of ``dict``
        # but is worth asserting on the source mapping for
        # robustness).
        keys = list(CONTROLNET_KEY_MAP.keys())
        assert len(keys) == len(set(keys))

    def test_controlnet_apply_smoke(self):
        """``adapter.apply`` returns a tensor of the same shape as
        ``conditioning``."""
        cfg = ControlNetConfig(num_layers=2, block_out_channels=(32, 64))
        adapter = ControlNetAdapter(cfg).eval()
        # Conditioning + control at the same spatial resolution.
        conditioning = torch.randn(1, 4, 8, 8)
        control_image = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            out = adapter.apply(conditioning, control_image, strength=0.5)
        assert out.shape == conditioning.shape


# ---------------------------------------------------------------------------
# Section B -- IP-Adapter (6 tests)
# ---------------------------------------------------------------------------
class TestIPAdapter:
    """Tests for the IP-Adapter adapter skeleton."""

    def test_ipadapter_config_defaults(self):
        """:class:`IPAdapterConfig` defaults are readable."""
        cfg = IPAdapterConfig()
        assert int(cfg.image_embed_dim) > 0
        assert int(cfg.cross_attention_dim) > 0
        assert int(cfg.num_tokens) > 0
        assert int(cfg.num_images) > 0
        assert int(cfg.num_layers) > 0
        # Scale in [0, 2] as per __post_init__ validation.
        assert 0.0 <= float(cfg.scale) <= 2.0

    def test_ipadapter_encode_image_shape(self):
        """``encode_image`` returns ``(B, image_embed_dim)``.

        The skeleton's :class:`MockImageEncoder` is a stand-in for
        the real CLIP image encoder: it takes a pre-pooled
        ``(B, image_embed_dim)`` tensor rather than a raw image.
        """
        cfg = IPAdapterConfig(
            image_embed_dim=64, cross_attention_dim=32,
            num_tokens=2, num_layers=2,
        )
        adapter = IPAdapter(cfg).eval()
        # Mock encoder input: (B, image_embed_dim).
        image = torch.randn(1, cfg.image_embed_dim)
        with torch.no_grad():
            encoded = adapter.encode_image(image)
        assert encoded.shape == (1, cfg.image_embed_dim)
        # Mock encoder exposes its image_embed_dim contract.
        assert adapter.mock_image_encoder.image_embed_dim == 64
        assert adapter.mock_image_encoder._is_mock_encoder is True

    def test_ipadapter_get_image_features_count(self):
        """``get_image_features`` returns ``num_layers`` entries."""
        cfg = IPAdapterConfig(
            image_embed_dim=64, cross_attention_dim=32,
            num_tokens=2, num_layers=2,
        )
        adapter = IPAdapter(cfg).eval()
        image_embeds = torch.randn(1, cfg.image_embed_dim)
        with torch.no_grad():
            features = adapter.get_image_features(image_embeds)
        assert len(features) == cfg.num_layers
        for f in features:
            assert f.shape == (1, cfg.num_tokens, cfg.cross_attention_dim)

    def test_ipadapter_forward_adds_to_hidden(self):
        """``forward`` returns same shape as ``hidden`` and changes it."""
        cfg = IPAdapterConfig(
            image_embed_dim=64, cross_attention_dim=32,
            num_tokens=2, num_layers=2,
        )
        adapter = IPAdapter(cfg).eval()
        hidden = torch.randn(1, 8, cfg.cross_attention_dim)
        image_embeds = torch.randn(1, cfg.image_embed_dim)
        with torch.no_grad():
            out = adapter(hidden, image_embeds, layer_idx=0)
        # Shape contract: same shape as hidden.
        assert out.shape == hidden.shape
        # The injected projection must change the first num_tokens
        # positions, so the result is not bit-equal to the input.
        assert not torch.equal(hidden, out)

    def test_ipadapter_apply_smoke(self):
        """``adapter.apply`` returns ``(B, L, cross_attention_dim)``."""
        cfg = IPAdapterConfig(
            image_embed_dim=64, cross_attention_dim=32,
            num_tokens=2, num_layers=2,
        )
        adapter = IPAdapter(cfg).eval()
        # The mock encoder takes a pre-pooled embedding of size
        # ``image_embed_dim``; the conditioning is the host
        # cross-attention's hidden states.
        conditioning = torch.randn(1, 8, cfg.cross_attention_dim)
        image = torch.randn(1, cfg.image_embed_dim)
        with torch.no_grad():
            out = adapter.apply(conditioning, image, scale=0.5)
        assert out.shape == (1, 8, cfg.cross_attention_dim)

    def test_ipadapter_key_map_count(self):
        """:data:`IP_ADAPTER_KEY_MAP` has at least 8 entries."""
        assert len(IP_ADAPTER_KEY_MAP) >= 8
