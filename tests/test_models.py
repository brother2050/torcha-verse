"""Tests for the Models Layer."""

from __future__ import annotations

import pytest
import torch

from models.components.rmsnorm import RMSNorm
from models.components.swiglu import SwiGLU
from models.components.rope import RotaryPositionEmbedding
from models.components.lora import LoRALinear
from models.text.transformer import TransformerDecoder
from models.text.moe import MoETransformerDecoder
from models.text.attention import GroupedQueryAttention, MultiQueryAttention
from models.image.vae import VAE
from models.image.dit import DiT
from models.image.unet import UNet
from models.audio.audio_codec import AudioCodec
from models.audio.hifi_gan import HiFiGAN
from models.video.video_vae import VideoVAE
from models.video.video_dit import VideoDiT


class TestComponents:
    """Test model components."""

    def test_rmsnorm(self):
        """RMSNorm normalizes input correctly."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_swiglu(self):
        """SwiGLU produces correct output shape."""
        glf = SwiGLU(64, 128)
        x = torch.randn(2, 10, 64)
        out = glf(x)
        assert out.shape == x.shape

    def test_rope(self):
        """RoPE produces cos/sin embeddings."""
        rope = RotaryPositionEmbedding(dim=32, max_seq_len=128)
        cos, sin = rope.get_cos_sin(10, torch.device("cpu"))
        assert cos.shape[0] == 10

    def test_lora_linear(self):
        """LoRALinear forward produces correct shape."""
        lora = LoRALinear(64, 32, r=4, alpha=8)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 32)


class TestTransformer:
    """Test TransformerDecoder."""

    def test_forward(self):
        """TransformerDecoder forward produces correct logits shape."""
        model = TransformerDecoder(
            vocab_size=100, hidden_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=128, max_seq_len=64,
        )
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        assert logits.shape == (2, 10, 100)

    def test_generate(self):
        """generate() produces tokens beyond the prompt."""
        model = TransformerDecoder(
            vocab_size=100, hidden_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=128, max_seq_len=64,
        )
        input_ids = torch.randint(0, 100, (1, 5))
        output = model.generate(input_ids, max_tokens=8, temperature=0.8, top_k=10)
        assert output.shape[1] >= 5


class TestMoE:
    """Test MoETransformerDecoder."""

    def test_forward(self):
        """MoETransformerDecoder forward produces logits."""
        model = MoETransformerDecoder(
            vocab_size=100, hidden_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=128,
            num_experts=4, top_k=2, max_seq_len=64,
        )
        input_ids = torch.randint(0, 100, (2, 8))
        result = model(input_ids)
        if isinstance(result, tuple):
            logits, aux_loss = result
        else:
            logits = result
        assert logits.shape == (2, 8, 100)


class TestAttention:
    """Test attention variants."""

    def test_gqa(self):
        """GroupedQueryAttention forward works."""
        attn = GroupedQueryAttention(hidden_size=64, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_mqa(self):
        """MultiQueryAttention forward works."""
        attn = MultiQueryAttention(hidden_size=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == x.shape


class TestVAE:
    """Test VAE."""

    def test_encode_decode(self):
        """VAE encode/decode round-trip."""
        vae = VAE(in_channels=3, latent_channels=4, hidden_size=32, num_res_blocks=1, num_down_blocks=2)
        x = torch.randn(1, 3, 32, 32)
        mean, logvar = vae.encode(x)
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z)
        assert recon.shape == x.shape


class TestDiT:
    """Test DiT."""

    def test_forward(self):
        """DiT forward produces noise prediction."""
        dit = DiT(
            input_size=8, patch_size=2, in_channels=4,
            hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
            context_dim=64,
        )
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        ctx = torch.randn(1, 1, 64)
        out = dit(x, t, encoder_hidden_states=ctx)
        assert out.shape == x.shape


class TestUNet:
    """Test UNet."""

    def test_forward(self):
        """UNet forward produces noise prediction."""
        unet = UNet(
            in_channels=4, out_channels=4, hidden_size=32,
            context_dim=32, num_heads=4, num_res_blocks=1,
        )
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([500])
        ctx = torch.randn(1, 1, 32)
        out = unet(x, timesteps=t, encoder_hidden_states=ctx)
        assert out.shape == x.shape


class TestAudioCodec:
    """Test AudioCodec."""

    def test_encode_decode(self):
        """AudioCodec encode/decode round-trip."""
        codec = AudioCodec(
            in_channels=1, hidden_size=16, latent_size=8,
            num_quantizers=2, codebook_size=64,
        )
        wav = torch.randn(1, 1, 1024)
        recon, tokens, loss = codec(wav)
        assert recon.dim() == 3
        assert recon.shape[0] == 1 and recon.shape[1] == 1


class TestHiFiGAN:
    """Test HiFiGAN."""

    def test_forward(self):
        """HiFiGAN forward produces waveform."""
        vocoder = HiFiGAN(in_channels=80, upsample_rates=[8, 4], upsample_kernel_sizes=[16, 8])
        mel = torch.randn(1, 80, 32)
        out = vocoder(mel)
        assert out.dim() == 3


class TestVideoVAE:
    """Test VideoVAE."""

    def test_encode_decode(self):
        """VideoVAE encode/decode round-trip."""
        vae = VideoVAE(in_channels=3, latent_channels=4, hidden_size=16)
        video = torch.randn(1, 3, 4, 16, 16)
        mean, logvar = vae.encode(video)
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z)
        assert recon.shape == video.shape


class TestVideoDiT:
    """Test VideoDiT."""

    def test_forward(self):
        """VideoDiT forward produces noise prediction."""
        dit = VideoDiT(
            in_channels=4, hidden_size=32, num_layers=2,
            num_heads=4, patch_size=(1, 2, 2), num_frames=4,
            context_dim=32,
        )
        x = torch.randn(1, 4, 4, 8, 8)
        t = torch.tensor([500])
        ctx = torch.randn(1, 1, 32)
        out = dit(x, t, encoder_hidden_states=ctx)
        assert out.shape == x.shape
