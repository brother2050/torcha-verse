"""Audio synthesis example.

Demonstrates AudioCodec encode/decode and HiFi-GAN vocoder.

Run with::

    python examples/audio_tts.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.audio.audio_codec import AudioCodec
from models.audio.hifi_gan import HiFiGAN


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 22050) -> None:
    """Save a waveform to a WAV file."""
    try:
        import soundfile as sf
        audio = waveform.squeeze().cpu().numpy()
        sf.write(path, audio, sample_rate)
        print(f"    Saved audio to {path}")
    except ImportError:
        print(f"    (soundfile not available, skipping save to {path})")


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — Audio Synthesis Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 22050

    # --- 1. AudioCodec encode/decode ---
    print("\n[1] AudioCodec encode/decode...")
    codec = AudioCodec(
        in_channels=1,
        hidden_size=32,
        latent_size=16,
        num_quantizers=3,
        codebook_size=256,
    ).to(device).eval()
    print(f"    Codec params: {sum(p.numel() for p in codec.parameters()):,}")

    # Create a dummy 1-second audio waveform.
    duration = 1.0
    num_samples = int(sample_rate * duration)
    waveform = torch.randn(1, 1, num_samples, device=device) * 0.1

    with torch.no_grad():
        recon, tokens, commit_loss = codec(waveform)
    print(f"    Input:   {waveform.shape}")
    print(f"    Tokens:  {tokens.shape}")
    print(f"    Recon:   {recon.shape}")
    print(f"    Commit loss: {commit_loss.item():.4f}")

    # --- 2. HiFi-GAN vocoder ---
    print("\n[2] HiFi-GAN vocoder...")
    vocoder = HiFiGAN(
        in_channels=80,
        upsample_rates=[8, 4, 2],
        upsample_kernel_sizes=[16, 8, 4],
    ).to(device).eval()
    print(f"    Vocoder params: {sum(p.numel() for p in vocoder.parameters()):,}")

    # Create a dummy mel-spectrogram.
    n_mels = 80
    hop_length = 256
    mel_length = num_samples // hop_length
    mel = torch.randn(1, n_mels, mel_length, device=device)

    with torch.no_grad():
        audio_out = vocoder(mel)
    print(f"    Mel input:  {mel.shape}")
    print(f"    Audio out:  {audio_out.shape}")

    # --- 3. Save audio ---
    print("\n[3] Saving audio...")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_audio(audio_out, os.path.join(output_dir, "generated_audio.wav"), sample_rate)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
