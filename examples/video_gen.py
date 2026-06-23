"""Video generation example.

Demonstrates VideoVAE encode/decode and VideoDiT noise prediction.

Run with::

    python examples/video_gen.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from models.video.video_vae import VideoVAE
from models.video.video_dit import VideoDiT


def save_video_frames(video: torch.Tensor, output_dir: str) -> None:
    """Save video frames as individual images."""
    os.makedirs(output_dir, exist_ok=True)
    frames = video[0].cpu()  # (C, T, H, W)
    for t in range(frames.shape[1]):
        frame = frames[:, t].clamp(0, 1).numpy()
        frame = (frame * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(frame).save(os.path.join(output_dir, f"frame_{t:03d}.png"))
    print(f"    Saved {frames.shape[1]} frames to {output_dir}")


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — Video Generation Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. VideoVAE encode/decode ---
    print("\n[1] VideoVAE encode/decode...")
    vae = VideoVAE(
        in_channels=3,
        latent_channels=4,
        hidden_size=32,
    ).to(device).eval()
    print(f"    VAE params: {sum(p.numel() for p in vae.parameters()):,}")

    # Dummy video: (batch, channels, frames, height, width)
    video = torch.randn(1, 3, 8, 32, 32, device=device) * 0.1
    with torch.no_grad():
        mean, logvar = vae.encode(video)
        latent = vae.reparameterize(mean, logvar)
        recon = vae.decode(latent)
    print(f"    Input:  {video.shape}")
    print(f"    Latent: {latent.shape}")
    print(f"    Recon:  {recon.shape}")

    # --- 2. VideoDiT noise prediction ---
    print("\n[2] VideoDiT noise prediction...")
    dit = VideoDiT(
        in_channels=4,
        latent_channels=4,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        patch_size=(2, 2, 2),
    ).to(device).eval()
    print(f"    DiT params: {sum(p.numel() for p in dit.parameters()):,}")

    noisy_latent = torch.randn(1, 4, 4, 16, 16, device=device)
    timesteps = torch.tensor([500], device=device)
    text_emb = torch.randn(1, 1, 64, device=device)

    with torch.no_grad():
        noise_pred = dit(noisy_latent, timesteps, encoder_hidden_states=text_emb)
    print(f"    Noisy latent: {noisy_latent.shape}")
    print(f"    Noise pred:   {noise_pred.shape}")

    # --- 3. Decode to video ---
    print("\n[3] Decoding latent to video...")
    with torch.no_grad():
        video_out = vae.decode(latent)
    print(f"    Video shape: {video_out.shape}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs", "video_frames")
    save_video_frames(video_out, output_dir)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
