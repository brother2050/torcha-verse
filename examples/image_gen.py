"""Image generation example.

Demonstrates VAE encoding/decoding and DiT-based diffusion sampling.

Run with::

    python examples/image_gen.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from models.image.vae import VAE
from models.image.dit import DiT
from core.diffusion_scheduler import DiffusionScheduler


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor as an image."""
    img = tensor.clamp(0, 1).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.squeeze(2)
    Image.fromarray(img).save(path)
    print(f"    Saved image to {path}")


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — Image Generation Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. VAE encode/decode ---
    print("\n[1] VAE encode/decode...")
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        hidden_size=64,
        num_res_blocks=1,
        num_down_blocks=2,
    ).to(device).eval()
    print(f"    VAE params: {sum(p.numel() for p in vae.parameters()):,}")

    dummy_image = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        mean, logvar = vae.encode(dummy_image)
        latent = vae.reparameterize(mean, logvar)
        recon = vae.decode(latent)
    print(f"    Input:  {dummy_image.shape}")
    print(f"    Latent: {latent.shape}")
    print(f"    Recon:  {recon.shape}")

    # --- 2. DiT noise prediction ---
    print("\n[2] DiT noise prediction...")
    dit = DiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        context_dim=128,
    ).to(device).eval()
    print(f"    DiT params: {sum(p.numel() for p in dit.parameters()):,}")

    noisy_latent = torch.randn(1, 4, 8, 8, device=device)
    timesteps = torch.tensor([500], device=device)
    text_emb = torch.randn(1, 1, 128, device=device)

    with torch.no_grad():
        noise_pred = dit(noisy_latent, timesteps, encoder_hidden_states=text_emb)
    print(f"    Noisy latent: {noisy_latent.shape}")
    print(f"    Noise pred:   {noise_pred.shape}")

    # --- 3. Diffusion sampling loop ---
    print("\n[3] Diffusion sampling loop (5 steps)...")
    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        noise_strategy="linear",
        sampler_name="ddim",
        device=device,
    )
    scheduler.set_timesteps(5)
    latents = torch.randn(1, 4, 8, 8, device=device)

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = dit(latents, t.unsqueeze(0), encoder_hidden_states=text_emb)
        latents = scheduler.step(noise_pred, t, latents)
        print(f"    Step {i+1}/5 — t={t.item()}, latent norm={latents.norm().item():.4f}")

    # --- 4. Decode to image ---
    print("\n[4] Decoding latent to image...")
    with torch.no_grad():
        image = vae.decode(latents)
    print(f"    Image shape: {image.shape}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_image(image[0], os.path.join(output_dir, "generated_image.png"))

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
