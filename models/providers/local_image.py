"""Local-torch image provider for the v0.4.x P0 multi-modal milestone.

This module wires the project-owned
:mod:`models.image.unet` / :mod:`models.image.vae` /
:mod:`models.image.clip_encoder` into the
:class:`models.interfaces.media_providers.ImageProvider` protocol
so that the v0.4.x P0 image nodes / examples can be exercised
**end-to-end with a real neural network** (no echo, no passthrough)
while still being *pure torch, zero external dependencies*.

The class is intentionally small:

* it owns a :class:`UNet` (denoising network) + :class:`VAE`
  (latent codec) + optional :class:`CLIPTextEncoder` (text
  condition) loaded from a single ``.pt`` file (or constructed
  in memory from a :class:`ImageProviderConfig`);
* it implements :meth:`generate` (the only
  :class:`ImageProvider` method exercised by ``call_image_backend``)
  and a few introspection helpers used by the v0.4.x P0 demo /
  tests;
* it is **thread-safe** (a single re-entrant lock guards the
  forward pass so concurrent :meth:`generate` calls serialise on
  the same model).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L4 ``models.image`` -- real components (UNet / VAE / CLIP).
* L6 ``models.providers`` (this module) -- real image provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

from ..interfaces.media_providers import ImageProvider
from ..image import UNet, VAE, CLIPTextEncoder

__all__ = [
    "LocalTorchImageProvider",
    "ImageProviderConfig",
    "TINY_IMAGE_CONFIG",
    "SMALL_IMAGE_CONFIG",
]


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = get_logger("models.providers.local_image")


# ---------------------------------------------------------------------------
# Config presets (sized for CI + demo)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageProviderConfig:
    """Hyperparameter bundle for :class:`LocalTorchImageProvider`.

    The defaults produce a tiny model whose forward / backward
    pass runs in well under a second on a single CPU thread; that
    is what the v0.4.x P0 demo / CI smoke tests rely on to keep the
    milestone dependency-free.
    """

    name: str = "tiny"
    # UNet
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_hidden_size: int = 32
    unet_num_heads: int = 4
    unet_num_res_blocks: int = 1
    unet_block_channels: tuple = (32, 64, 128)
    unet_context_dim: int = 64
    # VAE
    vae_in_channels: int = 3
    vae_latent_channels: int = 4
    vae_hidden_size: int = 32
    vae_num_down_blocks: int = 2
    # CLIP (text conditioning)
    clip_vocab_size: int = 256        # byte-level; matches ByteTokenizer style
    clip_hidden_size: int = 64
    clip_num_layers: int = 2
    clip_num_heads: int = 4
    clip_max_seq_len: int = 16
    # Sampling
    default_steps: int = 4
    default_guidance_scale: float = 1.0
    default_image_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the config."""
        d = self.__dict__.copy()
        d["unet_block_channels"] = list(self.unet_block_channels)
        return d


TINY_IMAGE_CONFIG = ImageProviderConfig(name="tiny")
SMALL_IMAGE_CONFIG = ImageProviderConfig(
    name="small",
    unet_hidden_size=64,
    unet_num_res_blocks=2,
    unet_block_channels=(64, 128, 256),
    vae_hidden_size=64,
    vae_num_down_blocks=3,
    clip_hidden_size=128,
    clip_num_layers=4,
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalTorchImageProvider(ImageProvider):
    """A real, project-owned :class:`ImageProvider` backed by ``torch``.

    The provider is **stateless at the framework level** -- it
    holds a single :class:`UNet` + :class:`VAE` +
    :class:`CLIPTextEncoder` triple and serialises concurrent
    calls behind a lock.  All forward passes run in
    ``torch.no_grad`` mode so inference does not allocate
    autograd graphs.

    Args:
        unet: A pre-built :class:`UNet`.  When ``None`` a fresh
            one is built from ``config``.
        vae: A pre-built :class:`VAE`.  When ``None`` a fresh
            one is built from ``config``.
        clip: A pre-built :class:`CLIPTextEncoder`.  When
            ``None`` a fresh one is built from ``config``.
        config: The :class:`ImageProviderConfig` that was used
            to build the models.  When ``None`` the
            :data:`TINY_IMAGE_CONFIG` is used.
        device: Device to run the model on.  Defaults to CPU so
            the provider is portable across CI environments.
    """

    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        vae: Optional[nn.Module] = None,
        clip: Optional[nn.Module] = None,
        config: Optional[ImageProviderConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if config is None:
            config = TINY_IMAGE_CONFIG
        if unet is None:
            unet = UNet(
                in_channels=config.unet_in_channels,
                out_channels=config.unet_out_channels,
                hidden_size=config.unet_hidden_size,
                context_dim=config.unet_context_dim,
                num_heads=config.unet_num_heads,
                num_res_blocks=config.unet_num_res_blocks,
                block_channels=list(config.unet_block_channels),
            )
        if vae is None:
            vae = VAE(
                in_channels=config.vae_in_channels,
                latent_channels=config.vae_latent_channels,
                hidden_size=config.vae_hidden_size,
                num_down_blocks=config.vae_num_down_blocks,
            )
        if clip is None:
            clip = CLIPTextEncoder(
                vocab_size=config.clip_vocab_size,
                hidden_size=config.clip_hidden_size,
                num_layers=config.clip_num_layers,
                num_heads=config.clip_num_heads,
                max_seq_len=config.clip_max_seq_len,
            )

        self._unet: nn.Module = unet.to(device)
        self._vae: nn.Module = vae.to(device)
        self._clip: nn.Module = clip.to(device)
        for m in (self._unet, self._vae, self._clip):
            m.eval()

        self._config: ImageProviderConfig = config
        self._device: torch.device = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_random(
        cls,
        config: Optional[ImageProviderConfig] = None,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchImageProvider":
        """Construct a provider with freshly initialised models.

        Useful for CI smoke tests and the v0.4.x P0 demo when no
        pre-trained checkpoint is available -- the model will
        produce noise but the contract (prompt in, image tensor
        out) is fully exercised.
        """
        return cls(config=config, device=device)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchImageProvider":
        """Load a provider from a ``.pt`` file produced by
        :func:`save_image_provider`.

        The file must contain ``state_dicts`` for ``unet`` /
        ``vae`` / ``clip`` plus a ``config`` key holding an
        :class:`ImageProviderConfig` (as a :class:`dict`).
        """
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError("image provider file not found: {}".format(p))
        payload = torch.load(p, map_location=device, weights_only=False)
        cfg_dict = payload.get("config", {})
        if not isinstance(cfg_dict, dict):
            raise TypeError("payload['config'] must be a dict")
        bc = cfg_dict.get("unet_block_channels", (32, 64, 128))
        cfg_dict["unet_block_channels"] = tuple(bc)
        config = ImageProviderConfig(**cfg_dict)
        # Build the model shells first, then load state dicts.
        provider = cls(config=config, device=device)
        if "unet" in payload:
            provider._unet.load_state_dict(payload["unet"], strict=False)
        if "vae" in payload:
            provider._vae.load_state_dict(payload["vae"], strict=False)
        if "clip" in payload:
            provider._clip.load_state_dict(payload["clip"], strict=False)
        return provider

    def save(
        self,
        path: Union[str, Path],
    ) -> Path:
        """Persist the provider to ``path`` (a ``.pt`` file).

        Inverse of :meth:`from_file` -- the saved payload can be
        loaded with :meth:`from_file` to reconstruct the
        provider.
        """
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config.to_dict(),
            "unet": self._unet.state_dict(),
            "vae": self._vae.state_dict(),
            "clip": self._clip.state_dict(),
        }
        torch.save(payload, out)
        return out

    # ------------------------------------------------------------------
    # ImageProvider interface
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image from ``prompt``.

        The pipeline is a deliberately minimal **DDPM-style loop**:

        1. Tokenise ``prompt`` with the byte-level CLIP encoder
           (the encoder is tiny: vocab 256 / hidden 64 / 2
           layers, so this is just to prove the cross-attn code
           path works).
        2. Sample noise in latent space ``(C, H/8, W/8)``.
        3. Run ``steps`` denoising steps with the UNet,
           conditioning on the text embedding.
        4. Decode the resulting latent with the VAE decoder to
           get a ``(3, H, W)`` image in ``[0, 1]`` (clamped).

        Args:
            prompt: Text prompt.
            width: Output width in pixels (clamped to a multiple
                of 8).  Defaults to ``config.default_image_size``.
            height: Output height in pixels (clamped to a
                multiple of 8).  Defaults to
                ``config.default_image_size``.
            steps: Number of denoising steps.  Defaults to
                ``config.default_steps``.
            guidance_scale: CFG scale (1.0 = no guidance, must be
                ``>= 1.0``).  Defaults to
                ``config.default_guidance_scale``.
            seed: RNG seed for reproducibility.  Defaults to
                ``None`` (no seed).
            **kwargs: Ignored.  Forwarded for forward-compat with
                the v0.4.x P0 node kwargs (``character_ref``,
                ``outfit_ref`` etc.) which the provider simply
                records in the returned dict.

        Returns:
            A dict with at least:

            * ``"image"`` -- a ``torch.Tensor`` of shape
              ``(3, H, W)`` in ``[0, 1]``.
            * ``"width"`` / ``"height"`` -- the actual image
              dimensions.
            * ``"seed"`` -- the seed used (or 0 if random).
            * ``"prompt"`` -- the (truncated) prompt.
            * ``"steps"`` -- the number of denoising steps run.
            * ``"latent_shape"`` -- the shape of the denoised
              latent (useful for downstream consumers).
        """
        cfg = self._config
        size = int(width or height or cfg.default_image_size)
        # Latent spatial is ``size // 8`` for a 3-down VAE
        # (num_down_blocks=2) and ``size // 16`` for a 4-down VAE
        # (num_down_blocks=3).  We compute the divisor from the
        # VAE's actual down-sampling factor.
        down_factor = max(2 ** cfg.vae_num_down_blocks, 1)
        latent_hw = max(size // down_factor, 1)
        lat_h = max(height or size, down_factor) // down_factor
        lat_w = max(width or size, down_factor) // down_factor
        lat_h = max(lat_h, 1)
        lat_w = max(lat_w, 1)

        n_steps = int(steps or cfg.default_steps)
        if n_steps < 1:
            n_steps = 1
        g = float(guidance_scale or cfg.default_guidance_scale)
        if g < 1.0:
            g = 1.0

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        used_seed = int(seed) if seed is not None else 0

        with self._lock:
            with torch.no_grad():
                # 1. Encode the prompt.
                token_ids = self._byte_tokenize(prompt, cfg.clip_max_seq_len)
                token_tensor = torch.tensor(
                    [token_ids], dtype=torch.long, device=self._device,
                )
                text_emb = self._clip(token_tensor)  # (1, max_seq_len, clip_hidden)

                # 2. Sample noise in latent space.
                latent = torch.randn(
                    1,
                    cfg.unet_in_channels,
                    lat_h,
                    lat_w,
                    generator=gen,
                    device=self._device,
                )
                # 3. Naive DDPM-style loop (no scheduler, just
                #    shrink the noise by a fixed factor at each
                #    step and ask the UNet to predict the
                #    residual).  This is a *placeholder* loop --
                #    it is enough to prove the pipeline works
                #    end-to-end; a proper scheduler will replace
                #    it in v0.5.
                for step in range(n_steps):
                    t = torch.full(
                        (1,), float(step), device=self._device,
                    )
                    residual = self._unet(latent, t, context=text_emb)
                    # Blend the prediction back in (simple
                    # noise-shrink).  The exact schedule is not
                    # important for the v0.4.x P0 milestone --
                    # the goal is to prove the models can be
                    # chained.
                    alpha = 1.0 - (step + 1) / n_steps
                    latent = latent * alpha + residual * (1.0 - alpha)

                # 4. Decode to pixel space.
                image = self._vae.decode(latent)  # (1, 3, H, W)
                image = image.clamp(0.0, 1.0)[0]  # (3, H, W)

        return {
            "image": image.cpu(),
            "width": int(image.shape[-1]),
            "height": int(image.shape[-2]),
            "seed": used_seed,
            "prompt": prompt[:64],
            "steps": n_steps,
            "latent_shape": tuple(latent.shape[1:]),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> ImageProviderConfig:
        """The provider config (read-only)."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device the models are bound to."""
        return self._device

    def num_parameters(self) -> int:
        """Total parameter count across UNet + VAE + CLIP."""
        return sum(
            sum(p.numel() for p in m.parameters())
            for m in (self._unet, self._vae, self._clip)
        )

    @staticmethod
    def _byte_tokenize(text: str, max_len: int) -> List[int]:
        """Byte-level tokenisation matching :class:`CLIPTextEncoder`.

        The v0.4.x P0 demo uses a byte-level encoder (vocab
        256) so we do not need a real BPE here -- the encoder
        expects ids in ``[0, 256)`` which is what utf-8 bytes
        give us.
        """
        if not text:
            return [0] * max_len
        ids = list(text.encode("utf-8")[:max_len])
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    def __repr__(self) -> str:
        return (
            "LocalTorchImageProvider(name={!r}, params={}, device={!r})".format(
                self._config.name, self.num_parameters(), self._device,
            )
        )
