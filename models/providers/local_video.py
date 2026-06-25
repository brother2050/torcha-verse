"""Local-torch video provider for the v0.4.x P0 multi-modal milestone.

This module wires the project-owned
:mod:`models.video.video_dit` (spatiotemporal denoising),
:mod:`models.video.video_vae` (3D latent codec) and
:mod:`models.video.motion_module` (temporal attention) into the
:class:`models.interfaces.media_providers.VideoProvider` protocol
so that the v0.4.x P0 video nodes / examples can be exercised
**end-to-end with a real neural network** (no echo, no passthrough)
while still being *pure torch, zero external dependencies*.

The class is intentionally small:

* it owns a :class:`VideoDiT` + :class:`VideoVAE` pair loaded
  from a single ``.pt`` file (or constructed in memory from a
  :class:`VideoProviderConfig`);
* it implements :meth:`generate` (the only
  :class:`VideoProvider` method exercised by ``call_video_backend``)
  and a few introspection helpers used by the v0.4.x P0 demo /
  tests;
* it is **thread-safe** (a single re-entrant lock guards the
  forward pass so concurrent :meth:`generate` calls serialise on
  the same model).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L4 ``models.video`` -- real components (VideoDiT / VideoVAE /
  MotionModule).
* L6 ``models.providers`` (this module) -- real video provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

from ..interfaces.media_providers import VideoProvider
from ..video import VideoDiT, VideoVAE

__all__ = [
    "LocalTorchVideoProvider",
    "VideoProviderConfig",
    "TINY_VIDEO_CONFIG",
    "SMALL_VIDEO_CONFIG",
]


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = get_logger("models.providers.local_video")


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoProviderConfig:
    """Hyperparameter bundle for :class:`LocalTorchVideoProvider`.

    Defaults produce a tiny model whose forward / backward pass
    runs in well under a second on a single CPU thread; that is
    what the v0.4.x P0 demo / CI smoke tests rely on to keep the
    milestone dependency-free.
    """

    name: str = "tiny"
    # VideoDiT
    dit_in_channels: int = 4
    dit_hidden_size: int = 64
    dit_num_layers: int = 2
    dit_num_heads: int = 4
    dit_patch_size: Tuple[int, int, int] = (1, 2, 2)
    dit_num_frames: int = 4
    dit_context_dim: int = 64
    # VideoVAE
    vae_in_channels: int = 3
    vae_latent_channels: int = 4
    vae_hidden_size: int = 32
    vae_num_down_blocks: int = 2
    vae_temporal_stride: int = 1
    # Sampling
    default_steps: int = 2
    default_fps: int = 8
    default_num_frames: int = 4
    default_height: int = 16
    default_width: int = 16

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["dit_patch_size"] = list(self.dit_patch_size)
        return d


TINY_VIDEO_CONFIG = VideoProviderConfig(name="tiny")
SMALL_VIDEO_CONFIG = VideoProviderConfig(
    name="small",
    dit_hidden_size=128,
    dit_num_layers=4,
    dit_num_heads=8,
    vae_hidden_size=64,
    vae_num_down_blocks=3,
    default_num_frames=8,
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalTorchVideoProvider(VideoProvider):
    """A real, project-owned :class:`VideoProvider` backed by ``torch``.

    The provider is **stateless at the framework level** -- it
    holds a single :class:`VideoDiT` + :class:`VideoVAE` pair
    and serialises concurrent calls behind a lock.  All forward
    passes run in ``torch.no_grad`` mode so inference does not
    allocate autograd graphs.

    Args:
        dit: A pre-built :class:`VideoDiT`.  When ``None`` a
            fresh one is built from ``config``.
        vae: A pre-built :class:`VideoVAE`.  When ``None`` a
            fresh one is built from ``config``.
        config: The :class:`VideoProviderConfig` that was used
            to build the models.  When ``None`` the
            :data:`TINY_VIDEO_CONFIG` is used.
        device: Device to run the model on.  Defaults to CPU so
            the provider is portable across CI environments.
    """

    def __init__(
        self,
        dit: Optional[nn.Module] = None,
        vae: Optional[nn.Module] = None,
        config: Optional[VideoProviderConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if config is None:
            config = TINY_VIDEO_CONFIG
        if dit is None:
            dit = VideoDiT(
                in_channels=config.dit_in_channels,
                latent_channels=config.dit_in_channels,
                hidden_size=config.dit_hidden_size,
                num_layers=config.dit_num_layers,
                num_heads=config.dit_num_heads,
                patch_size=config.dit_patch_size,
                num_frames=config.dit_num_frames,
                context_dim=config.dit_context_dim,
            )
        if vae is None:
            vae = VideoVAE(
                in_channels=config.vae_in_channels,
                latent_channels=config.vae_latent_channels,
                hidden_size=config.vae_hidden_size,
                num_down_blocks=config.vae_num_down_blocks,
                temporal_stride=config.vae_temporal_stride,
            )

        self._dit: nn.Module = dit.to(device)
        self._vae: nn.Module = vae.to(device)
        for m in (self._dit, self._vae):
            m.eval()

        self._config: VideoProviderConfig = config
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
        config: Optional[VideoProviderConfig] = None,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchVideoProvider":
        """Construct a provider with freshly initialised models."""
        return cls(config=config, device=device)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchVideoProvider":
        """Load a provider from a ``.pt`` file produced by :meth:`save`."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError("video provider file not found: {}".format(p))
        payload = torch.load(p, map_location=device, weights_only=False)
        cfg_dict = payload.get("config", {})
        if not isinstance(cfg_dict, dict):
            raise TypeError("payload['config'] must be a dict")
        if "dit_patch_size" in cfg_dict:
            cfg_dict["dit_patch_size"] = tuple(cfg_dict["dit_patch_size"])
        config = VideoProviderConfig(**cfg_dict)
        provider = cls(config=config, device=device)
        if "dit" in payload:
            provider._dit.load_state_dict(payload["dit"], strict=False)
        if "vae" in payload:
            provider._vae.load_state_dict(payload["vae"], strict=False)
        return provider

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the provider to ``path`` (a ``.pt`` file)."""
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config.to_dict(),
            "dit": self._dit.state_dict(),
            "vae": self._vae.state_dict(),
        }
        torch.save(payload, out)
        return out

    # ------------------------------------------------------------------
    # VideoProvider interface
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a video from ``prompt``.

        The pipeline is a deliberately minimal 3D diffusion loop:

        1. Sample noise in the spatiotemporal latent space
           ``(1, C, T, H/8, W/8)``.
        2. Run ``steps`` denoising steps with the :class:`VideoDiT`
           (conditioned on a fixed zero context vector -- the
           text encoder is omitted here to keep the provider
           dependency-free; a real CLIP context will replace the
           zeros in v0.5).
        3. Decode the latent to pixel space with the
           :class:`VideoVAE` decoder, producing a
           ``(1, T, 3, H, W)`` video tensor clamped to
           ``[0, 1]``.

        Args:
            prompt: Text prompt (recorded in the returned dict but
                not used to condition the model in v0.4.x P0).
            num_frames: Number of frames to generate.  Defaults to
                ``config.default_num_frames``.
            height: Output height.  Defaults to
                ``config.default_height``.
            width: Output width.  Defaults to
                ``config.default_width``.
            fps: Frames per second (metadata only).  Defaults to
                ``config.default_fps``.
            steps: Number of denoising steps.  Defaults to
                ``config.default_steps``.
            seed: RNG seed for reproducibility.  Defaults to
                ``None``.
            **kwargs: Ignored.  Forwarded for forward-compat with
                the v0.4.x P0 node kwargs (``character_ref``,
                ``motion_seed`` ...).

        Returns:
            A dict with at least:

            * ``"frames"`` -- a ``torch.Tensor`` of shape
              ``(T, 3, H, W)`` in ``[0, 1]``.
            * ``"num_frames"`` -- the actual number of frames.
            * ``"fps"`` -- the frame rate.
            * ``"width"`` / ``"height"`` -- the actual frame
              dimensions.
            * ``"seed"`` -- the seed used (or 0 if random).
            * ``"prompt"`` -- the (truncated) prompt.
        """
        cfg = self._config
        t = int(num_frames or cfg.default_num_frames)
        h = int(height or cfg.default_height)
        w = int(width or cfg.default_width)
        # Make spatial dims multiples of the VAE down factor.
        down_factor = max(2 ** cfg.vae_num_down_blocks, 1)
        # Also align to the DiT patch spatial size.
        patch_h, patch_w = cfg.dit_patch_size[1], cfg.dit_patch_size[2]
        align = max(down_factor * patch_h, down_factor)
        if h % align != 0:
            h = max(align, (h // align) * align)
        if w % align != 0:
            w = max(align, (w // align) * align)
        # Latent time axis is T // temporal_stride.
        t_lat = max(t // max(cfg.vae_temporal_stride, 1), 1)
        h_lat = max(h // down_factor, 1)
        w_lat = max(w // down_factor, 1)

        n_steps = max(int(steps or cfg.default_steps), 1)
        fr = int(fps or cfg.default_fps)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        used_seed = int(seed) if seed is not None else 0

        with self._lock:
            with torch.no_grad():
                # 1. noise
                latent = torch.randn(
                    1,
                    cfg.dit_in_channels,
                    t_lat,
                    h_lat,
                    w_lat,
                    generator=gen,
                    device=self._device,
                )
                # 2. naive denoise loop (no scheduler, just blend)
                # Context is a zero vector (cross-attn receives it
                # but we do not run a text encoder here to keep the
                # provider dependency-free).  DiT accepts
                # ``context=None`` for unconditional generation.
                for step in range(n_steps):
                    t_step = torch.full(
                        (1,), float(step), device=self._device,
                    )
                    residual = self._dit(
                        latent, t_step, context=None,
                    )
                    alpha = 1.0 - (step + 1) / n_steps
                    latent = latent * alpha + residual * (1.0 - alpha)
                # 3. decode
                video = self._vae.decode(latent)  # (1, 3, T, H, W)
                video = video.clamp(0.0, 1.0)
                # Rearrange to (T, 3, H, W).
                video = video[0].transpose(0, 1).contiguous()

        actual_t = int(video.shape[0])
        actual_h = int(video.shape[-2])
        actual_w = int(video.shape[-1])
        return {
            "frames": video.cpu(),
            "num_frames": actual_t,
            "fps": fr,
            "width": actual_w,
            "height": actual_h,
            "seed": used_seed,
            "prompt": prompt[:64],
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> VideoProviderConfig:
        """The provider config (read-only)."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device the models are bound to."""
        return self._device

    def num_parameters(self) -> int:
        """Total parameter count across VideoDiT + VideoVAE."""
        return sum(
            sum(p.numel() for p in m.parameters())
            for m in (self._dit, self._vae)
        )

    def __repr__(self) -> str:
        return (
            "LocalTorchVideoProvider(name={!r}, params={}, device={!r})".format(
                self._config.name, self.num_parameters(), self._device,
            )
        )
