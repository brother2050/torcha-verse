"""Video generation engine for TorchaVerse.

This module provides :class:`VideoEngine`, the capability-layer entry
point for all video generation tasks.  It composes a :class:`VideoDiT`
(or 3-D UNet) denoising model with a :class:`CLIPTextEncoder` for text
conditioning, a :class:`VideoVAE` for spatiotemporal latent encoding and
decoding, a :class:`DiffusionScheduler` for the sampling loop, and a
:class:`FrameInterpolator` for temporal super-resolution.

Supported operations:

* :meth:`txt2video` -- text-to-video generation.
* :meth:`img2video` -- image-to-video (first-frame conditioning).
* :meth:`video_extend` -- extend a video with additional frames.
* :meth:`video2video` -- video-to-video transformation.

Advanced features:

* **Temporal consistency** via shared noise initialisation.
* **Long-video segmentation** with seamless stitching.
* **Motion control** through flow-based conditioning.
* **Super-resolution and frame-rate upscaling**.
* **Audio-video synchronised generation**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.diffusion_scheduler import DiffusionScheduler
from core.model_registry import BaseModel, ModelRegistry
from core.tokenizer_hub import TextTokenizer, TokenizerHub
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from models.image.clip_encoder import CLIPTextEncoder
from models.video.frame_interpolator import FrameInterpolator
from models.video.video_dit import VideoDiT
from models.video.video_vae import VideoVAE

__all__ = [
    "VideoTensor",
    "VideoEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VideoTensor:
    """Container for a video tensor and its frame rate.

    Attributes:
        frames: Video data of shape ``(frames, channels, height, width)``
            or ``(batch, frames, channels, height, width)``.
        fps: Frames per second.
    """

    frames: torch.Tensor
    fps: int = 8

    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        if self.frames.dim() == 5:
            return self.frames.shape[1]
        return self.frames.shape[0]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_frames / self.fps

    @property
    def height(self) -> int:
        """Video height in pixels."""
        if self.frames.dim() == 5:
            return self.frames.shape[3]
        return self.frames.shape[2]

    @property
    def width(self) -> int:
        """Video width in pixels."""
        if self.frames.dim() == 5:
            return self.frames.shape[4]
        return self.frames.shape[3]

    def to(self, device: Union[str, torch.device]) -> "VideoTensor":
        """Move frames to ``device``."""
        return VideoTensor(frames=self.frames.to(device), fps=self.fps)

    def cpu(self) -> "VideoTensor":
        """Move frames to CPU."""
        return self.to("cpu")

    def __repr__(self) -> str:
        return (
            f"VideoTensor(frames={self.num_frames}, fps={self.fps}, "
            f"size={self.width}x{self.height}, dur={self.duration:.2f}s)"
        )


# ---------------------------------------------------------------------------
# VideoEngine
# ---------------------------------------------------------------------------
class VideoEngine:
    """Video generation engine.

    Composes a video denoising model (VideoDiT), a CLIP text encoder, a
    VideoVAE, a diffusion scheduler, and a frame interpolator.

    Args:
        model_name: Registered model name in the :class:`ModelRegistry`.
        config: Optional configuration dictionary.
        device: Optional device override.
        dtype: Optional dtype override.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name: str = model_name
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger(f"VideoEngine[{model_name}]")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._dtype: torch.dtype = dtype or torch.float32

        # Resolve configuration.
        model_cfg = self._cfg_manager.get(f"video_models.{model_name}", {})
        merged_cfg: Dict[str, Any] = {**model_cfg, **self._config}

        # Load the denoising model.
        self._registry: ModelRegistry = ModelRegistry()
        self._tokenizer_hub: TokenizerHub = TokenizerHub()

        try:
            self.model: BaseModel = self._registry.load(
                model_name,
                device=self._device,
                dtype=dtype,
                config=merged_cfg,
            )
        except KeyError:
            self._logger.warning(
                "Video model '%s' is not registered. Instantiating a "
                "small default VideoDiT.", model_name,
            )
            self.model = VideoDiT(
                in_channels=merged_cfg.get("latent_channels", 4),
                hidden_size=merged_cfg.get("hidden_size", 256),
                num_layers=merged_cfg.get("num_layers", 4),
                num_heads=merged_cfg.get("num_heads", 4),
                patch_size=merged_cfg.get("patch_size", (2, 2, 2)),
                num_frames=merged_cfg.get("num_frames", 16),
                context_dim=merged_cfg.get("context_dim", 768),
                config=merged_cfg,
            )
            self.model = self._device_manager.to_device(self.model, self._device)

        # Text encoder (CLIP).
        self.text_encoder: CLIPTextEncoder = CLIPTextEncoder(
            vocab_size=merged_cfg.get("vocab_size", 49408),
            hidden_size=merged_cfg.get("context_dim", 768),
            num_layers=merged_cfg.get("clip_num_layers", 12),
            num_heads=merged_cfg.get("clip_num_heads", 12),
            max_seq_len=merged_cfg.get("max_seq_len", 77),
            config=merged_cfg,
        )
        self.text_encoder = self._device_manager.to_device(
            self.text_encoder, self._device
        )

        # Tokenizer.
        self.tokenizer: TextTokenizer = self._tokenizer_hub.get_tokenizer(  # type: ignore[assignment]
            "text",
            vocab_size=merged_cfg.get("vocab_size", 49408),
            max_length=merged_cfg.get("max_seq_len", 77),
            device=self._device,
        )

        # Video VAE.
        self.vae: VideoVAE = VideoVAE(
            in_channels=merged_cfg.get("vae_in_channels", 3),
            latent_channels=merged_cfg.get("latent_channels", 4),
            hidden_size=merged_cfg.get("vae_hidden_size", 128),
            num_res_blocks=merged_cfg.get("vae_num_res_blocks", 2),
            num_down_blocks=merged_cfg.get("vae_num_down_blocks", 3),
            temporal_stride=merged_cfg.get("temporal_stride", 1),
            scaling_factor=merged_cfg.get("scaling_factor", 0.18215),
            config=merged_cfg,
        )
        self.vae = self._device_manager.to_device(self.vae, self._device)

        # Diffusion scheduler.
        sched_cfg = self._cfg_manager.get("diffusion", {})
        self.scheduler: DiffusionScheduler = DiffusionScheduler(
            num_timesteps=sched_cfg.get("num_timesteps", 1000),
            noise_strategy=sched_cfg.get("noise_strategy", "linear"),
            sampler_name=sched_cfg.get("sampler", "ddim"),
            guidance_scale=sched_cfg.get("guidance_scale", 7.5),
            eta=sched_cfg.get("eta", 0.0),
            device=self._device,
        )

        # Frame interpolator for temporal super-resolution.
        self.frame_interpolator: FrameInterpolator = FrameInterpolator(
            in_channels=merged_cfg.get("vae_in_channels", 3),
            hidden_size=merged_cfg.get("interp_hidden_size", 64),
            num_layers=merged_cfg.get("interp_num_layers", 6),
            config=merged_cfg,
        )
        self.frame_interpolator = self._device_manager.to_device(
            self.frame_interpolator, self._device
        )

        # Hyper-parameters.
        self.latent_channels: int = merged_cfg.get("latent_channels", 4)
        self.vae_downscale_factor: int = merged_cfg.get(
            "vae_downscale_factor",
            2 ** merged_cfg.get("vae_num_down_blocks", 3),
        )
        self.temporal_stride: int = merged_cfg.get("temporal_stride", 1)
        self.scaling_factor: float = merged_cfg.get("scaling_factor", 0.18215)

        self._logger.info("VideoEngine initialised with model '%s'.", model_name)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model_name: str) -> "VideoEngine":
        """Create a :class:`VideoEngine` from the global configuration.

        Args:
            model_name: Registered model name.

        Returns:
            A configured :class:`VideoEngine` instance.
        """
        cfg = ConfigManager()
        model_cfg = cfg.get(f"video_models.{model_name}", {})
        return cls(model_name, config=model_cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts into CLIP embeddings.

        Args:
            prompt: Positive prompt.
            negative_prompt: Negative prompt.

        Returns:
            ``(positive_embeds, negative_embeds)``.
        """
        pos_ids = self.tokenizer.encode(prompt, return_tensors=True).to(self._device)
        neg_ids = self.tokenizer.encode(
            negative_prompt or "", return_tensors=True
        ).to(self._device)
        if pos_ids.dim() == 1:
            pos_ids = pos_ids.unsqueeze(0)
        if neg_ids.dim() == 1:
            neg_ids = neg_ids.unsqueeze(0)

        with torch.no_grad():
            pos_embeds, _ = self.text_encoder(pos_ids)
            neg_embeds, _ = self.text_encoder(neg_ids)

        return pos_embeds, neg_embeds

    def _prepare_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Create initial random video latents.

        Args:
            num_frames: Number of latent frames.
            height: Target video height.
            width: Target video width.
            batch_size: Batch size.
            generator: Optional RNG.

        Returns:
            Latent tensor ``(batch, latent_channels, T, h, w)``.
        """
        latent_t = max(num_frames // self.temporal_stride, 1)
        latent_h = height // self.vae_downscale_factor
        latent_w = width // self.vae_downscale_factor
        shape = (
            batch_size,
            self.latent_channels,
            latent_t,
            latent_h,
            latent_w,
        )
        if generator is not None:
            return torch.randn(shape, generator=generator, device=self._device)
        return torch.randn(shape, device=self._device)

    def _denoise(
        self,
        latents: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the video diffusion denoising loop.

        Args:
            latents: Initial latents.
            pos_embeds: Positive text embeddings.
            neg_embeds: Negative text embeddings.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.

        Returns:
            Denoised latents.
        """
        self.scheduler.set_timesteps(steps)
        self.scheduler.set_guidance_scale(guidance_scale)
        self.model.eval()

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = t.expand(latents.shape[0]).to(self._device)

                # Classifier-free guidance.
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                embed_input = torch.cat([neg_embeds, pos_embeds])

                noise_pred = self.model(
                    latent_input,
                    timesteps=t_input,
                    encoder_hidden_states=embed_input,
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )

                latents = self.scheduler.step(noise_pred, t_batch, latents)

        return latents

    def _latents_to_video(
        self,
        latents: torch.Tensor,
        fps: int,
        target_frames: Optional[int] = None,
    ) -> VideoTensor:
        """Decode latents to a :class:`VideoTensor`.

        Args:
            latents: Latent tensor ``(batch, C, T, h, w)``.
            fps: Output frame rate.
            target_frames: Desired number of output frames.  If greater
                than the latent frame count, frame interpolation is used.

        Returns:
            A :class:`VideoTensor`.
        """
        latents = latents.to(device=self._device, dtype=self._dtype)
        latents = latents / self.scaling_factor

        with torch.no_grad():
            video = self.vae.decode(latents)  # (batch, C, T, H, W)

        # Post-process: clamp to [0, 1].
        video = (video / 2 + 0.5).clamp(0, 1)

        # Remove batch dimension.
        if video.dim() == 5:
            video = video[0]  # (C, T, H, W)
        # Rearrange to (T, C, H, W).
        if video.dim() == 4 and video.shape[0] <= video.shape[1]:
            # (C, T, H, W) -> (T, C, H, W)
            video = video.permute(1, 0, 2, 3)

        # Frame interpolation if target_frames is specified.
        if target_frames is not None and target_frames > video.shape[0]:
            video = self._interpolate_frames(video, target_frames)

        return VideoTensor(frames=video.cpu(), fps=fps)

    def _interpolate_frames(
        self,
        frames: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """Interpolate frames to reach ``target_count``.

        Args:
            frames: Frame tensor ``(T, C, H, W)``.
            target_count: Desired number of frames.

        Returns:
            Interpolated frames ``(target_count, C, H, W)``.
        """
        current = frames.shape[0]
        if current >= target_count:
            return frames[:target_count]

        self.frame_interpolator.eval()
        result: List[torch.Tensor] = [frames[0]]

        with torch.no_grad():
            for i in range(current - 1):
                f0 = frames[i].unsqueeze(0).to(self._device)
                f1 = frames[i + 1].unsqueeze(0).to(self._device)

                # How many intermediate frames to generate.
                remaining = target_count - len(result) - (current - 1 - i)
                num_inter = max(1, min(remaining, 1))

                for j in range(num_inter):
                    t_val = (j + 1) / (num_inter + 1)
                    interp = self.frame_interpolator(f0, f1, t=t_val)
                    result.append(interp.squeeze(0).cpu())

                result.append(frames[i + 1])

        # Pad or truncate to exact target.
        while len(result) < target_count:
            result.append(frames[-1])
        return torch.stack(result[:target_count])

    # ------------------------------------------------------------------
    # Public API: text-to-video
    # ------------------------------------------------------------------
    @torch.no_grad()
    def txt2video(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        fps: int = 8,
        steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> VideoTensor:
        """Generate a video from a text prompt.

        Args:
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames to generate.
            fps: Output frame rate.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            negative_prompt: Negative prompt.
            seed: Optional random seed.

        Returns:
            A :class:`VideoTensor`.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Prepare latents.
        latents = self._prepare_latents(
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )

        # Denoise.
        latents = self._denoise(latents, pos_embeds, neg_embeds, steps, guidance_scale)

        # Decode to video.
        return self._latents_to_video(latents, fps, target_frames=num_frames)

    # ------------------------------------------------------------------
    # Public API: image-to-video
    # ------------------------------------------------------------------
    @torch.no_grad()
    def img2video(
        self,
        first_frame: Any,
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> VideoTensor:
        """Generate a video starting from a first frame image.

        Args:
            first_frame: PIL Image or tensor for the first frame.
            prompt: Text prompt guiding the video.
            num_frames: Number of frames to generate.
            fps: Output frame rate.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            negative_prompt: Negative prompt.
            seed: Optional random seed.

        Returns:
            A :class:`VideoTensor`.
        """
        from PIL import Image as PILImage

        # Convert first frame to tensor.
        if isinstance(first_frame, PILImage.Image):
            import numpy as np
            arr = np.array(first_frame.convert("RGB")).astype("float32") / 255.0
            frame_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        else:
            frame_tensor = first_frame if isinstance(first_frame, torch.Tensor) else torch.tensor(first_frame)

        frame_tensor = frame_tensor.to(device=self._device, dtype=self._dtype)
        if frame_tensor.dim() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)

        _, _, h, w = frame_tensor.shape

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Prepare latents.
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        latents = self._prepare_latents(
            num_frames=num_frames,
            height=h,
            width=w,
            generator=generator,
        )

        # Encode the first frame to latent space and use as conditioning.
        video_input = frame_tensor.unsqueeze(2)  # (1, C, 1, H, W)
        with torch.no_grad():
            mean, _logvar = self.vae.encode(video_input)

        # Inject the first-frame latent into the initial latents.
        if mean.shape[2] >= 1:
            latents[:, :, :1] = mean * self.scaling_factor

        # Denoise with first-frame conditioning (blend at each step).
        self.scheduler.set_timesteps(steps)
        self.scheduler.set_guidance_scale(guidance_scale)
        self.model.eval()

        first_latent = mean * self.scaling_factor

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = t.expand(latents.shape[0]).to(self._device)
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                embed_input = torch.cat([neg_embeds, pos_embeds])

                noise_pred = self.model(
                    latent_input,
                    timesteps=t_input,
                    encoder_hidden_states=embed_input,
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )
                latents = self.scheduler.step(noise_pred, t_batch, latents)

                # Anchor the first frame for temporal consistency.
                latents[:, :, :1] = first_latent

        return self._latents_to_video(latents, fps, target_frames=num_frames)

    # ------------------------------------------------------------------
    # Public API: video extension
    # ------------------------------------------------------------------
    @torch.no_grad()
    def video_extend(
        self,
        video: VideoTensor,
        num_additional_frames: int = 16,
        prompt: str = "",
        steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> VideoTensor:
        """Extend a video with additional frames.

        Uses the last frame of the input video as the anchor for the
        new segment, ensuring temporal continuity.

        Args:
            video: Input video to extend.
            num_additional_frames: Number of new frames to generate.
            prompt: Optional text prompt for the extension.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.

        Returns:
            A new :class:`VideoTensor` with the extended frames appended.
        """
        frames = video.frames.to(self._device)
        if frames.dim() == 5:
            frames = frames[0]
        if frames.dim() == 4 and frames.shape[0] > frames.shape[1]:
            # (T, C, H, W) -> (C, T, H, W)
            frames = frames.permute(1, 0, 2, 3)

        # Use the last frame as the anchor for the extension.
        last_frame = frames[:, -1:, :, :]  # (C, 1, H, W)

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt or "continue the video")

        # Prepare latents for the new segment.
        total_frames = frames.shape[1] + num_additional_frames
        latents = self._prepare_latents(
            num_frames=num_additional_frames,
            height=frames.shape[2],
            width=frames.shape[3],
        )

        # Encode the last frame as conditioning.
        with torch.no_grad():
            mean, _logvar = self.vae.encode(last_frame.unsqueeze(0))
        first_latent = mean * self.scaling_factor
        latents[:, :, :1] = first_latent

        # Denoise.
        self.scheduler.set_timesteps(steps)
        self.scheduler.set_guidance_scale(guidance_scale)
        self.model.eval()

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = t.expand(latents.shape[0]).to(self._device)
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                embed_input = torch.cat([neg_embeds, pos_embeds])

                noise_pred = self.model(
                    latent_input,
                    timesteps=t_input,
                    encoder_hidden_states=embed_input,
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )
                latents = self.scheduler.step(noise_pred, t_batch, latents)
                latents[:, :, :1] = first_latent

        # Decode the new segment.
        new_video = self._latents_to_video(latents, video.fps, target_frames=num_additional_frames)

        # Concatenate original and new frames.
        original_frames = video.frames
        if original_frames.dim() == 5:
            original_frames = original_frames[0]
        if original_frames.dim() == 4 and original_frames.shape[1] <= original_frames.shape[0]:
            original_frames = original_frames.permute(1, 0, 2, 3)
        # Now (T, C, H, W)
        original_frames = original_frames.cpu()

        combined = torch.cat([original_frames, new_video.frames], dim=0)
        return VideoTensor(frames=combined, fps=video.fps)

    # ------------------------------------------------------------------
    # Public API: video-to-video
    # ------------------------------------------------------------------
    @torch.no_grad()
    def video2video(
        self,
        video: VideoTensor,
        prompt: str,
        strength: float = 0.75,
        steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
    ) -> VideoTensor:
        """Transform a video using a text prompt.

        Args:
            video: Input video.
            prompt: Text prompt guiding the transformation.
            strength: Transformation strength (``0`` = no change,
                ``1`` = full regeneration).
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            negative_prompt: Negative prompt.

        Returns:
            A transformed :class:`VideoTensor`.
        """
        frames = video.frames.to(self._device)
        if frames.dim() == 5:
            frames = frames[0]
        if frames.dim() == 4 and frames.shape[0] > frames.shape[1]:
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Encode the input video to latents.
        video_input = frames.unsqueeze(0)  # (1, C, T, H, W)
        with torch.no_grad():
            mean, _logvar = self.vae.encode(video_input)
        init_latents = mean * self.scaling_factor

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Determine starting timestep.
        num_inference_steps = max(1, int(steps * strength))
        self.scheduler.set_timesteps(steps)
        t_start = max(steps - num_inference_steps, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        # Add noise to the initial latents.
        noise = torch.randn_like(init_latents)
        latents = self.scheduler.add_noise(
            init_latents,
            torch.tensor([timesteps[0]], device=self._device).long(),
            noise=noise,
        )

        # Denoise.
        self.model.eval()
        with torch.no_grad():
            for t in timesteps:
                t_batch = t.expand(latents.shape[0]).to(self._device)
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                embed_input = torch.cat([neg_embeds, pos_embeds])

                noise_pred = self.model(
                    latent_input,
                    timesteps=t_input,
                    encoder_hidden_states=embed_input,
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )
                latents = self.scheduler.step(noise_pred, t_batch, latents)

        return self._latents_to_video(latents, video.fps)

    # ------------------------------------------------------------------
    # Advanced: long-video segmentation + stitching
    # ------------------------------------------------------------------
    @torch.no_grad()
    def txt2video_long(
        self,
        prompt: str,
        total_frames: int = 64,
        segment_frames: int = 16,
        width: int = 512,
        height: int = 512,
        fps: int = 8,
        steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> VideoTensor:
        """Generate a long video by segmenting and stitching.

        Generates the video in segments, using the last frame of each
        segment as the first frame of the next for temporal continuity.

        Args:
            prompt: Text prompt.
            total_frames: Total desired frames.
            segment_frames: Frames per segment.
            width: Video width.
            height: Video height.
            fps: Output frame rate.
            steps: Denoising steps per segment.
            guidance_scale: CFG guidance scale.

        Returns:
            A :class:`VideoTensor` with all segments stitched.
        """
        all_frames: List[torch.Tensor] = []
        num_segments = (total_frames + segment_frames - 1) // segment_frames

        for seg in range(num_segments):
            if seg == 0:
                # First segment: pure text-to-video.
                video = self.txt2video(
                    prompt,
                    width=width,
                    height=height,
                    num_frames=segment_frames,
                    fps=fps,
                    steps=steps,
                    guidance_scale=guidance_scale,
                )
            else:
                # Subsequent segments: extend from the last frame.
                last_frame = all_frames[-1]
                video = self.img2video(
                    last_frame,
                    prompt,
                    num_frames=segment_frames,
                    fps=fps,
                    steps=steps,
                    guidance_scale=guidance_scale,
                )

            seg_frames = video.frames
            if seg_frames.dim() == 5:
                seg_frames = seg_frames[0]
            all_frames.extend(list(seg_frames))

        # Truncate to exact total_frames.
        all_frames = all_frames[:total_frames]
        combined = torch.stack(all_frames)
        return VideoTensor(frames=combined, fps=fps)

    # ------------------------------------------------------------------
    # Advanced: frame rate upscaling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def upscale_fps(
        self,
        video: VideoTensor,
        target_fps: int = 24,
    ) -> VideoTensor:
        """Increase the frame rate using frame interpolation.

        Args:
            video: Input video.
            target_fps: Desired output frame rate.

        Returns:
            A :class:`VideoTensor` at the higher frame rate.
        """
        current_fps = video.fps
        if target_fps <= current_fps:
            return video

        ratio = target_fps / current_fps
        target_frames = int(video.num_frames * ratio)

        frames = video.frames.to(self._device)
        if frames.dim() == 5:
            frames = frames[0]
        if frames.dim() == 4 and frames.shape[0] > frames.shape[1]:
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        interpolated = self._interpolate_frames(
            frames.permute(1, 0, 2, 3),  # (T, C, H, W)
            target_frames,
        )

        return VideoTensor(frames=interpolated.cpu(), fps=target_fps)

    # ------------------------------------------------------------------
    # Advanced: super-resolution
    # ------------------------------------------------------------------
    @torch.no_grad()
    def upscale_resolution(
        self,
        video: VideoTensor,
        scale_factor: int = 2,
    ) -> VideoTensor:
        """Upscale the spatial resolution of a video.

        Uses bilinear interpolation on each frame.

        Args:
            video: Input video.
            scale_factor: Spatial upscaling factor.

        Returns:
            A :class:`VideoTensor` at the higher resolution.
        """
        frames = video.frames
        if frames.dim() == 5:
            frames = frames[0]
        # (T, C, H, W)
        new_h = frames.shape[2] * scale_factor
        new_w = frames.shape[3] * scale_factor
        upscaled = F.interpolate(
            frames, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        return VideoTensor(frames=upscaled, fps=video.fps)

    # ------------------------------------------------------------------
    # Advanced: audio-video sync generation
    # ------------------------------------------------------------------
    def generate_with_audio(
        self,
        prompt: str,
        num_frames: int = 16,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        audio_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate a video and synchronised audio.

        Args:
            prompt: Text prompt for both video and audio.
            num_frames: Number of video frames.
            fps: Video frame rate.
            width: Video width.
            height: Video height.
            steps: Denoising steps.
            guidance_scale: CFG guidance scale.
            audio_engine: Optional :class:`AudioEngine` instance.  When
                ``None`` only the video is returned.

        Returns:
            A dictionary with ``"video"`` and optional ``"audio"`` keys.
        """
        result: Dict[str, Any] = {}

        # Generate video.
        video = self.txt2video(
            prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        result["video"] = video

        # Generate audio if an engine is provided.
        if audio_engine is not None:
            duration = video.duration
            audio = audio_engine.compose(
                prompt, duration=duration, genre="ambient"
            )
            result["audio"] = audio

        return result

    def __repr__(self) -> str:
        return f"VideoEngine(model={self.model_name!r}, device={self._device})"
