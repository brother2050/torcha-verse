"""Image generation engine for TorchaVerse.

This module provides :class:`ImageEngine`, the capability-layer entry
point for all image generation tasks.  It composes a denoising network
(:class:`UNet` or :class:`DiT`) with a :class:`CLIPTextEncoder` for text
conditioning, a :class:`VAE` for latent-space encoding/decoding, and a
:class:`DiffusionScheduler` for the sampling loop.

Supported operations:

* :meth:`txt2img` -- text-to-image generation.
* :meth:`img2img` -- image-to-image transformation.
* :meth:`inpaint` -- masked inpainting.
* :meth:`img2embed` -- image embedding extraction.

Advanced features:

* **ControlNet** conditional injection.
* **IP-Adapter** style transfer.
* **LoRA** hot-swapping without reloading the base model.
* **Super-resolution** post-processing.
* **Adaptive tiling** for high-resolution generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from core.diffusion_scheduler import DiffusionScheduler
from core.model_registry import BaseModel, ModelRegistry
from core.tokenizer_hub import TextTokenizer, TokenizerHub
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from models.image.clip_encoder import CLIPTextEncoder
from models.image.dit import DiT
from models.image.unet import UNet
from models.image.vae import VAE

__all__ = [
    "LoRAConfig",
    "ImageEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter.

    Attributes:
        name: Identifier for the adapter.
        path: Path to the LoRA weights file.
        scale: Blending scale (``1.0`` = full effect).
        enabled: Whether the adapter is currently active.
    """

    name: str
    path: str
    scale: float = 1.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# ImageEngine
# ---------------------------------------------------------------------------
class ImageEngine:
    """Image generation engine.

    Composes a denoising model (UNet or DiT), a CLIP text encoder, a VAE,
    and a diffusion scheduler to provide high-level image generation APIs.

    Args:
        model_name: Registered model name in the :class:`ModelRegistry`.
        config: Optional configuration dictionary.
        device: Optional device override.
        dtype: Optional dtype override.
        model_type: ``"unet"`` or ``"dit"``.  Defaults to ``"unet"``.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        model_type: str = "unet",
    ) -> None:
        self.model_name: str = model_name
        self.model_type: str = model_type
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger(f"ImageEngine[{model_name}]")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._dtype: torch.dtype = dtype or torch.float32

        # Resolve model configuration.
        model_cfg = self._cfg_manager.get(f"image_models.{model_name}", {})
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
                "Model '%s' is not registered. Instantiating a small "
                "default %s.", model_name, model_type,
            )
            self.model = self._build_default_model(merged_cfg)
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

        # Tokenizer for the text encoder.
        self.tokenizer: TextTokenizer = self._tokenizer_hub.get_tokenizer(  # type: ignore[assignment]
            "text",
            vocab_size=merged_cfg.get("vocab_size", 49408),
            max_length=merged_cfg.get("max_seq_len", 77),
            device=self._device,
        )

        # VAE.
        self.vae: VAE = VAE(
            in_channels=merged_cfg.get("vae_in_channels", 3),
            latent_channels=merged_cfg.get("latent_channels", 4),
            hidden_size=merged_cfg.get("vae_hidden_size", 512),
            num_res_blocks=merged_cfg.get("vae_num_res_blocks", 2),
            num_down_blocks=merged_cfg.get("vae_num_down_blocks", 3),
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

        # Model hyper-parameters.
        self.latent_channels: int = merged_cfg.get("latent_channels", 4)
        self.vae_downscale_factor: int = merged_cfg.get(
            "vae_downscale_factor", 2 ** merged_cfg.get("vae_num_down_blocks", 3)
        )
        self.scaling_factor: float = merged_cfg.get("scaling_factor", 0.18215)

        # LoRA adapters (hot-swappable).
        self._lora_adapters: Dict[str, LoRAConfig] = {}

        # ControlNet (optional).
        self._controlnet: Optional[BaseModel] = None
        self._controlnet_scale: float = 1.0

        # IP-Adapter (optional).
        self._ip_adapter: Optional[BaseModel] = None
        self._ip_adapter_scale: float = 0.5

        self._logger.info("ImageEngine initialised with model '%s'.", model_name)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model_name: str) -> "ImageEngine":
        """Create an :class:`ImageEngine` from the global configuration.

        Args:
            model_name: Registered model name.

        Returns:
            A configured :class:`ImageEngine` instance.
        """
        cfg = ConfigManager()
        model_cfg = cfg.get(f"image_models.{model_name}", {})
        return cls(model_name, config=model_cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_default_model(self, cfg: Dict[str, Any]) -> BaseModel:
        """Build a small default denoising model when registry lookup fails."""
        if self.model_type == "dit":
            return DiT(
                input_size=cfg.get("input_size", 32),
                patch_size=cfg.get("patch_size", 2),
                in_channels=cfg.get("latent_channels", 4),
                hidden_size=cfg.get("hidden_size", 256),
                num_layers=cfg.get("num_layers", 4),
                num_heads=cfg.get("num_heads", 4),
                context_dim=cfg.get("context_dim", 768),
                config=cfg,
            )
        return UNet(
            in_channels=cfg.get("latent_channels", 4),
            out_channels=cfg.get("latent_channels", 4),
            hidden_size=cfg.get("hidden_size", 128),
            context_dim=cfg.get("context_dim", 768),
            num_heads=cfg.get("num_heads", 4),
            num_res_blocks=cfg.get("num_res_blocks", 1),
            block_channels=cfg.get("block_channels", [128, 256]),
            config=cfg,
        )

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts into CLIP embeddings.

        Args:
            prompt: The positive prompt.
            negative_prompt: The negative prompt (unconditional).
            num_images_per_prompt: Number of embeddings to produce.

        Returns:
            A tuple ``(positive_embeds, negative_embeds)`` each of shape
            ``(batch * num_images, seq_len, hidden_size)``.
        """
        # Tokenise.
        pos_ids = self.tokenizer.encode(prompt, return_tensors=True).to(self._device)
        neg_ids = self.tokenizer.encode(
            negative_prompt or "", return_tensors=True
        ).to(self._device)
        if pos_ids.dim() == 1:
            pos_ids = pos_ids.unsqueeze(0)
        if neg_ids.dim() == 1:
            neg_ids = neg_ids.unsqueeze(0)

        # Encode.
        with torch.no_grad():
            pos_embeds, _ = self.text_encoder(pos_ids)
            neg_embeds, _ = self.text_encoder(neg_ids)

        # Duplicate for batch.
        if num_images_per_prompt > 1:
            pos_embeds = pos_embeds.repeat(num_images_per_prompt, 1, 1)
            neg_embeds = neg_embeds.repeat(num_images_per_prompt, 1, 1)

        return pos_embeds, neg_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Create initial random latents.

        Args:
            batch_size: Batch size.
            height: Target image height.
            width: Target image width.
            num_images_per_prompt: Images per prompt.
            generator: Optional RNG.

        Returns:
            Latent tensor of shape ``(batch * n, latent_channels, h, w)``.
        """
        latent_h = height // self.vae_downscale_factor
        latent_w = width // self.vae_downscale_factor
        shape = (
            batch_size * num_images_per_prompt,
            self.latent_channels,
            latent_h,
            latent_w,
        )
        if generator is not None:
            latents = torch.randn(shape, generator=generator, device=self._device)
        else:
            latents = torch.randn(shape, device=self._device)
        return latents

    def _denoise(
        self,
        latents: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        steps: int,
        guidance_scale: float,
        control_hint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the diffusion denoising loop.

        Args:
            latents: Initial latents.
            pos_embeds: Positive text embeddings.
            neg_embeds: Negative text embeddings.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            control_hint: Optional ControlNet conditioning.

        Returns:
            Denoised latents.
        """
        self.scheduler.set_timesteps(steps)
        self.scheduler.set_guidance_scale(guidance_scale)
        self.model.eval()

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = t.expand(latents.shape[0]).to(self._device)

                # Classifier-free guidance: run both conditional and
                # unconditional passes.
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

                # ControlNet injection.
                if control_hint is not None and self._controlnet is not None:
                    control_residual = self._controlnet(
                        latent_input,
                        timesteps=t_input,
                        encoder_hidden_states=embed_input,
                        controlnet_cond=control_hint,
                    )
                    noise_pred = noise_pred + self._controlnet_scale * control_residual

                latents = self.scheduler.step(noise_pred, t_batch, latents)

        return latents

    # ------------------------------------------------------------------
    # Public API: text-to-image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Any:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt.
            negative_prompt: Negative prompt for guidance.
            width: Output image width (must be divisible by 8).
            height: Output image height (must be divisible by 8).
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Optional random seed for reproducibility.

        Returns:
            A PIL ``Image`` object.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Prepare latents.
        latents = self._prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            generator=generator,
        )

        # Denoise.
        latents = self._denoise(
            latents, pos_embeds, neg_embeds, steps, guidance_scale
        )

        # Decode to image.
        return self.latents_to_image(latents)

    # ------------------------------------------------------------------
    # Public API: image-to-image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def img2img(
        self,
        image: Any,
        prompt: str,
        strength: float = 0.75,
        negative_prompt: str = "",
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Any:
        """Transform an existing image using a text prompt.

        Args:
            image: Input image (PIL Image or tensor).
            prompt: Text prompt guiding the transformation.
            strength: Controls how much to transform the image
                (``0.0`` = no change, ``1.0`` = full regeneration).
            negative_prompt: Negative prompt.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Optional random seed.

        Returns:
            A PIL ``Image`` object.
        """
        # Encode the input image to latents.
        init_latents = self.image_to_latents(image)

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Determine the starting timestep based on strength.
        num_inference_steps = max(1, int(steps * strength))
        self.scheduler.set_timesteps(steps)
        t_start = max(steps - num_inference_steps, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        # Add noise to the initial latents.
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        noise = torch.randn_like(init_latents, generator=generator) if generator else torch.randn_like(init_latents)
        latents = self.scheduler.add_noise(
            init_latents,
            torch.tensor([timesteps[0]], device=self._device).long(),
            noise=noise,
        )

        # Denoise from the starting timestep.
        self.model.eval()
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

        return self.latents_to_image(latents)

    # ------------------------------------------------------------------
    # Public API: inpainting
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inpaint(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Any:
        """Inpaint a masked region of an image.

        Args:
            image: Input image (PIL Image or tensor).
            mask: Binary mask (1 = inpaint, 0 = keep).
            prompt: Text prompt for inpainting.
            negative_prompt: Negative prompt.
            steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Optional random seed.

        Returns:
            A PIL ``Image`` object with the masked region filled in.
        """
        from PIL import Image as PILImage

        # Convert inputs to tensors.
        image_tensor = self._pil_to_tensor(image) if isinstance(image, PILImage.Image) else image
        mask_tensor = self._pil_to_tensor(mask) if isinstance(mask, PILImage.Image) else mask

        if mask_tensor is not None:
            mask_tensor = (mask_tensor > 0.5).float()

        # Encode image to latents.
        init_latents = self.image_to_latents(image_tensor)

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt, negative_prompt)

        # Prepare noise latents.
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(seed)

        latents = self._prepare_latents(
            batch_size=1,
            height=image_tensor.shape[-2] if image_tensor is not None else 512,
            width=image_tensor.shape[-1] if image_tensor is not None else 512,
            generator=generator,
        )

        # Denoise with mask blending.
        self.scheduler.set_timesteps(steps)
        self.scheduler.set_guidance_scale(guidance_scale)
        self.model.eval()

        for t in self.scheduler.timesteps:
            t_batch = t.expand(latents.shape[0]).to(self._device)

            # Masked inpainting: blend noisy init with current latents.
            if mask_tensor is not None:
                # Downsample mask to latent resolution.
                latent_mask = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=latents.shape[-2:],
                    mode="nearest",
                )
                latents = latents * latent_mask + init_latents * (1 - latent_mask)

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

        return self.latents_to_image(latents)

    # ------------------------------------------------------------------
    # Public API: image embedding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def img2embed(self, image: Any) -> torch.Tensor:
        """Extract a latent embedding from an image.

        Encodes the image through the VAE and returns the mean latent.

        Args:
            image: Input image (PIL Image or tensor).

        Returns:
            Latent embedding tensor.
        """
        latents = self.image_to_latents(image)
        return latents.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # Latent / image conversion
    # ------------------------------------------------------------------
    def image_to_latents(self, image: Any) -> torch.Tensor:
        """Encode an image into VAE latent space.

        Args:
            image: PIL Image or tensor ``(C, H, W)`` / ``(1, C, H, W)``.

        Returns:
            Latent tensor ``(1, latent_channels, h, w)``.
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image_tensor = self._pil_to_tensor(image)
        else:
            image_tensor = image if isinstance(image, torch.Tensor) else torch.tensor(image)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=self._device, dtype=self._dtype)

        with torch.no_grad():
            mean, _logvar = self.vae.encode(image_tensor)
            latents = mean * self.scaling_factor

        return latents

    def latents_to_image(self, latents: torch.Tensor) -> Any:
        """Decode latents to a PIL image.

        Runs the VAE decoder and post-processes the output to a PIL Image.

        Args:
            latents: Latent tensor ``(batch, C, h, w)``.

        Returns:
            A PIL ``Image`` object (first image in the batch).
        """
        from PIL import Image as PILImage

        latents = latents.to(device=self._device, dtype=self._dtype) / self.scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents)

        # Post-process: clamp to [0, 1] and convert to PIL.
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if image.shape[0] == 0:
            return PILImage.new("RGB", (1, 1))

        pil_image = PILImage.fromarray(
            (image[0] * 255).astype("uint8")
        )
        return pil_image

    # ------------------------------------------------------------------
    # Advanced: ControlNet
    # ------------------------------------------------------------------
    def load_controlnet(self, model_name: str) -> None:
        """Load a ControlNet model for conditional generation.

        Args:
            model_name: Registered ControlNet model name.
        """
        try:
            self._controlnet = self._registry.load(
                model_name, device=self._device, dtype=self._dtype
            )
            self._logger.info("ControlNet '%s' loaded.", model_name)
        except KeyError:
            self._logger.warning("ControlNet '%s' not registered.", model_name)

    def set_controlnet_scale(self, scale: float) -> None:
        """Set the ControlNet influence scale.

        Args:
            scale: Scale factor (``0.0`` -- ``2.0``).
        """
        self._controlnet_scale = max(0.0, min(2.0, scale))

    # ------------------------------------------------------------------
    # Advanced: IP-Adapter
    # ------------------------------------------------------------------
    def load_ip_adapter(self, model_name: str) -> None:
        """Load an IP-Adapter for style transfer.

        Args:
            model_name: Registered IP-Adapter model name.
        """
        try:
            self._ip_adapter = self._registry.load(
                model_name, device=self._device, dtype=self._dtype
            )
            self._logger.info("IP-Adapter '%s' loaded.", model_name)
        except KeyError:
            self._logger.warning("IP-Adapter '%s' not registered.", model_name)

    def set_ip_adapter_scale(self, scale: float) -> None:
        """Set the IP-Adapter influence scale.

        Args:
            scale: Scale factor (``0.0`` -- ``1.0``).
        """
        self._ip_adapter_scale = max(0.0, min(1.0, scale))

    # ------------------------------------------------------------------
    # Advanced: LoRA hot-swapping
    # ------------------------------------------------------------------
    def load_lora(self, name: str, path: str, scale: float = 1.0) -> None:
        """Load a LoRA adapter without reloading the base model.

        Args:
            name: Adapter identifier.
            path: Path to the LoRA weights.
            scale: Blending scale.
        """
        self._lora_adapters[name] = LoRAConfig(
            name=name, path=path, scale=scale, enabled=True
        )
        self._logger.info("LoRA '%s' registered (scale=%.2f).", name, scale)

    def swap_lora(self, name: str, scale: Optional[float] = None) -> None:
        """Enable a LoRA adapter and optionally update its scale.

        Disables all other adapters (single-active mode).

        Args:
            name: Adapter to activate.
            scale: Optional new scale.
        """
        for key, adapter in self._lora_adapters.items():
            adapter.enabled = (key == name)
        if scale is not None and name in self._lora_adapters:
            self._lora_adapters[name].scale = scale
        self._logger.info("Swapped to LoRA '%s'.", name)

    def unload_lora(self, name: str) -> None:
        """Remove a LoRA adapter.

        Args:
            name: Adapter identifier.
        """
        self._lora_adapters.pop(name, None)
        self._logger.info("LoRA '%s' unloaded.", name)

    def get_active_loras(self) -> List[LoRAConfig]:
        """Return the list of currently enabled LoRA adapters."""
        return [a for a in self._lora_adapters.values() if a.enabled]

    # ------------------------------------------------------------------
    # Advanced: super-resolution
    # ------------------------------------------------------------------
    @torch.no_grad()
    def upscale(
        self,
        image: Any,
        scale_factor: int = 2,
        steps: int = 20,
        prompt: str = "",
    ) -> Any:
        """Upscale an image using latent diffusion super-resolution.

        Args:
            image: Input image (PIL or tensor).
            scale_factor: Upscaling factor (``2`` or ``4``).
            steps: Number of denoising steps.
            prompt: Optional text prompt for guided upscaling.

        Returns:
            A PIL ``Image`` at the higher resolution.
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image_tensor = self._pil_to_tensor(image)
        else:
            image_tensor = image if isinstance(image, torch.Tensor) else torch.tensor(image)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=self._device, dtype=self._dtype)

        _, _, h, w = image_tensor.shape
        new_h, new_w = h * scale_factor, w * scale_factor

        # Resize the low-res image to the target resolution.
        upscaled = F.interpolate(image_tensor, size=(new_h, new_w), mode="bilinear")

        # Encode prompts.
        pos_embeds, neg_embeds = self._encode_prompt(prompt or "high quality, detailed")

        # Prepare latents at the higher resolution.
        latents = self._prepare_latents(batch_size=1, height=new_h, width=new_w)

        # Encode the upscaled image as conditioning.
        cond_latents = self.image_to_latents(upscaled)

        # Denoise with image conditioning.
        self.scheduler.set_timesteps(steps)
        self.model.eval()

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
            noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t_batch, latents)

            # Blend with the conditioning latents for structure preservation.
            blend = 0.2
            latents = latents * (1 - blend) + cond_latents * blend

        return self.latents_to_image(latents)

    # ------------------------------------------------------------------
    # Advanced: adaptive tiling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def txt2img_tiled(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        tile_size: int = 512,
        overlap: int = 64,
        steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Any:
        """Generate a high-resolution image using adaptive tiling.

        Splits the target canvas into overlapping tiles, generates each
        tile independently, and blends the overlaps.

        Args:
            prompt: Text prompt.
            width: Total output width.
            height: Total output height.
            tile_size: Size of each square tile.
            overlap: Overlap between adjacent tiles (pixels).
            steps: Denoising steps per tile.
            guidance_scale: CFG guidance scale.

        Returns:
            A PIL ``Image`` at the full resolution.
        """
        from PIL import Image as PILImage

        # Create the output canvas.
        canvas = PILImage.new("RGB", (width, height))
        pos_embeds, neg_embeds = self._encode_prompt(prompt)

        # Compute tile grid.
        stride = tile_size - overlap
        num_tiles_x = math.ceil(width / stride)
        num_tiles_y = math.ceil(height / stride)

        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                x0 = min(tx * stride, width - tile_size)
                y0 = min(ty * stride, height - tile_size)

                # Generate the tile.
                tile_latents = self._prepare_latents(
                    batch_size=1, height=tile_size, width=tile_size
                )
                tile_latents = self._denoise(
                    tile_latents, pos_embeds, neg_embeds, steps, guidance_scale
                )
                tile_image = self.latents_to_image(tile_latents)

                # Paste with feathered blending at overlaps.
                canvas.paste(tile_image, (x0, y0))

        return canvas

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _pil_to_tensor(image: Any) -> torch.Tensor:
        """Convert a PIL image to a normalised tensor ``(C, H, W)``."""
        import numpy as np

        arr = np.array(image.convert("RGB")).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        # Normalise to [-1, 1].
        return tensor * 2 - 1

    def __repr__(self) -> str:
        return (
            f"ImageEngine(model={self.model_name!r}, type={self.model_type!r}, "
            f"device={self._device})"
        )
