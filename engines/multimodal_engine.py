"""Multi-modal fusion engine for TorchaVerse.

This module provides :class:`MultiModalEngine`, the capability-layer
entry point for cross-modal understanding and generation.  It composes
:class:`TextEngine`, :class:`ImageEngine`, :class:`AudioEngine`, and
:class:`VideoEngine` into a unified interface, and optionally leverages a
:class:`VisionLanguageModel` or :class:`OmniModel` for native multi-modal
reasoning.

Supported operations:

* :meth:`understand` -- multi-modal understanding (image/audio/video +
  optional question -> text answer).
* :meth:`generate` -- multi-modal generation (prompt -> one or more
  output modalities).
* :meth:`caption` -- image captioning.
* :meth:`retrieve` -- cross-modal retrieval (text query -> matching
  images / audio / video).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel, ModelRegistry
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from models.multimodal.omni_model import OmniModel
from models.multimodal.vision_language import VisionLanguageModel
from .audio_engine import AudioEngine, AudioTensor
from .image_engine import ImageEngine
from .text_engine import TextEngine
from .video_engine import VideoEngine, VideoTensor

__all__ = [
    "GenerateRequest",
    "MultiModalEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GenerateRequest:
    """A unified generation request that can target any modality.

    Attributes:
        text: Optional text prompt.
        image: Optional input image (PIL or tensor).
        audio: Optional input audio (:class:`AudioTensor`).
        video: Optional input video (:class:`VideoTensor`).
        output_modality: Desired output modality (``"text"``,
            ``"image"``, ``"audio"``, ``"video"``, or a list thereof).
        params: Additional generation parameters.
    """

    text: Optional[str] = None
    image: Optional[Any] = None
    audio: Optional[AudioTensor] = None
    video: Optional[VideoTensor] = None
    output_modality: Union[str, List[str]] = "text"
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MultiModalEngine
# ---------------------------------------------------------------------------
class MultiModalEngine:
    """Multi-modal fusion engine.

    Composes text, image, audio, and video engines, and optionally a
    native multi-modal model (VLM or OmniModel) for unified
    understanding and generation.

    Args:
        text_engine: Optional pre-configured :class:`TextEngine`.
        image_engine: Optional pre-configured :class:`ImageEngine`.
        audio_engine: Optional pre-configured :class:`AudioEngine`.
        video_engine: Optional pre-configured :class:`VideoEngine`.
        vlm_model_name: Optional Vision-Language model name for native
            multi-modal understanding.
        omni_model_name: Optional OmniModel name for full multi-modal
            understanding and generation.
        config: Optional configuration dictionary.
        device: Optional device override.
    """

    def __init__(
        self,
        text_engine: Optional[TextEngine] = None,
        image_engine: Optional[ImageEngine] = None,
        audio_engine: Optional[AudioEngine] = None,
        video_engine: Optional[VideoEngine] = None,
        vlm_model_name: Optional[str] = None,
        omni_model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("MultiModalEngine")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        # Sub-engines (create defaults if not provided).
        self.text_engine: TextEngine = text_engine or TextEngine(
            "default", device=self._device
        )
        self.image_engine: ImageEngine = image_engine or ImageEngine(
            "default", device=self._device
        )
        self.audio_engine: AudioEngine = audio_engine or AudioEngine(
            device=self._device
        )
        self.video_engine: VideoEngine = video_engine or VideoEngine(
            "default", device=self._device
        )

        # Optional native multi-modal models.
        self._registry: ModelRegistry = ModelRegistry()
        self.vlm: Optional[VisionLanguageModel] = None
        self.omni: Optional[OmniModel] = None

        if vlm_model_name:
            try:
                self.vlm = self._registry.load(
                    vlm_model_name, device=self._device
                )
            except KeyError:
                self._logger.warning(
                    "VLM '%s' not registered; using engine composition.",
                    vlm_model_name,
                )

        if omni_model_name:
            try:
                self.omni = self._registry.load(
                    omni_model_name, device=self._device
                )
            except KeyError:
                self._logger.warning(
                    "OmniModel '%s' not registered; using engine composition.",
                    omni_model_name,
                )

        # Cross-modal embedding cache for retrieval.
        self._embedding_cache: Dict[str, List[Tuple[Any, torch.Tensor]]] = {
            "image": [],
            "audio": [],
            "video": [],
        }

        self._logger.info("MultiModalEngine initialised.")

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def understand(
        self,
        image: Optional[Any] = None,
        audio: Optional[AudioTensor] = None,
        video: Optional[VideoTensor] = None,
        text: Optional[str] = None,
        question: Optional[str] = None,
        max_tokens: int = 256,
    ) -> str:
        """Understand multi-modal input and optionally answer a question.

        When a native VLM or OmniModel is available, it is used directly.
        Otherwise, the method falls back to composing the individual
        engines: extract features from each modality, concatenate them
        into a text prompt, and generate a response with the text engine.

        Args:
            image: Optional input image.
            audio: Optional input audio.
            video: Optional input video.
            text: Optional input text.
            question: Optional question about the input.
            max_tokens: Maximum tokens for the response.

        Returns:
            A text answer.
        """
        # Build the question prompt.
        prompt_parts: List[str] = []
        if text:
            prompt_parts.append(f"[Text] {text}")
        if question:
            prompt_parts.append(f"[Question] {question}")

        # Native OmniModel path.
        if self.omni is not None:
            return self._understand_omni(
                image, audio, video, text, question, max_tokens
            )

        # Native VLM path (image + text only).
        if self.vlm is not None and image is not None:
            return self._understand_vlm(image, question or text or "", max_tokens)

        # Fallback: compose individual engines.
        if image is not None:
            caption = self.caption(image)
            prompt_parts.append(f"[Image Description] {caption}")

        if audio is not None:
            transcript = self.audio_engine.transcribe(audio)
            prompt_parts.append(f"[Audio Transcript] {transcript}")

        if video is not None:
            # Use the middle frame for a quick caption.
            frames = video.frames
            if frames.dim() == 5:
                frames = frames[0]
            mid_idx = frames.shape[0] // 2 if frames.dim() == 4 else 0
            if frames.dim() == 4:
                mid_frame = frames[mid_idx]
                from PIL import Image as PILImage
                import numpy as np
                arr = (mid_frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                pil_frame = PILImage.fromarray(arr)
                video_caption = self.caption(pil_frame)
                prompt_parts.append(f"[Video Frame Description] {video_caption}")

        prompt_parts.append("[Answer]")
        prompt = "\n".join(prompt_parts)

        return self.text_engine.generate(prompt, max_tokens=max_tokens)

    def _understand_omni(
        self,
        image: Optional[Any],
        audio: Optional[AudioTensor],
        video: Optional[VideoTensor],
        text: Optional[str],
        question: Optional[str],
        max_tokens: int,
    ) -> str:
        """Use the OmniModel for multi-modal understanding."""
        inputs: Dict[str, torch.Tensor] = {}

        if image is not None:
            img_tensor = self._to_image_tensor(image)
            inputs["image"] = img_tensor

        if audio is not None:
            # Encode audio to mel-like features.
            mel = self._audio_to_features(audio)
            inputs["audio"] = mel

        if video is not None:
            # Use the first frame of the video.
            frames = video.frames
            if frames.dim() == 5:
                frames = frames[0]
            if frames.dim() == 4:
                inputs["image"] = frames[0]

        # Text prompt.
        prompt = question or text or "Describe the input."
        text_ids = self.text_engine.tokenize(prompt)
        inputs["text_ids"] = torch.tensor([text_ids], device=self._device)

        # Generate.
        output_ids = self.omni.generate(inputs, max_tokens=max_tokens)  # type: ignore[union-attr]
        new_ids = output_ids[0].tolist()
        return self.text_engine.detokenize(new_ids)

    def _understand_vlm(
        self,
        image: Any,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Use the Vision-Language Model for image understanding."""
        img_tensor = self._to_image_tensor(image).to(self._device)
        text_ids = self.text_engine.tokenize(prompt)
        text_tensor = torch.tensor([text_ids], device=self._device)

        with torch.no_grad():
            logits = self.vlm.forward(image=img_tensor, text_ids=text_tensor)  # type: ignore[union-attr]

        # Greedy decode from the logits.
        output_ids = torch.argmax(logits, dim=-1)[0].tolist()
        return self.text_engine.detokenize(output_ids)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        output_modalities: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate one or more modalities from a text prompt.

        Args:
            prompt: Input text prompt.
            output_modalities: Desired output modality or list of
                modalities (``"text"``, ``"image"``, ``"audio"``,
                ``"video"``).  Defaults to ``["text"]``.
            **kwargs: Additional generation parameters forwarded to the
                respective engine.

        Returns:
            A dictionary mapping modality names to generated outputs.
        """
        if output_modalities is None:
            output_modalities = ["text"]
        if isinstance(output_modalities, str):
            output_modalities = [output_modalities]

        results: Dict[str, Any] = {}

        for modality in output_modalities:
            if modality == "text":
                results["text"] = self.text_engine.generate(prompt, **kwargs)
            elif modality == "image":
                results["image"] = self.image_engine.txt2img(prompt, **kwargs)
            elif modality == "audio":
                results["audio"] = self.audio_engine.synthesize(prompt, **kwargs)
            elif modality == "video":
                results["video"] = self.video_engine.txt2video(prompt, **kwargs)
            else:
                self._logger.warning("Unknown output modality '%s'.", modality)

        return results

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------
    def route_request(self, request: GenerateRequest) -> Dict[str, Any]:
        """Route a :class:`GenerateRequest` to the appropriate engine(s).

        Args:
            request: The generation request.

        Returns:
            A dictionary of generated outputs keyed by modality.
        """
        modalities = request.output_modality
        if isinstance(modalities, str):
            modalities = [modalities]

        results: Dict[str, Any] = {}
        prompt = request.text or ""

        for modality in modalities:
            if modality == "text":
                results["text"] = self.text_engine.generate(
                    prompt, **request.params
                )
            elif modality == "image":
                if request.image is not None:
                    results["image"] = self.image_engine.img2img(
                        request.image, prompt, **request.params
                    )
                else:
                    results["image"] = self.image_engine.txt2img(
                        prompt, **request.params
                    )
            elif modality == "audio":
                results["audio"] = self.audio_engine.generate(
                    prompt, **request.params
                )
            elif modality == "video":
                results["video"] = self.video_engine.txt2video(
                    prompt, **request.params
                )

        return results

    # ------------------------------------------------------------------
    # Cross-modal: captioning
    # ------------------------------------------------------------------
    def caption(self, image: Any, max_length: int = 100) -> str:
        """Generate a text caption for an image.

        Args:
            image: Input image (PIL or tensor).
            max_length: Maximum caption length in tokens.

        Returns:
            A caption string.
        """
        # Use VLM if available.
        if self.vlm is not None:
            return self._understand_vlm(image, "Describe this image.", max_length)

        # Fallback: extract image embedding and generate text.
        embedding = self.image_engine.img2embed(image)
        # Use the embedding as a pseudo-prompt.
        prompt = f"[Image embedding: {embedding.shape}] Describe this image."
        return self.text_engine.generate(prompt, max_tokens=max_length)

    # ------------------------------------------------------------------
    # Cross-modal: retrieval
    # ------------------------------------------------------------------
    def index(
        self,
        items: List[Any],
        modality: str = "image",
    ) -> None:
        """Index items for cross-modal retrieval.

        Computes and stores embeddings for later retrieval.

        Args:
            items: List of items (images, audio, or video).
            modality: Modality of the items.
        """
        if modality not in self._embedding_cache:
            self._embedding_cache[modality] = []

        for item in items:
            if modality == "image":
                embed = self.image_engine.img2embed(item)
            elif modality == "audio":
                # Use codec encoding as a simple embedding.
                tokens = self.audio_engine.encode_audio(item)
                embed = tokens.float().mean(dim=-1).squeeze(0)
            elif modality == "video":
                frames = item.frames
                if frames.dim() == 5:
                    frames = frames[0]
                mid = frames[frames.shape[0] // 2] if frames.dim() == 4 else frames
                embed = self.image_engine.img2embed(mid)
            else:
                continue
            self._embedding_cache[modality].append((item, embed))

        self._logger.info("Indexed %d %s items.", len(items), modality)

    def retrieve(
        self,
        text_query: str,
        modality: str = "image",
        top_k: int = 5,
    ) -> List[Any]:
        """Retrieve items matching a text query.

        Computes the text embedding and finds the closest indexed items
        using cosine similarity.

        Args:
            text_query: Text query.
            modality: Modality to search (``"image"``, ``"audio"``,
                ``"video"``).
            top_k: Number of results to return.

        Returns:
            A list of the top-k matching items.
        """
        if modality not in self._embedding_cache:
            return []

        cache = self._embedding_cache[modality]
        if not cache:
            return []

        # Compute query embedding.
        query_embed = self.text_engine.embed(text_query)

        # Compute cosine similarities.
        similarities: List[Tuple[float, Any]] = []
        for item, embed in cache:
            # Ensure compatible dimensions.
            if embed.shape[0] != query_embed.shape[0]:
                min_dim = min(embed.shape[0], query_embed.shape[0])
                e = embed[:min_dim]
                q = query_embed[:min_dim]
            else:
                e = embed
                q = query_embed

            sim = F.cosine_similarity(
                q.unsqueeze(0), e.unsqueeze(0), dim=-1
            ).item()
            similarities.append((sim, item))

        # Sort by similarity (descending) and return top-k.
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:top_k]]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _to_image_tensor(self, image: Any) -> torch.Tensor:
        """Convert a PIL image or tensor to a normalised tensor."""
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(image, PILImage.Image):
            arr = np.array(image.convert("RGB")).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.tensor(image, dtype=torch.float32)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _audio_to_features(self, audio: AudioTensor) -> torch.Tensor:
        """Convert audio to mel-like features for the OmniModel."""
        waveform = audio.waveform.to(self._device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Simple STFT-based mel features.
        with torch.no_grad():
            stft = torch.stft(
                waveform.squeeze(0),
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                return_complex=True,
            )
            mel = stft.abs()  # (1, freq_bins, time)
            # Reduce to 80 mel bins via simple averaging.
            if mel.shape[1] > 80:
                mel = mel[:, :80, :]

        return mel

    def __repr__(self) -> str:
        engines = [
            "text" if self.text_engine else None,
            "image" if self.image_engine else None,
            "audio" if self.audio_engine else None,
            "video" if self.video_engine else None,
        ]
        active = [e for e in engines if e]
        return f"MultiModalEngine(engines=[{', '.join(active)}], device={self._device})"
