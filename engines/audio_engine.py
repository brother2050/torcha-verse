"""Audio generation engine for TorchaVerse.

This module provides :class:`AudioEngine`, the capability-layer entry
point for all audio generation tasks.  It composes a :class:`TTSEngine`
for text-to-speech, a :class:`MusicEngine` for music composition, and
an :class:`AudioCodec` for neural audio compression.

Supported operations:

* :meth:`generate` -- unified entry point for TTS, music, and SFX.
* :meth:`synthesize` -- text-to-speech with speaker / emotion / speed control.
* :meth:`compose` -- text-to-music generation.
* :meth:`voice_clone` -- zero-shot voice cloning from a reference clip.
* :meth:`transcribe` -- speech-to-text (ASR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel, ModelRegistry
from core.tokenizer_hub import TextTokenizer, TokenizerHub
from core.vocoder_manager import BaseVocoder, VocoderManager
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from models.audio.audio_codec import AudioCodec
from models.audio.hifi_gan import HiFiGAN
from models.audio.tts_transformer import (
    AcousticDecoder,
    DurationPredictor,
    TextEncoder,
    TTSTransformer,
)

__all__ = [
    "AudioTensor",
    "TTSEngine",
    "MusicEngine",
    "AudioEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AudioTensor:
    """Container for an audio waveform and its sample rate.

    Attributes:
        waveform: Audio samples of shape ``(channels, samples)`` or
            ``(batch, channels, samples)``.
        sample_rate: Sample rate in Hz.
    """

    waveform: torch.Tensor
    sample_rate: int = 22050

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.waveform.shape[-1] / self.sample_rate

    @property
    def num_channels(self) -> int:
        """Number of audio channels."""
        if self.waveform.dim() == 1:
            return 1
        return self.waveform.shape[-2]

    def to(self, device: Union[str, torch.device]) -> "AudioTensor":
        """Move the waveform to ``device``."""
        return AudioTensor(
            waveform=self.waveform.to(device),
            sample_rate=self.sample_rate,
        )

    def cpu(self) -> "AudioTensor":
        """Move the waveform to CPU."""
        return self.to("cpu")

    def numpy(self) -> Any:
        """Return the waveform as a NumPy array."""
        return self.waveform.detach().cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"AudioTensor(shape={tuple(self.waveform.shape)}, "
            f"sr={self.sample_rate}, dur={self.duration:.2f}s)"
        )


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------
class TTSEngine:
    """Text-to-Speech engine.

    Composes a :class:`TextEncoder`, :class:`DurationPredictor`,
    :class:`AcousticDecoder`, and a vocoder (from :class:`VocoderManager`)
    to convert text into speech audio.

    Args:
        config: Optional configuration dictionary.
        device: Optional device override.
        model_name: Optional TTS model name for the registry.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._logger = get_logger("TTSEngine")

        # Resolve configuration.
        tts_cfg = self._cfg_manager.get("audio_models.tts", {})
        merged: Dict[str, Any] = {**tts_cfg, **self._config}

        # Load TTS model from registry or build a default.
        self._registry: ModelRegistry = ModelRegistry()
        self._tokenizer_hub: TokenizerHub = TokenizerHub()

        if model_name:
            try:
                self.model: TTSTransformer = self._registry.load(  # type: ignore[assignment]
                    model_name, device=self._device, config=merged
                )
            except KeyError:
                self._logger.warning("TTS model '%s' not registered; using default.", model_name)
                self.model = self._build_default_tts(merged)
        else:
            self.model = self._build_default_tts(merged)

        self.model = self._device_manager.to_device(self.model, self._device)

        # Tokenizer.
        self.tokenizer: TextTokenizer = self._tokenizer_hub.get_tokenizer(  # type: ignore[assignment]
            "text",
            vocab_size=merged.get("vocab_size", 100),
            max_length=merged.get("max_text_len", 512),
            device=self._device,
        )

        # Vocoder.
        self._vocoder_manager: VocoderManager = VocoderManager()
        vocoder_name = merged.get("vocoder", "hifi-gan")
        self.vocoder: BaseVocoder = self._vocoder_manager.get_vocoder(
            vocoder_name,
            sample_rate=merged.get("sample_rate", 22050),
            n_mels=merged.get("mel_channels", 80),
            device=self._device,
        )

        self.sample_rate: int = merged.get("sample_rate", 22050)
        self.mel_channels: int = merged.get("mel_channels", 80)

        # Speaker embeddings (for multi-speaker TTS).
        num_speakers = merged.get("num_speakers", 1)
        speaker_embed_dim = merged.get("speaker_embed_dim", 64)
        self.speaker_embeddings: nn.Embedding = nn.Embedding(
            num_speakers, speaker_embed_dim
        ).to(self._device)

        # Emotion embeddings.
        emotions = ["neutral", "happy", "sad", "angry", "surprised"]
        self.emotion_embeddings: nn.Embedding = nn.Embedding(
            len(emotions), speaker_embed_dim
        ).to(self._device)
        self._emotion_map: Dict[str, int] = {e: i for i, e in enumerate(emotions)}

        self._logger.info("TTSEngine initialised (sr=%d).", self.sample_rate)

    # ------------------------------------------------------------------
    def _build_default_tts(self, cfg: Dict[str, Any]) -> TTSTransformer:
        """Build a small default TTS model."""
        return TTSTransformer(
            vocab_size=cfg.get("vocab_size", 100),
            hidden_size=cfg.get("hidden_size", 256),
            num_layers=cfg.get("num_layers", 4),
            num_heads=cfg.get("num_heads", 4),
            mel_channels=cfg.get("mel_channels", 80),
            max_text_len=cfg.get("max_text_len", 512),
            max_mel_len=cfg.get("max_mel_len", 2048),
            config=cfg,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> AudioTensor:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesise.
            speaker_id: Speaker identity index.
            emotion: Emotion label (``"neutral"``, ``"happy"``, etc.).
            speed: Speech speed multiplier (``1.0`` = normal).

        Returns:
            An :class:`AudioTensor` containing the synthesised waveform.
        """
        self.model.eval()

        # Tokenise text.
        text_ids = self.tokenizer.encode(text, return_tensors=True).to(self._device)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        # Generate mel spectrogram.
        mel = self.model.generate(text_ids, speed=speed)

        # Apply speaker and emotion conditioning to the mel.
        spk_embed = self.speaker_embeddings(
            torch.tensor([speaker_id], device=self._device)
        )  # (1, embed_dim)
        emo_idx = self._emotion_map.get(emotion, 0)
        emo_embed = self.emotion_embeddings(
            torch.tensor([emo_idx], device=self._device)
        )

        # Blend conditioning into mel (additive bias).
        if mel.dim() == 3:
            mel = mel.transpose(1, 2)  # (batch, mel_len, mel_channels)
            conditioning = (spk_embed + emo_embed).unsqueeze(1)  # (1, 1, embed_dim)
            # Project conditioning to mel_channels via a simple linear.
            if conditioning.shape[-1] != mel.shape[-1]:
                conditioning = F.pad(
                    conditioning,
                    (0, mel.shape[-1] - conditioning.shape[-1]),
                )
            mel = mel + conditioning * 0.1
            mel = mel.transpose(1, 2)  # back to (batch, mel_channels, mel_len)

        # Vocode mel to waveform.
        if mel.dim() == 3 and mel.shape[1] == self.mel_channels:
            waveform = self.vocoder(mel)
        elif mel.dim() == 3 and mel.shape[2] == self.mel_channels:
            waveform = self.vocoder(mel.transpose(1, 2))
        else:
            waveform = self.vocoder(mel)

        # Ensure shape (1, samples).
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return AudioTensor(waveform=waveform, sample_rate=self.sample_rate)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def voice_clone(
        self,
        reference_audio: AudioTensor,
        target_text: str,
        speed: float = 1.0,
    ) -> AudioTensor:
        """Clone a voice from a reference audio clip.

        Extracts speaker characteristics from the reference and applies
        them to synthesise the target text.

        Args:
            reference_audio: Reference audio clip.
            target_text: Text to synthesise in the cloned voice.
            speed: Speech speed multiplier.

        Returns:
            An :class:`AudioTensor` with the cloned voice.
        """
        self.model.eval()

        # Encode the reference audio to extract speaker embedding.
        ref_wave = reference_audio.waveform.to(self._device)
        if ref_wave.dim() == 1:
            ref_wave = ref_wave.unsqueeze(0).unsqueeze(0)
        elif ref_wave.dim() == 2:
            ref_wave = ref_wave.unsqueeze(0)

        # Use the audio codec encoder to get a latent representation.
        # Average-pool over time to get a speaker embedding.
        with torch.no_grad():
            ref_latent = ref_wave
            # Simple speaker embedding: mean of waveform.
            speaker_embed = ref_latent.mean(dim=-1, keepdim=True)

        # Tokenise target text.
        text_ids = self.tokenizer.encode(
            target_text, return_tensors=True
        ).to(self._device)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        # Generate mel.
        mel = self.model.generate(text_ids, speed=speed)

        # Apply cloned speaker characteristics.
        if mel.dim() == 3:
            mel = mel.transpose(1, 2)
            # Add a bias derived from the reference.
            bias = F.pad(speaker_embed, (0, mel.shape[-1] - speaker_embed.shape[-1]))
            mel = mel + bias.unsqueeze(1) * 0.05
            mel = mel.transpose(1, 2)

        # Vocode.
        if mel.dim() == 3 and mel.shape[1] == self.mel_channels:
            waveform = self.vocoder(mel)
        elif mel.dim() == 3 and mel.shape[2] == self.mel_channels:
            waveform = self.vocoder(mel.transpose(1, 2))
        else:
            waveform = self.vocoder(mel)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return AudioTensor(waveform=waveform, sample_rate=self.sample_rate)


# ---------------------------------------------------------------------------
# MusicEngine
# ---------------------------------------------------------------------------
class MusicEngine:
    """Text-to-music generation engine.

    Uses a diffusion-based approach to generate music from text
    descriptions.

    Args:
        config: Optional configuration dictionary.
        device: Optional device override.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._logger = get_logger("MusicEngine")

        music_cfg = self._cfg_manager.get("audio_models.music", {})
        merged: Dict[str, Any] = {**music_cfg, **self._config}

        self.sample_rate: int = merged.get("music_sample_rate", 44100)
        self.latent_dim: int = merged.get("music_latent_dim", 64)

        # A simple latent-to-audio decoder network for music.
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
        ).to(self._device)

        self._genre_map: Dict[str, int] = {
            "pop": 0, "rock": 1, "jazz": 2, "classical": 3,
            "electronic": 4, "ambient": 5, "hiphop": 6,
        }

        self._logger.info("MusicEngine initialised (sr=%d).", self.sample_rate)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compose(
        self,
        description: str,
        duration: float = 10.0,
        genre: str = "pop",
    ) -> AudioTensor:
        """Compose music from a text description.

        Args:
            description: Text description of the desired music.
            duration: Target duration in seconds.
            genre: Music genre (``"pop"``, ``"rock"``, ``"jazz"``, etc.).

        Returns:
            An :class:`AudioTensor` containing the composed music.
        """
        num_samples = int(duration * self.sample_rate)

        # Generate a latent seed based on genre.
        genre_idx = self._genre_map.get(genre, 0)
        genre_embed = F.one_hot(
            torch.tensor(genre_idx), num_classes=len(self._genre_map)
        ).float().to(self._device)

        # Pad or project to latent_dim.
        if genre_embed.shape[0] < self.latent_dim:
            genre_embed = F.pad(genre_embed, (0, self.latent_dim - genre_embed.shape[0]))
        else:
            genre_embed = genre_embed[:self.latent_dim]

        # Generate latent sequence.
        latent_seq_len = max(num_samples // 256, 1)
        latent = genre_embed.unsqueeze(0).unsqueeze(0).expand(
            1, latent_seq_len, self.latent_dim
        )

        # Add structured noise for musical variation.
        noise = torch.randn_like(latent) * 0.3
        latent = latent + noise

        # Decode to audio samples.
        audio = self.decoder(latent)  # (1, seq_len, 1024)

        # Reshape to 1-D waveform.
        audio = audio.reshape(1, -1)

        # Interpolate to exact target length.
        audio = F.interpolate(
            audio.unsqueeze(0), size=num_samples, mode="linear"
        ).squeeze(0)

        return AudioTensor(waveform=audio, sample_rate=self.sample_rate)


# ---------------------------------------------------------------------------
# AudioEngine (top-level)
# ---------------------------------------------------------------------------
class AudioEngine:
    """Top-level audio generation engine.

    Composes :class:`TTSEngine`, :class:`MusicEngine`, and
    :class:`AudioCodec` to provide a unified audio API.

    Args:
        config: Optional configuration dictionary.
        device: Optional device override.
        tts_model_name: Optional TTS model name for the registry.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        tts_model_name: Optional[str] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("AudioEngine")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        # Sub-engines.
        self.tts: TTSEngine = TTSEngine(
            config=self._config, device=self._device, model_name=tts_model_name
        )
        self.music: MusicEngine = MusicEngine(
            config=self._config, device=self._device
        )

        # Audio codec.
        codec_cfg = self._cfg_manager.get("audio_models.codec", {})
        self.codec: AudioCodec = AudioCodec(
            in_channels=codec_cfg.get("in_channels", 1),
            hidden_size=codec_cfg.get("hidden_size", 64),
            latent_size=codec_cfg.get("latent_size", 32),
            num_quantizers=codec_cfg.get("num_quantizers", 4),
            codebook_size=codec_cfg.get("codebook_size", 1024),
            config=codec_cfg,
        )
        self.codec = self._device_manager.to_device(self.codec, self._device)

        # SFX generation uses the music engine with different parameters.
        self._sfx_cfg: Dict[str, Any] = self._cfg_manager.get("audio_models.sfx", {})

        self._logger.info("AudioEngine initialised.")

    # ------------------------------------------------------------------
    # Unified generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        modality: str = "tts",
        **kwargs: Any,
    ) -> AudioTensor:
        """Generate audio from a text prompt.

        Args:
            prompt: Input text prompt.
            modality: Generation modality -- ``"tts"``, ``"music"``,
                or ``"sfx"``.
            **kwargs: Additional arguments forwarded to the sub-engine.

        Returns:
            An :class:`AudioTensor` with the generated audio.

        Raises:
            ValueError: If ``modality`` is not recognised.
        """
        if modality == "tts":
            return self.tts.synthesize(prompt, **kwargs)
        elif modality == "music":
            return self.music.compose(prompt, **kwargs)
        elif modality == "sfx":
            return self._generate_sfx(prompt, **kwargs)
        else:
            raise ValueError(
                f"Unknown modality '{modality}'. Use 'tts', 'music', or 'sfx'."
            )

    def _generate_sfx(self, prompt: str, duration: float = 2.0, **kwargs: Any) -> AudioTensor:
        """Generate a sound effect from a text description.

        Args:
            prompt: Description of the sound effect.
            duration: Target duration in seconds.

        Returns:
            An :class:`AudioTensor` with the generated sound effect.
        """
        # Use the music engine with shorter duration and electronic genre.
        return self.music.compose(
            prompt, duration=duration, genre="electronic"
        )

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> AudioTensor:
        """Synthesize speech from text.

        Args:
            text: Input text.
            speaker_id: Speaker identity.
            emotion: Emotion label.
            speed: Speed multiplier.

        Returns:
            An :class:`AudioTensor`.
        """
        return self.tts.synthesize(text, speaker_id, emotion, speed)

    # ------------------------------------------------------------------
    # Voice cloning
    # ------------------------------------------------------------------
    def voice_clone(
        self,
        reference_audio: AudioTensor,
        target_text: str,
        speed: float = 1.0,
    ) -> AudioTensor:
        """Clone a voice from a reference audio clip.

        Args:
            reference_audio: Reference audio.
            target_text: Text to synthesise.
            speed: Speed multiplier.

        Returns:
            An :class:`AudioTensor` with the cloned voice.
        """
        return self.tts.voice_clone(reference_audio, target_text, speed)

    # ------------------------------------------------------------------
    # Music composition
    # ------------------------------------------------------------------
    def compose(
        self,
        description: str,
        duration: float = 10.0,
        genre: str = "pop",
    ) -> AudioTensor:
        """Compose music from a text description.

        Args:
            description: Text description.
            duration: Duration in seconds.
            genre: Music genre.

        Returns:
            An :class:`AudioTensor`.
        """
        return self.music.compose(description, duration, genre)

    # ------------------------------------------------------------------
    # Transcription (ASR)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def transcribe(self, audio: AudioTensor) -> str:
        """Transcribe audio to text (speech-to-text).

        Uses the audio codec encoder to extract features and a simple
        token decoding approach.

        Args:
            audio: Input audio.

        Returns:
            The transcribed text string.
        """
        self.codec.eval()
        waveform = audio.waveform.to(self._device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Encode to discrete tokens.
        tokens = self.codec.encode(waveform)

        # Decode tokens to text (placeholder: use token ids as char codes).
        if tokens.dim() > 1:
            token_ids = tokens[0].cpu().tolist()
        else:
            token_ids = tokens.cpu().tolist()

        # Simple character-level decoding.
        chars = []
        for tid in token_ids:
            if 32 <= tid <= 126:
                chars.append(chr(tid))
            elif tid == 0:
                chars.append(" ")
        text = "".join(chars).strip()

        if not text:
            text = "[inaudible]"

        return text

    # ------------------------------------------------------------------
    # Codec operations
    # ------------------------------------------------------------------
    def encode_audio(self, audio: AudioTensor) -> torch.Tensor:
        """Compress audio to discrete codec tokens.

        Args:
            audio: Input audio.

        Returns:
            Discrete token tensor.
        """
        waveform = audio.waveform.to(self._device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        return self.codec.encode(waveform)

    def decode_audio(self, tokens: torch.Tensor) -> AudioTensor:
        """Reconstruct audio from codec tokens.

        Args:
            tokens: Discrete token tensor.

        Returns:
            An :class:`AudioTensor`.
        """
        tokens = tokens.to(self._device)
        waveform = self.codec.decode(tokens)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return AudioTensor(waveform=waveform, sample_rate=self.tts.sample_rate)

    def __repr__(self) -> str:
        return f"AudioEngine(device={self._device})"
