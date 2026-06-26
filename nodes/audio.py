"""Audio generation nodes for the TorchaVerse L4 capability layer.

This module decomposes the v0.1.0 ``audio_engine.py`` "god class" (703
lines) into two single-responsibility nodes:

* :class:`AudioTTSNode` (``audio_tts``) -- text-to-speech synthesis
  parameterised by voice, speed and optional emotion.
* :class:`AudioMusicNode` (``audio_music``) -- music generation from a
  text prompt and a target duration.

Both nodes carry a real :meth:`validate_inputs` (speed / duration
ranges, non-empty text / prompt) and a real :meth:`estimate_resources`
(time scales with the requested duration).  Their :meth:`execute`
bodies are placeholder stubs returning deterministic mock data.

Media types (``audio``) are typed as :data:`typing.Any` so that this
module stays free of heavy imports (``torch`` / ``librosa``); the
concrete tensor representation is decided by the backend at execution
time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseNode, NodeContext, NodeSpec, register_node
from ._helpers import coerce_float as _coerce_float

__all__ = ["AudioTTSNode", "AudioMusicNode"]


# ---------------------------------------------------------------------------
# Shared constants (audio-specific estimation coefficients)
# ---------------------------------------------------------------------------
#: Default output sample rate (Hz) when none is configured.
_DEFAULT_SAMPLE_RATE: int = 22050
#: Minimum supported speech speed multiplier.
_AUDIO_MIN_SPEED: float = 0.25
#: Maximum supported speech speed multiplier.
_AUDIO_MAX_SPEED: float = 4.0
#: Minimum supported duration (seconds).
_AUDIO_MIN_DURATION_S: float = 0.1
#: Maximum supported duration (seconds).
_AUDIO_MAX_DURATION_S: float = 3600.0
#: Wall-clock seconds of compute per second of generated audio (TTS).
_TTS_TIME_PER_AUDIO_S: float = 0.5
#: Wall-clock seconds of compute per second of generated audio (music).
_MUSIC_TIME_PER_AUDIO_S: float = 2.0
#: VRAM (GB) for the TTS model weights.
_TTS_MODEL_VRAM_GB: float = 1.5
#: VRAM (GB) for the music model weights.
_MUSIC_MODEL_VRAM_GB: float = 3.0
#: Host RAM (GB) per second of generated audio.
_AUDIO_RAM_PER_SECOND_GB: float = 0.005


# ---------------------------------------------------------------------------
# AudioTTSNode
# ---------------------------------------------------------------------------
@register_node("audio_tts")
class AudioTTSNode(BaseNode):
    """Text-to-speech synthesis node (``audio_tts``).

    Synthesises speech from ``text`` using the given ``voice``.

    Inputs:
        text: The text to synthesise (required).
        voice: Voice identifier (e.g. ``"en-US-AriaNeural"``).
        speed: Speech speed multiplier in ``[0.25, 4.0]``.
        emotion: Optional emotion tag (e.g. ``"cheerful"``).

    Outputs:
        audio: The synthesised audio waveform (tensor).
        sample_rate: The output sample rate in Hz.
    """

    spec = NodeSpec(
        type="audio_tts",
        name="Audio TTS",
        description="Synthesise speech from text.",
        inputs={
            "text": "TEXT",
            "voice": "TEXT",
            "speed": "FLOAT",
            "emotion": "Optional[TEXT]",
        },
        outputs={
            "audio": "AUDIO",
            "sample_rate": "INT",
        },
        tags=["audio", "generation", "tts", "speech"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate TTS inputs.

        Extends the base checks with:

        * ``speed`` in ``[0.25, 4.0]``.
        * ``text`` non-empty.
        * ``voice`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        speed = _coerce_float(inputs.get("speed"))
        if speed is not None and not (
            _AUDIO_MIN_SPEED <= speed <= _AUDIO_MAX_SPEED
        ):
            errors.append(
                "Input 'speed' for node 'audio_tts' must be in "
                "[{}, {}], got {}.".format(
                    _AUDIO_MIN_SPEED, _AUDIO_MAX_SPEED, speed
                )
            )

        text = inputs.get("text")
        if isinstance(text, str) and not text.strip():
            errors.append(
                "Input 'text' for node 'audio_tts' must be a non-empty "
                "string."
            )

        voice = inputs.get("voice")
        if isinstance(voice, str) and not voice.strip():
            errors.append(
                "Input 'voice' for node 'audio_tts' must be a non-empty "
                "string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a TTS run.

        Time scales with the length of the input text (approximated by
        word count) divided by the speech speed.
        """
        text = inputs.get("text", "")
        text = text if isinstance(text, str) else ""
        _sp = _coerce_float(inputs.get("speed"))
        speed = _sp if _sp is not None else 1.0
        speed = max(_AUDIO_MIN_SPEED, min(_AUDIO_MAX_SPEED, speed))

        word_count = max(1, len(text.split()))
        # Roughly 0.5 seconds of speech per word at 1x speed.
        audio_seconds = (word_count * 0.5) / speed

        vram_gb = _TTS_MODEL_VRAM_GB
        ram_gb = audio_seconds * _AUDIO_RAM_PER_SECOND_GB
        time_s = audio_seconds * _TTS_TIME_PER_AUDIO_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Synthesise speech (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``text``, ``voice``, ``speed``, ``emotion``.

        Returns:
            A dict with ``audio`` and ``sample_rate``.
        """
        text = str(inputs.get("text", ""))
        voice = str(inputs.get("voice", ""))
        _sp = _coerce_float(inputs.get("speed"))
        speed = _sp if _sp is not None else 1.0
        emotion = inputs.get("emotion")
        sample_rate = int(
            ctx.config.get("default_tts_sample_rate", _DEFAULT_SAMPLE_RATE)
        )
        model = ctx.config.get("default_tts_model")

        ctx.logger.debug(
            "audio_tts run_id=%s voice=%s speed=%.2f emotion=%s",
            ctx.run_id, voice, speed, emotion,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.audio_tts",
                action="synthesise",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "voice": voice,
                    "speed": speed,
                    "emotion": emotion,
                    "sample_rate": sample_rate,
                },
                severity="info",
            )

        from ._helpers import call_audio_backend

        # ``speed`` is applied post hoc by the audio backend if it
        # supports the keyword; otherwise it is accepted silently.
        result = call_audio_backend(
            ctx.bus,
            model,
            text=text,
            sample_rate=sample_rate,
            duration_s=max(0.1, len(text.split()) * 0.3 / max(0.1, float(speed))),
            voice=voice,
            speed=float(speed),
            emotion=emotion,
        )
        return {"audio": result, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# AudioMusicNode
# ---------------------------------------------------------------------------
@register_node("audio_music")
class AudioMusicNode(BaseNode):
    """Music generation node (``audio_music``).

    Generates a music clip from a text prompt with a target duration.

    Inputs:
        prompt: Text prompt describing the desired music (required).
        duration: Target duration in seconds.

    Outputs:
        audio: The generated audio waveform (tensor).
    """

    spec = NodeSpec(
        type="audio_music",
        name="Audio Music",
        description="Generate a music clip from a text prompt.",
        inputs={
            "prompt": "PROMPT",
            "duration": "FLOAT",
        },
        outputs={
            "audio": "AUDIO",
        },
        tags=["audio", "generation", "music"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate music-generation inputs.

        Extends the base checks with:

        * ``duration`` in ``[0.1, 3600]``.
        * ``prompt`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        duration = _coerce_float(inputs.get("duration"))
        if duration is not None and not (
            _AUDIO_MIN_DURATION_S <= duration <= _AUDIO_MAX_DURATION_S
        ):
            errors.append(
                "Input 'duration' for node 'audio_music' must be in "
                "[{}, {}], got {}.".format(
                    _AUDIO_MIN_DURATION_S, _AUDIO_MAX_DURATION_S, duration
                )
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'audio_music' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a music run.

        Time scales with the requested ``duration``.
        """
        duration = _coerce_float(inputs.get("duration")) or 0.0
        duration = max(
            _AUDIO_MIN_DURATION_S, min(_AUDIO_MAX_DURATION_S, duration)
        )

        vram_gb = _MUSIC_MODEL_VRAM_GB
        ram_gb = duration * _AUDIO_RAM_PER_SECOND_GB
        time_s = duration * _MUSIC_TIME_PER_AUDIO_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a music clip (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``prompt``, ``duration``.

        Returns:
            A dict with ``audio``.
        """
        prompt = str(inputs.get("prompt", ""))
        duration = _coerce_float(inputs.get("duration")) or 0.0
        model = ctx.config.get("default_music_model")

        ctx.logger.debug(
            "audio_music run_id=%s duration=%.2f",
            ctx.run_id, duration,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.audio_music",
                action="generate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "duration": duration,
                    "prompt": prompt[: 64],
                },
                severity="info",
            )

        from ._helpers import (
            call_audio_backend, call_music_backend,
        )

        # F-12: real music DiT (mel) + HiFi-GAN vocoder.  We always
        # fall back to the audio backend so the result payload
        # remains populated when the music backend is unavailable.
        sample_rate = int(inputs.get("sample_rate", 22050))
        num_inference_steps = int(inputs.get("steps", 30))
        music_result = call_music_backend(
            ctx.bus, model or "music_dit",
            prompt=prompt,
            duration_s=float(duration),
            sample_rate=sample_rate,
            num_inference_steps=num_inference_steps,
        )
        result = call_audio_backend(
            ctx.bus,
            model,
            text=prompt,
            sample_rate=sample_rate,
            duration_s=float(duration),
        )
        if isinstance(music_result, dict) and music_result.get(
            "backend", ""
        ) != "placeholder":
            result["music_backend"] = music_result.get("backend")
            if "mel" in music_result:
                result["mel"] = music_result["mel"]
            if "duration_s" in music_result:
                result["duration_s"] = music_result["duration_s"]
        return {"audio": result}
