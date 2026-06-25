"""Media-provider interfaces for the v0.4.x P0 multi-modal milestone.

This package defines the *contracts* that any image / audio / video /
multimodal backend must satisfy to be plugged into the framework's
nodes, agents and serving endpoints.  The text counterpart
(:class:`models.interfaces.llm_provider.LLMProvider`) was created
during the v0.4.0 P0 text milestone; these four are its multi-modal
siblings.

Each Protocol is intentionally minimal: it exposes the **call
signature** the L4 nodes / examples actually use, and nothing more.
Concretely, the contract is::

    image.generate(prompt, **kwargs) -> dict        # {"image": tensor, ...}
    audio.generate(text,   **kwargs) -> dict        # {"waveform": tensor, ...}
    video.generate(prompt, **kwargs) -> dict        # {"frames": tensor, ...}
    omni.generate(input,   **kwargs) -> dict        # {"text": str, "audio": ..., ...}

Reference implementations live in
:mod:`models.providers.local_image`,
:mod:`models.providers.local_audio`,
:mod:`models.providers.local_video`,
:mod:`models.providers.local_multimodal`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, Union, runtime_checkable

__all__ = [
    "ImageProvider",
    "AudioProvider",
    "VideoProvider",
    "MultimodalProvider",
    "EchoImageProvider",
    "EchoAudioProvider",
    "EchoVideoProvider",
    "EchoMultimodalProvider",
]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class ImageProvider(Protocol):
    """Contract that any image-generation backend must satisfy.

    The single method is :meth:`generate` which takes a text prompt
    plus a bag of model-specific kwargs (e.g. ``width`` / ``height``
    / ``steps`` / ``guidance_scale`` / ``seed``) and returns a
    dictionary.  The dictionary MUST contain an ``"image"`` key whose
    value is a ``torch.Tensor`` of shape ``(3, H, W)`` in
    ``[0, 1]`` range (so the v0.4.x P0 nodes can do
    ``out["image"].clamp(0, 1)`` and write to disk).  The contract
    intentionally permits additional keys (``"seed"``, ``"latents"``,
    ``"metadata"``) for downstream consumers.
    """

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class AudioProvider(Protocol):
    """Contract that any audio-generation backend must satisfy.

    The :meth:`generate` method takes a text prompt plus kwargs
    (``sample_rate``, ``duration_s``, ``steps``, ``temperature`` ...)
    and returns a dict with at least a ``"waveform"`` key (a
    ``torch.Tensor`` of shape ``(1, num_samples)``) and a
    ``"sample_rate"`` key (int).
    """

    def generate(
        self,
        text: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class VideoProvider(Protocol):
    """Contract that any video-generation backend must satisfy.

    The :meth:`generate` method takes a text prompt plus kwargs
    (``num_frames``, ``fps``, ``height`` / ``width``, ``steps`` ...)
    and returns a dict with at least a ``"frames"`` key (a
    ``torch.Tensor`` of shape ``(T, 3, H, W)`` in ``[0, 1]``) and a
    ``"fps"`` key (int).
    """

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class MultimodalProvider(Protocol):
    """Contract that any omni-modal backend must satisfy.

    The :meth:`generate` method takes a heterogeneous input
    (text + audio + image, any combination) plus kwargs and returns
    a dict that may include ``"text"``, ``"audio"``, ``"image"``
    keys.  The set of present keys reflects what the model emitted
    for the given input.
    """

    def generate(
        self,
        input: Union[str, Dict[str, Any], Sequence[Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Reference echo implementations (test fixtures + "no model" fallback)
# ---------------------------------------------------------------------------
class _EchoBase:
    """Common helpers for the echo providers (deterministic, non-empty)."""

    _PROMPT_PREFIXES = {
        "image": "[echo-image]",
        "audio": "[echo-audio]",
        "video": "[echo-video]",
        "multimodal": "[echo-omni]",
    }

    @staticmethod
    def _truncate(text: str, limit: int = 64) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."


class EchoImageProvider(_EchoBase):
    """Deterministic image backend that returns a small placeholder tensor.

    Used in tests and as the "no model" fallback when no
    real provider has been registered.  Always returns a
    ``(3, H, W)`` zero tensor of size ``max(width, 16)`` ×
    ``max(height, 16)`` plus the metadata that the v0.4.x P0
    nodes look up (``seed``, ``prompt``).
    """

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        import torch
        w = int(kwargs.get("width", 16))
        h = int(kwargs.get("height", 16))
        return {
            "image": torch.zeros(3, h, w, dtype=torch.float32),
            "width": w,
            "height": h,
            "seed": int(kwargs.get("seed", 0)),
            "prompt": self._truncate(prompt),
        }


class EchoAudioProvider(_EchoBase):
    """Deterministic audio backend that returns a silent waveform."""

    def generate(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        import torch
        sample_rate = int(kwargs.get("sample_rate", 16000))
        duration_s = float(kwargs.get("duration_s", 0.5))
        n = max(int(sample_rate * duration_s), 1)
        return {
            "waveform": torch.zeros(1, n, dtype=torch.float32),
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "text": self._truncate(text),
        }


class EchoVideoProvider(_EchoBase):
    """Deterministic video backend that returns a black frame tensor."""

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        import torch
        t = int(kwargs.get("num_frames", 4))
        h = int(kwargs.get("height", 16))
        w = int(kwargs.get("width", 16))
        return {
            "frames": torch.zeros(t, 3, h, w, dtype=torch.float32),
            "num_frames": t,
            "fps": int(kwargs.get("fps", 8)),
            "prompt": self._truncate(prompt),
        }


class EchoMultimodalProvider(_EchoBase):
    """Deterministic omni-modal backend that just echoes the input text."""

    def generate(
        self,
        input: Union[str, Dict[str, Any], Sequence[Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(input, str):
            text = input
        elif isinstance(input, dict):
            text = str(input.get("text", ""))
        else:
            text = str(next(iter(input), "")) if input else ""
        return {
            "text": f"{self._PROMPT_PREFIXES['multimodal']} {self._truncate(text)}",
        }
