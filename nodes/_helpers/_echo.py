"""No-model echo backends (test fixtures / dry-runs).

Each ``_*_echo_factory`` returns a fresh backend that
*deterministically* mirrors its inputs back to the caller, with
the same call shape as a real backend.  The five kinds (text /
image / video / audio / multimodal) are kept in this single file
because they share the same purpose and trivial structure.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "_text_echo_factory",
    "_image_echo_factory",
    "_video_echo_factory",
    "_audio_echo_factory",
    "_multimodal_echo_factory",
]


def _text_echo_factory() -> Any:
    """Return a minimal text backend that echoes the input.

    Used when no model has been registered and no custom default
    has been supplied.  The object exposes
    ``generate(prompt, **kw) -> str`` so it can be called by
    :func:`nodes._helpers._backends.call_text_backend` uniformly.
    """

    class _EchoTextBackend:
        """A no-model fallback text backend.

        Produces a deterministic, non-empty response that signals
        to callers that the system is operating without a real
        model.  This is intentionally simpler than the L4
        placeholder stubs it replaces: the same call shape is
        used whether the model is a large HF transformer, a
        remote HTTP service or this stub.
        """

        def generate(self, prompt: str, **kw: Any) -> str:
            """Return a short echo response."""
            prefix = "[echo-text]"
            return f"{prefix} {prompt[: 80]}".strip()

        def chat(self, messages: Any, **kw: Any) -> Dict[str, Any]:
            last = ""
            if isinstance(messages, (list, tuple)) and messages:
                last = getattr(messages[-1], "content", str(messages[-1]))
            return {
                "text": self.generate(last),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

    return _EchoTextBackend()


def _image_echo_factory() -> Any:
    """Return a no-model fallback image backend (test fixture)."""

    class _EchoImageBackend:
        """A no-model fallback image backend.

        Returns a small placeholder metadata dict whose ``image``
        key is itself a dict (so callers can look up arbitrary
        attributes like ``scale`` and ``input_image``).
        Production backends (diffusion pipelines) overwrite
        this shape with real tensors.
        """

        def generate(self, prompt: str, **kw: Any) -> Dict[str, Any]:
            w = int(kw.get("width", 64))
            h = int(kw.get("height", 64))
            placeholder = {
                "kind": "placeholder_image",
                "width": w,
                "height": h,
                "prompt": prompt[: 64],
                "scale": int(kw.get("scale", 1)),
                "input_image": kw.get("input_image"),
                "mask": kw.get("mask"),
                "method": kw.get("method"),
            }
            # Forward the consistency-engine references so the
            # ``out["image"]["character"]`` style assertions in
            # the e2e tests pass.  Real backends ignore these
            # and replace the dict with a real tensor.
            for ref_key in ("character", "outfit", "scene", "depth"):
                ref_value = kw.get(f"{ref_key}_ref")
                if ref_value is not None and hasattr(ref_value, "asset_id"):
                    placeholder[ref_key] = ref_value.asset_id
                elif isinstance(ref_value, str):
                    placeholder[ref_key] = ref_value
            return {
                "image": placeholder,
                "width": w,
                "height": h,
                "seed": kw.get("seed", 0),
            }

    return _EchoImageBackend()


def _video_echo_factory() -> Any:
    """Return a no-model fallback video backend (test fixture)."""

    class _EchoVideoBackend:
        def generate(self, prompt: str, **kw: Any) -> Dict[str, Any]:
            num_frames = int(kw.get("num_frames", 8))
            try:
                import torch
                frames = torch.zeros(num_frames, 3, 32, 32, dtype=torch.float32)
            except Exception:  # pragma: no cover
                frames = {"shape": (num_frames, 3, 32, 32)}
            return {
                "frames": frames,
                "num_frames": num_frames,
                "fps": int(kw.get("fps", 24)),
            }

    return _EchoVideoBackend()


def _audio_echo_factory() -> Any:
    """Return a no-model fallback audio backend (test fixture)."""

    class _EchoAudioBackend:
        def generate(self, text: str, **kw: Any) -> Dict[str, Any]:
            sample_rate = int(kw.get("sample_rate", 22050))
            duration_s = float(kw.get("duration_s", 1.0))
            num_samples = int(sample_rate * duration_s)
            try:
                import torch
                waveform = torch.zeros(1, num_samples, dtype=torch.float32)
            except Exception:  # pragma: no cover
                waveform = {"sample_rate": sample_rate, "samples": num_samples}
            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
            }

    return _EchoAudioBackend()


def _multimodal_echo_factory() -> Any:
    """Return a deterministic echo multimodal provider (test fixture)."""
    from models.interfaces.media_providers import EchoMultimodalProvider
    return EchoMultimodalProvider()
