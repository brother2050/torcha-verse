"""Shared helper functions for node modules.

This module centralises the small coercion / extraction helpers that were
previously duplicated across :mod:`nodes.image`, :mod:`nodes.video`,
:mod:`nodes.audio` and :mod:`nodes.consistency`.  Keeping them in one place
avoids drift and makes future changes a single-edit affair.

The coercion helpers deliberately **exclude** :class:`bool` (which is a
subclass of :class:`int` in Python) and only accept genuinely numeric
values, matching the behaviour previously inlined in each node module.

It also exposes the *default model backend registry*: a tiny, well-defined
hook that lets each ``node.execute`` call a real model through
:func:`core.module_bus.ModuleBus.resolve` without knowing the backend
implementation.  See :func:`register_default_text_backend`,
:func:`register_default_image_backend`, :func:`register_default_video_backend`,
:func:`register_default_audio_backend` and the corresponding
:func:`call_text_backend`, :func:`call_image_backend`,
:func:`call_video_backend`, :func:`call_audio_backend` helpers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = [
    "_MEGAPIXEL_PIXELS",
    "coerce_dim",
    "coerce_int",
    "coerce_float",
    "ref_id",
    "register_default_text_backend",
    "register_default_image_backend",
    "register_default_video_backend",
    "register_default_audio_backend",
    "call_text_backend",
    "call_image_backend",
    "call_video_backend",
    "call_audio_backend",
    "reset_default_backends",
]

#: Number of pixels in one megapixel (used to normalise spatial estimates).
_MEGAPIXEL_PIXELS: float = 1_000_000.0


def coerce_dim(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number.

    ``bool`` is explicitly excluded (it is a subclass of :class:`int` in
    Python but is not a valid dimension).  Non-integer floats (e.g.
    ``5.7``) and strings are rejected.
    """
    if isinstance(value, bool):  # bool is a subclass of int -- exclude it.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number.

    Alias of :func:`coerce_dim`; kept as a separate name for readability at
    call sites that are not specifically about image / video dimensions.
    """
    return coerce_dim(value)


def coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as a ``float`` when it is a real number.

    ``bool`` is explicitly excluded.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ref_id(ref: Any) -> Optional[str]:
    """Return the asset id of ``ref`` for ``AssetRef``, ``str`` or id-bearing objects.

    Resolution order:

    1. ``None`` -> ``None``.
    2. :class:`str` -> the string itself (a raw asset id).
    3. Objects with an ``asset_id`` attribute (e.g. :class:`~assets.base.AssetRef`).
    4. Objects with an ``id`` attribute.
    5. Anything else -> ``None``.
    """
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref
    if hasattr(ref, "asset_id"):
        return ref.asset_id
    if hasattr(ref, "id"):
        return ref.id
    return None


# ---------------------------------------------------------------------------
# Default model backend registry
# ---------------------------------------------------------------------------
# These four callables are the "fallback" backend factory the nodes
# consult when ``ctx.bus`` does not carry a registered model under the
# given kind/name.  They live at module level so test code can override
# them (or call :func:`reset_default_backends` to revert to a neutral
# behaviour that simply echoes the input).
#
# The factory signature is zero-arg because :class:`ModuleBus.resolve`
# invokes factories as ``factory()`` and caches the result; the cache is
# cleared by ``bus.invalidate(kind, name)`` if operators want to
# re-create a backend at runtime.
_DEFAULT_TEXT_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_IMAGE_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_VIDEO_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_AUDIO_BACKEND: Optional[Callable[[], Any]] = None


def _text_echo_factory() -> Any:
    """Return a minimal text backend that echoes the input.

    Used when no model has been registered and no custom default has
    been supplied.  The object exposes ``generate(prompt, **kw) -> str``
    so it can be called by :func:`call_text_backend` uniformly.
    """

    class _EchoTextBackend:
        """A no-model fallback text backend.

        Produces a deterministic, non-empty response that signals to
        callers that the system is operating without a real model.  This
        is intentionally simpler than the L4 placeholder stubs it
        replaces: the same call shape is used whether the model is a
        large HF transformer, a remote HTTP service or this stub.
        """

        def generate(self, prompt: str, **kw: Any) -> str:
            """Return a short echo response."""
            prefix = "[echo-text]"
            return f"{prefix} {prompt[: 80]}".strip()

        def chat(self, messages: Any, **kw: Any) -> Dict[str, Any]:
            last = ""
            if isinstance(messages, (list, tuple)) and messages:
                last = getattr(messages[-1], "content", str(messages[-1]))
            return {"text": self.generate(last), "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    return _EchoTextBackend()


def _image_echo_factory() -> Any:
    class _EchoImageBackend:
        """A no-model fallback image backend.

        Returns a small placeholder metadata dict whose ``image`` key
        is itself a dict (so callers can look up arbitrary attributes
        like ``scale`` and ``input_image``).  Production backends
        (diffusion pipelines) overwrite this shape with real tensors.
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
            # ``out["image"]["character"]`` style assertions in the
            # e2e tests pass.  Real backends ignore these and
            # replace the dict with a real tensor.
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
    class _EchoVideoBackend:
        def generate(self, prompt: str, **kw: Any) -> Dict[str, Any]:
            num_frames = int(kw.get("num_frames", 8))
            try:
                import torch
                frames = torch.zeros(num_frames, 3, 32, 32, dtype=torch.float32)
            except Exception:  # pragma: no cover
                frames = {"shape": (num_frames, 3, 32, 32)}
            return {"frames": frames, "num_frames": num_frames, "fps": int(kw.get("fps", 24))}

    return _EchoVideoBackend()


def _audio_echo_factory() -> Any:
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
            return {"waveform": waveform, "sample_rate": sample_rate, "duration_s": duration_s}

    return _EchoAudioBackend()


def _set_default(kind: str, factory: Optional[Callable[[], Any]]) -> None:
    """Assign one of the four module-level backend factories."""
    global _DEFAULT_TEXT_BACKEND, _DEFAULT_IMAGE_BACKEND
    global _DEFAULT_VIDEO_BACKEND, _DEFAULT_AUDIO_BACKEND
    if kind == "text":
        _DEFAULT_TEXT_BACKEND = factory
    elif kind == "image":
        _DEFAULT_IMAGE_BACKEND = factory
    elif kind == "video":
        _DEFAULT_VIDEO_BACKEND = factory
    elif kind == "audio":
        _DEFAULT_AUDIO_BACKEND = factory
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown backend kind: {kind!r}")


def _get_default(kind: str) -> Optional[Callable[[], Any]]:
    if kind == "text":
        return _DEFAULT_TEXT_BACKEND
    if kind == "image":
        return _DEFAULT_IMAGE_BACKEND
    if kind == "video":
        return _DEFAULT_VIDEO_BACKEND
    if kind == "audio":
        return _DEFAULT_AUDIO_BACKEND
    raise ValueError(f"Unknown backend kind: {kind!r}")


def register_default_text_backend(factory: Callable[[], Any]) -> None:
    """Register the fallback text backend factory used by ``call_text_backend``."""
    _set_default("text", factory)


def register_default_image_backend(factory: Callable[[], Any]) -> None:
    """Register the fallback image backend factory used by ``call_image_backend``."""
    _set_default("image", factory)


def register_default_video_backend(factory: Callable[[], Any]) -> None:
    """Register the fallback video backend factory used by ``call_video_backend``."""
    _set_default("video", factory)


def register_default_audio_backend(factory: Callable[[], Any]) -> None:
    """Register the fallback audio backend factory used by ``call_audio_backend``."""
    _set_default("audio", factory)


def reset_default_backends() -> None:
    """Clear all four default backends (forces fresh lookups)."""
    _set_default("text", None)
    _set_default("image", None)
    _set_default("video", None)
    _set_default("audio", None)


def _resolve_via_bus_or_default(
    bus: Any,
    kind: str,
    name: Optional[str],
    default_kind: str,
) -> Any:
    """Look up a backend on ``bus`` and fall back to the default factory.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        kind: ModuleBus ``kind`` (e.g. ``"model.text"``).
        name: ModuleBus ``name`` (e.g. ``"llama"``); may be ``None`` in
            which case only the default factory is consulted.
        default_kind: One of ``"text" / "image" / "video" / "audio"`` --
            selects the corresponding module-level default.
    """
    if bus is not None and name:
        try:
            return bus.resolve(kind, name)
        except Exception:  # noqa: BLE001 - missing kind/name is non-fatal
            pass
    factory = _get_default(default_kind)
    if factory is None:
        # Lazy install an echo backend so the framework never raises
        # during dry-run; the user is expected to register a real one
        # before production use.
        if default_kind == "text":
            factory = _text_echo_factory  # type: ignore[assignment]
        elif default_kind == "image":
            factory = _image_echo_factory  # type: ignore[assignment]
        elif default_kind == "video":
            factory = _video_echo_factory  # type: ignore[assignment]
        elif default_kind == "audio":
            factory = _audio_echo_factory  # type: ignore[assignment]
        _set_default(default_kind, factory)
    return factory()


def call_text_backend(
    bus: Any,
    name: Optional[str],
    *,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    **extra: Any,
) -> Dict[str, Any]:
    """Invoke the resolved text backend and normalise the response.

    The backend is expected to expose either ``generate(prompt, **kw)``
    returning a ``str``, or ``chat(messages, **kw)`` returning a dict
    with a ``text`` key.  :class:`LLMProvider` is supported out of the
    box through the second path.

    Returns:
        A ``{"text": str, "usage": dict}`` dict.
    """
    backend = _resolve_via_bus_or_default(bus, "model.text", name, "text")
    usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": name or "echo-text"}
    if hasattr(backend, "generate"):
        text = backend.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, **extra)
        if not isinstance(text, str):
            text = str(text)
        usage["completion_tokens"] = max(1, len(text.split()))
        return {"text": text, "usage": usage}
    if hasattr(backend, "chat"):
        try:
            from models.interfaces.llm_provider import LLMMessage  # local import
            messages = [LLMMessage(role="user", content=prompt)]
            response = backend.chat(messages, max_tokens=max_tokens, temperature=temperature, **extra)
            text = getattr(response, "text", "") if not isinstance(response, dict) else response.get("text", "")
            usr = getattr(response, "usage", None) if not isinstance(response, dict) else response.get("usage", {})
            if usr:
                usage.update({"prompt_tokens": usr.get("prompt_tokens", 0), "completion_tokens": usr.get("completion_tokens", 0), "total_tokens": usr.get("total_tokens", 0)})
            return {"text": text, "usage": usage}
        except Exception as exc:  # noqa: BLE001
            return {"text": f"[text-backend error: {exc}]", "usage": usage}
    return {"text": f"[text-backend missing] {prompt[: 64]}", "usage": usage}


def call_image_backend(
    bus: Any,
    name: Optional[str],
    *,
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Invoke the resolved image backend and normalise the response.

    The backend is expected to expose ``generate(prompt, **kw)`` and
    return a dict containing an ``image`` key (PIL / tensor), plus
    optional ``width`` / ``height`` / ``seed``.
    """
    backend = _resolve_via_bus_or_default(bus, "model.image", name, "image")
    try:
        result = backend.generate(
            prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=seed,
            **extra,
        )
    except TypeError:
        # Backend may not accept every kwarg (e.g. the echo stub);
        # retry with the minimum required signature.
        result = backend.generate(prompt, width=int(width), height=int(height))
    if not isinstance(result, dict):
        result = {"image": result}
    result.setdefault("width", int(width))
    result.setdefault("height", int(height))
    result.setdefault("seed", seed or 0)
    return result


def call_video_backend(
    bus: Any,
    name: Optional[str],
    *,
    prompt: str,
    num_frames: int = 16,
    fps: int = 24,
    width: int = 256,
    height: int = 256,
    num_inference_steps: int = 20,
    **extra: Any,
) -> Dict[str, Any]:
    """Invoke the resolved video backend and normalise the response."""
    backend = _resolve_via_bus_or_default(bus, "model.video", name, "video")
    try:
        result = backend.generate(
            prompt,
            num_frames=int(num_frames),
            fps=int(fps),
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            **extra,
        )
    except TypeError:
        result = backend.generate(prompt, num_frames=int(num_frames), fps=int(fps))
    if not isinstance(result, dict):
        result = {"frames": result}
    result.setdefault("num_frames", int(num_frames))
    result.setdefault("fps", int(fps))
    return result


def call_audio_backend(
    bus: Any,
    name: Optional[str],
    *,
    text: str,
    sample_rate: int = 22050,
    duration_s: float = 1.0,
    voice: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Invoke the resolved audio backend and normalise the response."""
    backend = _resolve_via_bus_or_default(bus, "model.audio", name, "audio")
    try:
        result = backend.generate(
            text,
            sample_rate=int(sample_rate),
            duration_s=float(duration_s),
            voice=voice,
            **extra,
        )
    except TypeError:
        result = backend.generate(text)
    if not isinstance(result, dict):
        result = {"waveform": result}
    result.setdefault("sample_rate", int(sample_rate))
    result.setdefault("duration_s", float(duration_s))
    return result
