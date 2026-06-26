"""Default model backend registry + ``call_*_backend`` helpers (v0.6.x).

This module owns the "fallback backend factory" the nodes
consult when ``ctx.bus`` does not carry a registered model under
the given kind/name.  The five factory slots (text / image /
video / audio / multimodal) live at module level so test code
can override them or call :func:`reset_default_backends` to
revert to a neutral behaviour that simply echoes the input.

Public surface (re-exported by :mod:`nodes._helpers.__init__`):

* :func:`register_default_text_backend`
* :func:`register_default_image_backend`
* :func:`register_default_video_backend`
* :func:`register_default_audio_backend`
* :func:`register_default_multimodal_backend`
* :func:`call_text_backend`
* :func:`call_image_backend`
* :func:`call_video_backend`
* :func:`call_audio_backend`
* :func:`call_multimodal_backend`
* :func:`reset_default_backends`

The factory signature is zero-arg because
:class:`ModuleBus.resolve` invokes factories as ``factory()`` and
caches the result; the cache is cleared by
``bus.invalidate(kind, name)`` if operators want to re-create a
backend at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from infrastructure.logger import get_logger

from ._echo import (
    _audio_echo_factory,
    _image_echo_factory,
    _multimodal_echo_factory,
    _text_echo_factory,
    _video_echo_factory,
)

__all__ = [
    "register_default_text_backend",
    "register_default_image_backend",
    "register_default_video_backend",
    "register_default_audio_backend",
    "register_default_multimodal_backend",
    "call_text_backend",
    "call_image_backend",
    "call_video_backend",
    "call_audio_backend",
    "call_multimodal_backend",
    "reset_default_backends",
]


_logger = get_logger("nodes._helpers._backends")

# ---------------------------------------------------------------------------
# Module-level default factory slots
# ---------------------------------------------------------------------------
_DEFAULT_TEXT_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_IMAGE_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_VIDEO_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_AUDIO_BACKEND: Optional[Callable[[], Any]] = None
_DEFAULT_MULTIMODAL_BACKEND: Optional[Callable[[], Any]] = None


def _set_default(kind: str, factory: Optional[Callable[[], Any]]) -> None:
    """Assign one of the five module-level backend factories."""
    global _DEFAULT_TEXT_BACKEND, _DEFAULT_IMAGE_BACKEND
    global _DEFAULT_VIDEO_BACKEND, _DEFAULT_AUDIO_BACKEND
    global _DEFAULT_MULTIMODAL_BACKEND
    if kind == "text":
        _DEFAULT_TEXT_BACKEND = factory
    elif kind == "image":
        _DEFAULT_IMAGE_BACKEND = factory
    elif kind == "video":
        _DEFAULT_VIDEO_BACKEND = factory
    elif kind == "audio":
        _DEFAULT_AUDIO_BACKEND = factory
    elif kind == "multimodal":
        _DEFAULT_MULTIMODAL_BACKEND = factory
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown backend kind: {kind!r}")


# ---------------------------------------------------------------------------
# v0.4.x P0 multi-modal: real local backend factories
# ---------------------------------------------------------------------------
# These factories wire the project-owned multi-modal providers
# (see :mod:`models.providers.local_image` etc.) into the node
# registry.  They are deliberately *opt-in* -- the default
# behaviour is still the echo backend (which never raises) and
# the real backend is only installed when one of
# ``register_default_image_backend(...)`` etc. is called from an
# example / script that explicitly asks for the v0.4.x P0 path.
def _local_image_factory() -> Any:
    """Factory returning a fresh :class:`LocalTorchImageProvider`.

    The provider is built **on demand** (not at registration
    time) so that ``register_default_image_backend`` does not
    pay the ~5M-param build cost until the backend is actually
    invoked.  The provider is created with the TINY preset to
    keep the default cheap; callers that need a different
    preset should construct their own
    :class:`LocalTorchImageProvider` and register *that* as the
    factory.
    """
    try:
        from models.providers import LocalTorchImageProvider
    except Exception:  # noqa: BLE001 - fall back to echo
        return _image_echo_factory()
    return LocalTorchImageProvider.from_random()


def _local_video_factory() -> Any:
    """Factory returning a fresh :class:`LocalTorchVideoProvider`."""
    try:
        from models.providers import LocalTorchVideoProvider
    except Exception:  # noqa: BLE001
        return _video_echo_factory()
    return LocalTorchVideoProvider.from_random()


def _local_audio_factory() -> Any:
    """Factory returning a fresh :class:`LocalTorchAudioProvider`."""
    try:
        from models.providers import LocalTorchAudioProvider
    except Exception:  # noqa: BLE001
        return _audio_echo_factory()
    return LocalTorchAudioProvider.from_random()


def _local_multimodal_factory() -> Any:
    """Factory returning a fresh :class:`LocalTorchMultimodalProvider`."""
    try:
        from models.providers import LocalTorchMultimodalProvider
    except Exception:  # noqa: BLE001
        return _multimodal_echo_factory()
    return LocalTorchMultimodalProvider.from_random()


def _local_text_factory() -> Any:
    """Factory returning a fresh :class:`LocalTorchTextProvider`."""
    try:
        from models.providers import LocalTorchTextProvider, TINY_CONFIG
    except Exception:  # noqa: BLE001
        return _text_echo_factory()
    return LocalTorchTextProvider.from_random(TINY_CONFIG)


# ---------------------------------------------------------------------------
# Public registration helpers
# ---------------------------------------------------------------------------
def register_default_image_backend(
    factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Register the fallback image backend factory used by ``call_image_backend``.

    When ``factory`` is ``None`` (the default) a v0.4.x P0
    :class:`LocalTorchImageProvider` is registered; pass an
    explicit ``factory`` to plug in a different backend (e.g.
    one wrapping a remote service).  Pass the explicit
    :data:`_image_echo_factory` to revert to the echo behaviour.
    """
    if factory is None:
        factory = _local_image_factory
    _set_default("image", factory)


def register_default_video_backend(
    factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Register the fallback video backend factory used by ``call_video_backend``."""
    if factory is None:
        factory = _local_video_factory
    _set_default("video", factory)


def register_default_audio_backend(
    factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Register the fallback audio backend factory used by ``call_audio_backend``."""
    if factory is None:
        factory = _local_audio_factory
    _set_default("audio", factory)


def register_default_multimodal_backend(
    factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Register the fallback multimodal backend factory used by ``call_multimodal_backend``."""
    if factory is None:
        factory = _local_multimodal_factory
    _set_default("multimodal", factory)


def register_default_text_backend(
    factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Register the fallback text backend factory used by ``call_text_backend``.

    Preserved for compatibility with v0.4.0 callers; the v0.4.x
    P0 default is the project-owned
    :class:`LocalTorchTextProvider` instead of the echo stub.
    """
    if factory is None:
        factory = _local_text_factory
    _set_default("text", factory)


def _get_default(kind: str) -> Optional[Callable[[], Any]]:
    if kind == "text":
        return _DEFAULT_TEXT_BACKEND
    if kind == "image":
        return _DEFAULT_IMAGE_BACKEND
    if kind == "video":
        return _DEFAULT_VIDEO_BACKEND
    if kind == "audio":
        return _DEFAULT_AUDIO_BACKEND
    if kind == "multimodal":
        return _DEFAULT_MULTIMODAL_BACKEND
    raise ValueError(f"Unknown backend kind: {kind!r}")


def reset_default_backends() -> None:
    """Clear all five default backends (forces fresh lookups)."""
    _set_default("text", None)
    _set_default("image", None)
    _set_default("video", None)
    _set_default("audio", None)
    _set_default("multimodal", None)


# ---------------------------------------------------------------------------
# Bus-or-default resolution
# ---------------------------------------------------------------------------
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
        name: ModuleBus ``name`` (e.g. ``"llama"``); may be
            ``None`` in which case only the default factory is
            consulted.
        default_kind: One of ``"text" / "image" / "video" / "audio" /
            "multimodal"`` -- selects the corresponding module-level
            default.
    """
    if bus is not None and name:
        try:
            return bus.resolve(kind, name)
        except Exception as exc:  # noqa: BLE001 - missing kind/name is non-fatal
            _logger.debug(
                "ModuleBus.resolve failed for kind=%s name=%s: %s",
                kind, name, exc,
            )
    factory = _get_default(default_kind)
    if factory is None:
        # Lazy install an echo backend so the framework never
        # raises during dry-run; the user is expected to
        # register a real one before production use.
        if default_kind == "text":
            factory = _text_echo_factory  # type: ignore[assignment]
        elif default_kind == "image":
            factory = _image_echo_factory  # type: ignore[assignment]
        elif default_kind == "video":
            factory = _video_echo_factory  # type: ignore[assignment]
        elif default_kind == "audio":
            factory = _audio_echo_factory  # type: ignore[assignment]
        elif default_kind == "multimodal":
            factory = _multimodal_echo_factory  # type: ignore[assignment]
        _set_default(default_kind, factory)
    return factory()


# ---------------------------------------------------------------------------
# Public call helpers
# ---------------------------------------------------------------------------
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

    The backend is expected to expose either
    ``generate(prompt, **kw)`` returning a ``str``, or
    ``chat(messages, **kw)`` returning a dict with a ``text``
    key.  :class:`LLMProvider` is supported out of the box
    through the second path.

    Returns:
        A ``{"text": str, "usage": dict}`` dict.
    """
    backend = _resolve_via_bus_or_default(bus, "model.text", name, "text")
    usage: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": name or "echo-text",
    }
    if hasattr(backend, "generate"):
        text = backend.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        if not isinstance(text, str):
            text = str(text)
        usage["completion_tokens"] = max(1, len(text.split()))
        return {"text": text, "usage": usage}
    if hasattr(backend, "chat"):
        try:
            from models.interfaces.llm_provider import LLMMessage  # local import
            messages = [LLMMessage(role="user", content=prompt)]
            response = backend.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            text = (
                getattr(response, "text", "")
                if not isinstance(response, dict)
                else response.get("text", "")
            )
            usr = (
                getattr(response, "usage", None)
                if not isinstance(response, dict)
                else response.get("usage", {})
            )
            if usr:
                usage.update({
                    "prompt_tokens": usr.get("prompt_tokens", 0),
                    "completion_tokens": usr.get("completion_tokens", 0),
                    "total_tokens": usr.get("total_tokens", 0),
                })
            return {"text": text, "usage": usage}
        except Exception as exc:  # noqa: BLE001
            return {"text": f"[text-backend error: {exc}]", "usage": usage}
    return {
        "text": f"[text-backend missing] {prompt[: 64]}",
        "usage": usage,
    }


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

    The backend is expected to expose ``generate(prompt, **kw)``
    and return a dict containing an ``image`` key (PIL / tensor),
    plus optional ``width`` / ``height`` / ``seed``.
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
        result = backend.generate(
            prompt, width=int(width), height=int(height)
        )
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
        result = backend.generate(
            prompt, num_frames=int(num_frames), fps=int(fps)
        )
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


def call_multimodal_backend(
    bus: Any,
    name: Optional[str],
    *,
    input: Any,
    **extra: Any,
) -> Dict[str, Any]:
    """Invoke the resolved multimodal backend and normalise the response.

    ``input`` may be a plain string (text-only), a :class:`dict`
    with ``"text"`` / ``"image"`` / ``"audio"`` keys, or a
    list of mixed-modality items -- the
    :class:`MultimodalProvider` protocol handles all three.
    """
    backend = _resolve_via_bus_or_default(
        bus, "model.multimodal", name, "multimodal"
    )
    try:
        result = backend.generate(input, **extra)
    except TypeError:
        # Some echo providers take just positional text; fall back.
        if isinstance(input, dict) and "text" in input:
            result = backend.generate(input["text"])
        else:
            result = backend.generate(str(input))
    if not isinstance(result, dict):
        result = {"text": str(result)}
    return result
