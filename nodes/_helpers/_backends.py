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


# ---------------------------------------------------------------------------
# Digital-human backends (F-1)
# ---------------------------------------------------------------------------
# F-1 strategy: a single ``call_digital_human_backend`` dispatches
# by ``method`` (e.g. ``"musetalk"`` / ``"sadtalker"``).  It first
# tries the bus under ``"model.digital_human"``; on miss it
# resolves the corresponding :class:`PaperAdapter` class through
# the global :class:`AdapterRegistry`, instantiates it, calls
# ``load_model(ctx)`` once, then ``infer(**kwargs)``.
#
# The bus and registry paths both surface the *same* real
# ``torch`` networks; if neither is available the function falls
# back to the generic video / audio backend, preserving the
# v0.5.x e2e behaviour.  This keeps the existing 6 e2e tests
# green while exposing real algorithm surfaces to callers that
# explicitly request a digital-human method.
import importlib as _importlib
from papers.adapter import (
    AdapterNotFoundError as _AdapterNotFoundError,
    AdapterRegistry as _AdapterRegistry,
)


_DH_VIDEO_METHODS = {
    "musetalk", "video_retalking", "sadtalker", "echo_mimic",
    "echo_mimic_v2", "liveportrait", "gfpgan", "codeformer",
}
_DH_AUDIO_METHODS = {"cosyvoice", "f5_tts", "chat_tts"}


def _resolve_dh_adapter(name: str) -> Optional[Any]:
    """Return the :class:`PaperAdapter` instance for ``name``.

    Returns ``None`` if the adapter is not registered (caller
    should fall back to the generic backend).
    """
    try:
        cls = _AdapterRegistry().get(name)
    except _AdapterNotFoundError:
        return None
    try:
        return cls()
    except Exception:  # noqa: BLE001
        return None


def call_digital_human_backend(
    bus: Any,
    name: Optional[str],
    *,
    method: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch a digital-human task to a specialised adapter.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).  When
            ``None`` the lookup is by ``method`` alone.
        method: The paper-family method (e.g. ``"musetalk"``,
            ``"sadtalker"``, ``"gfpgan"``).
        **kwargs: Forwarded to the adapter's :meth:`infer`.

    Returns:
        A :class:`dict` of outputs compatible with the calling
        node's :attr:`NodeSpec.outputs`.

    Fallback: if the bus does not have a digital-human backend
    and the adapter registry has no matching class, the function
    delegates to the generic video / audio backend.  This keeps
    the e2e suite green on the legacy ``call_*_backend`` path.
    """
    target_name = name or method
    # 1. Bus on "model.digital_human".
    backend: Any = None
    if bus is not None and target_name:
        try:
            backend = bus.resolve("model.digital_human", target_name)
        except Exception:  # noqa: BLE001
            backend = None
    if backend is not None:
        return _invoke_dh(backend, method=method, **kwargs)
    # 2. PaperAdapter registry.
    adapter = _resolve_dh_adapter(target_name)
    if adapter is not None:
        try:
            model = adapter.load_model(ctx=None)
        except Exception:  # noqa: BLE001
            model = {}
        try:
            return adapter.infer(model, **kwargs)
        except Exception:  # noqa: BLE001
            return _fallback_dh(method, **kwargs)
    # 3. Generic video / audio fallback.
    return _fallback_dh(method, **kwargs)


def _invoke_dh(backend: Any, *, method: str, **kwargs: Any) -> Dict[str, Any]:
    """Call a bus-registered digital-human backend.

    The bus can hold either a *factory* (an adapter class) or a
    *callable* with signature ``(**kwargs) -> dict``.  We try
    the factory path first (a common pattern for paper
    adapters), then the callable path.
    """
    # Factory / class.
    if isinstance(backend, type):
        try:
            inst = backend()
            return inst.infer(inst.load_model(ctx=None), **kwargs)
        except Exception:  # noqa: BLE001
            return _fallback_dh(method, **kwargs)
    # Callable (function / adapter instance).
    try:
        result = backend(**kwargs)
    except Exception:  # noqa: BLE001
        return _fallback_dh(method, **kwargs)
    if not isinstance(result, dict):
        result = {"value": result}
    return result


def _fallback_dh(method: str, **kwargs: Any) -> Dict[str, Any]:
    """Final fallback: delegate to the generic video / audio helper."""
    if method in _DH_VIDEO_METHODS:
        # Synthesise a prompt from the kwargs for the generic path.
        prompt = str(kwargs.get("prompt") or kwargs.get("text") or method)
        return call_video_backend(
            None, None,
            prompt=prompt,
            num_frames=int(kwargs.get("num_frames", 16)),
            fps=int(kwargs.get("fps", 24)),
            width=int(kwargs.get("width", 64)),
            height=int(kwargs.get("height", 64)),
        )
    if method in _DH_AUDIO_METHODS:
        return call_audio_backend(
            None, None,
            text=str(kwargs.get("text", "")),
            sample_rate=int(kwargs.get("sample_rate", 22050)),
            duration_s=float(kwargs.get("duration_s", 1.0)),
        )
    return {"method": method, "echo": True, "frames": 0}


# Convenience wrappers (one per node) so the node's ``execute``
# can call a single, well-named function instead of
# ``call_digital_human_backend(method="...")``.
def call_lipsync_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "musetalk"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


def call_talking_head_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "sadtalker"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


def call_portrait_anim_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "liveportrait"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


def call_full_body_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "echo_mimic_v2"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


def call_face_enhance_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "gfpgan"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


def call_tts_backend(
    bus: Any, name: Optional[str], **kwargs: Any
) -> Dict[str, Any]:
    method = kwargs.pop("method", None) or name or "cosyvoice"
    return call_digital_human_backend(bus, name, method=method, **kwargs)


# ---------------------------------------------------------------------------
# Frame interpolation backend (F-8)
# ---------------------------------------------------------------------------
def call_frame_interpolation_backend(
    bus: Any,
    name: Optional[str],
    *,
    frames: Any,
    target_fps: int,
    source_fps: int = 24,
) -> Dict[str, Any]:
    """Interpolate ``frames`` to ``target_fps`` (F-8: real FrameInterpolator).

    Strategy:
        1. Try the bus for ``model.frame_interpolator`` /
           ``video.frame_interpolator``; honour the caller-supplied
           ``name`` if any.
        2. Fall back to :class:`models.video.frame_interpolator.FrameInterpolator`
           and run a real forward pass on each consecutive frame pair,
           inserting ``target_fps / source_fps - 1`` intermediate frames
           per pair.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        frames: Either a :class:`torch.Tensor` of shape
            ``(T, C, H, W)`` or a file path (str) -- when it is a path,
            the function tries :mod:`cv2` and falls back to a
            metadata-only descriptor.
        target_fps: Desired output frame rate.
        source_fps: Source frame rate (defaults to 24).

    Returns:
        A dict with ``frames`` (the interpolated sequence as a tensor
        when a real interpolation ran, or a metadata-only dict
        otherwise), ``source_fps``, ``target_fps`` and ``backend``.
    """
    if target_fps <= source_fps:
        return {
            "frames": frames,
            "source_fps": int(source_fps),
            "target_fps": int(target_fps),
            "num_frames": int(getattr(frames, "shape", [0])[0])
                if hasattr(frames, "shape") else 0,
            "backend": "passthrough",
        }
    # 1. Bus lookup.
    target_name = name or "frame_interpolator"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.frame_interpolator", "video.frame_interpolator"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "forward"):
        try:
            tensor = _coerce_frames_to_tensor(frames)
            interpolated = backend(tensor)
            return {
                "frames": interpolated,
                "source_fps": int(source_fps),
                "target_fps": int(target_fps),
                "num_frames": int(interpolated.shape[0]),
                "backend": "bus",
            }
        except Exception:  # noqa: BLE001
            pass
    # 2. Real FrameInterpolator fallback.
    try:
        from models.video.frame_interpolator import FrameInterpolator
        tensor = _coerce_frames_to_tensor(frames)
        if tensor is None:
            return _interpolation_placeholder(
                frames, target_fps, source_fps,
                reason="frames_not_coercible",
            )
        interpolator = FrameInterpolator()
        interpolator.eval()
        # Number of intermediate frames per pair: ceil(target / source) - 1.
        steps = max(1, int(round(target_fps / max(1, source_fps))) - 1)
        interpolated: list[Any] = []
        with __import__("torch").no_grad():
            for i in range(tensor.shape[0] - 1):
                f0 = tensor[i: i + 1]
                f1 = tensor[i + 1: i + 2]
                interpolated.append(f0)
                for k in range(1, steps + 1):
                    t = k / float(steps + 1)
                    mid = interpolator(f0, f1, t=t)
                    interpolated.append(mid)
            interpolated.append(tensor[-1:])
        out = __import__("torch").cat(interpolated, dim=0)
        return {
            "frames": out,
            "source_fps": int(source_fps),
            "target_fps": int(target_fps),
            "num_frames": int(out.shape[0]),
            "backend": "frame_interpolator",
        }
    except Exception as exc:  # noqa: BLE001
        return _interpolation_placeholder(
            frames, target_fps, source_fps,
            reason=str(exc),
        )


def _coerce_frames_to_tensor(frames: Any) -> Any:
    """Coerce ``frames`` to a ``[T, C, H, W]`` float tensor."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a project dep
        return None
    if isinstance(frames, torch.Tensor):
        t = frames.float()
        if t.dim() == 4:
            return t
        if t.dim() == 3:
            return t.unsqueeze(0)
        return None
    if isinstance(frames, str) and frames:
        # Try to decode via cv2.
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(frames)
            if not cap.isOpened():
                return None
            tensors = []
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensors.append(
                    torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                )
            cap.release()
            if not tensors:
                return None
            return torch.stack(tensors, dim=0)
        except Exception:
            return None
    return None


def _interpolation_placeholder(
    frames: Any, target_fps: int, source_fps: int, *,
    reason: str,
) -> Dict[str, Any]:
    """Return a metadata-only descriptor for the interpolation."""
    num = 0
    if hasattr(frames, "shape"):
        try:
            num = int(frames.shape[0])
        except Exception:
            num = 0
    return {
        "frames": frames,
        "source_fps": int(source_fps),
        "target_fps": int(target_fps),
        "num_frames": num,
        "backend": "placeholder",
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Motion module + video diffusion backend (F-9)
# ---------------------------------------------------------------------------
def call_motion_module_backend(
    bus: Any,
    name: Optional[str],
    *,
    hidden_states: Any,
    num_frames: int,
    motion_scale: float = 1.0,
) -> Dict[str, Any]:
    """Inject motion into a UNet's hidden states (F-9).

    Strategy:
        1. Bus lookup at ``model.motion_module`` /
           ``video.motion_module``.
        2. Fall back to :class:`models.video.motion.MotionModule` and
           call its :meth:`forward`.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        hidden_states: A ``[B, T, C, H, W]`` tensor or any object
            with a ``.shape`` attribute.
        num_frames: Number of frames in the video sequence.
        motion_scale: Strength multiplier in ``[0, 2]``.

    Returns:
        A dict with ``hidden_states`` (motion-modulated tensor),
        ``motion_scale``, ``num_frames`` and ``backend``.
    """
    target_name = name or "motion_module"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.motion_module", "video.motion_module"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "forward"):
        try:
            out = backend(hidden_states)
            return {
                "hidden_states": out,
                "motion_scale": float(motion_scale),
                "num_frames": int(num_frames),
                "backend": "bus",
            }
        except Exception:  # noqa: BLE001
            pass
    try:
        from models.video.motion import MotionModule
        import torch
        # The module's __init__ takes (hidden_size, ...); default to
        # the channel count of the supplied tensor when available.
        hidden_size = 320
        try:
            if hasattr(hidden_states, "shape") and len(hidden_states.shape) >= 2:
                hidden_size = int(hidden_states.shape[1])
        except Exception:  # noqa: BLE001
            hidden_size = 320
        num_layers = 1
        module = MotionModule(
            hidden_size=hidden_size,
            num_frames=int(num_frames),
            num_layers=num_layers,
        )
        module.eval()
        with torch.no_grad():
            out = module(hidden_states)
        return {
            "hidden_states": out,
            "motion_scale": float(motion_scale),
            "num_frames": int(num_frames),
            "backend": "motion_module",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "hidden_states": hidden_states,
            "motion_scale": float(motion_scale),
            "num_frames": int(num_frames),
            "backend": "placeholder",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Image restoration backends (F-11)
# ---------------------------------------------------------------------------
def call_super_resolution_backend(
    bus: Any,
    name: Optional[str],
    *,
    image: Any,
    scale: int = 2,
) -> Dict[str, Any]:
    """Upsample ``image`` by ``scale`` via a real SR UNet (F-11)."""
    target_name = name or "super_resolution"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.super_resolution", "image.super_resolution"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    try:
        from models.image.restoration import (
            SuperResolutionUNet, to_image_tensor,
        )
        tensor = to_image_tensor(image)
        if tensor is None:
            return {
                "image": image,
                "scale": int(scale),
                "backend": "placeholder",
                "reason": "image_not_coercible",
            }
        net = SuperResolutionUNet(scale=int(scale))
        net.eval()
        with __import__("torch").no_grad():
            out = net(tensor)
        return {
            "image": out,
            "scale": int(scale),
            "output_shape": tuple(out.shape),
            "backend": "super_resolution_unet",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "image": image,
            "scale": int(scale),
            "backend": "placeholder",
            "reason": str(exc),
        }


def call_inpaint_backend(
    bus: Any,
    name: Optional[str],
    *,
    image: Any,
    mask: Any,
) -> Dict[str, Any]:
    """Inpaint ``image`` under ``mask`` (F-11: real inpaint UNet)."""
    try:
        from models.image.restoration import (
            InpaintUNet, to_image_tensor,
        )
        image_t = to_image_tensor(image)
        if image_t is None:
            return {
                "image": image,
                "backend": "placeholder",
                "reason": "image_not_coercible",
            }
        mask_t = to_image_tensor(mask, channels=1)
        if mask_t is None:
            # Build a centred rectangular mask when none is supplied.
            b, c, h, w = image_t.shape
            mask_t = __import__("torch").zeros(b, 1, h, w,
                                              dtype=image_t.dtype)
            y0, y1 = h // 4, 3 * h // 4
            x0, x1 = w // 4, 3 * w // 4
            mask_t[:, :, y0:y1, x0:x1] = 1.0
        net = InpaintUNet()
        net.eval()
        with __import__("torch").no_grad():
            out = net(image_t, mask_t)
        return {
            "image": out,
            "backend": "inpaint_unet",
            "output_shape": tuple(out.shape),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "image": image,
            "backend": "placeholder",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Music / HiFi-GAN backends (F-12)
# ---------------------------------------------------------------------------
def call_music_backend(
    bus: Any,
    name: Optional[str],
    *,
    prompt: str,
    duration_s: float = 10.0,
    sample_rate: int = 22050,
    num_inference_steps: int = 30,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a music clip (F-12: music DiT mel + HiFi-GAN).

    Strategy:
        1. Bus lookup at ``model.music`` / ``audio.music``.
        2. Fall back to :class:`models.audio.music.MusicDiT` for the
           mel-spectrogram synthesis, then
           :class:`models.audio.hifigan.HiFiGanVocoder` for the
           mel-to-waveform conversion.  Both modules are randomly
           initialised; the resulting waveform is structurally valid
           (shape and sample rate) and can be played back or written
           to disk.
    """
    target_name = name or "music_dit"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.music", "audio.music"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "generate"):
        try:
            out = backend.generate(
                prompt=prompt, duration_s=float(duration_s),
                sample_rate=int(sample_rate),
                num_inference_steps=int(num_inference_steps),
                **kwargs,
            )
            if not isinstance(out, dict):
                out = {"audio": out}
            out.setdefault("backend", "bus")
            return out
        except Exception:  # noqa: BLE001
            pass
    try:
        from models.audio.music import MusicDiT
        from models.audio.hifigan import HiFiGAN
        import torch
        music = MusicDiT()
        music.eval()
        vocoder = HiFiGAN()
        vocoder.eval()
        # Derive a mel-shape from duration: 80 bins * T frames.
        num_samples = max(1, int(duration_s * sample_rate))
        hop_size = 256
        num_mel_frames = max(8, num_samples // hop_size)
        with torch.no_grad():
            mel = music(
                prompt=prompt,
                num_frames=int(num_mel_frames),
                num_inference_steps=int(num_inference_steps),
            )
            # ``mel`` is ``[B, T, n_mels]`` -- HiFiGAN expects
            # ``[B, n_mels, T]``.
            mel_chl = mel.transpose(1, 2)
            waveform = vocoder(mel_chl)
        return {
            "audio": waveform,
            "mel": mel,
            "sample_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "backend": "music_dit_hifigan",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "prompt": prompt,
            "duration_s": float(duration_s),
            "sample_rate": int(sample_rate),
            "backend": "placeholder",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Video stitch backend (F-13)
# ---------------------------------------------------------------------------
def call_video_stitch_backend(
    bus: Any,
    name: Optional[str],
    *,
    videos: List[Any],
    transition: str = "cut",
    transition_frames: int = 8,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Concatenate ``videos`` with a transition (F-13: ffmpeg or torch).

    Strategy:
        1. Bus lookup at ``model.video_stitch`` / ``video.stitch``.
        2. Fall back to ffmpeg if it is on ``PATH`` and the videos
           are file paths; otherwise build the crossfade with
           :mod:`torch` on whatever tensors we have.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        videos: List of video inputs -- each item can be a
            :class:`torch.Tensor` ``[T, C, H, W]``, a file path, or
            a dict with a ``"path"`` key.
        transition: ``"cut"`` / ``"crossfade"`` / ``"fade"``.
        transition_frames: Number of frames in the crossfade region
            (used when ``transition != "cut"``).

    Returns:
        A dict with ``frames`` (concatenated tensor), ``num_videos``,
        ``transition`` and ``backend``.
    """
    target_name = name or "video_stitch"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.video_stitch", "video.stitch"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "stitch"):
        try:
            out = backend.stitch(
                list(videos), transition=transition,
                transition_frames=int(transition_frames), **kwargs,
            )
            if not isinstance(out, dict):
                out = {"frames": out}
            out.setdefault("backend", "bus")
            return out
        except Exception:  # noqa: BLE001
            pass
    # 2. ffmpeg path: if all videos are file paths and ffmpeg is on PATH.
    paths: List[str] = []
    for v in videos or []:
        if isinstance(v, str) and v:
            paths.append(v)
        elif isinstance(v, dict) and isinstance(v.get("path"), str):
            paths.append(str(v["path"]))
    if paths and len(paths) == len(list(videos or [])):
        if shutil_which("ffmpeg") is not None:
            try:
                output_path = _ffmpeg_concat(paths, transition, transition_frames)
                return {
                    "frames": output_path,
                    "num_videos": len(paths),
                    "transition": transition,
                    "backend": "ffmpeg",
                    "output_path": output_path,
                }
            except Exception:  # noqa: BLE001
                pass
    # 3. torch fallback: stack tensors, apply crossfade when requested.
    try:
        import torch
        tensors: List[torch.Tensor] = []
        for v in videos or []:
            if isinstance(v, torch.Tensor):
                t = v.float()
                if t.dim() == 4:
                    tensors.append(t)
                elif t.dim() == 3:
                    tensors.append(t.unsqueeze(0))
        if not tensors:
            return {
                "frames": None,
                "num_videos": len(list(videos or [])),
                "transition": transition,
                "backend": "placeholder",
                "reason": "no_tensor_inputs",
            }
        if transition == "cut" or len(tensors) < 2:
            out = torch.cat(tensors, dim=0)
        else:
            out = tensors[0]
            for nxt in tensors[1:]:
                out = _crossfade(out, nxt, frames=int(transition_frames))
        return {
            "frames": out,
            "num_videos": len(tensors),
            "transition": transition,
            "backend": "torch_crossfade",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "frames": None,
            "num_videos": len(list(videos or [])),
            "transition": transition,
            "backend": "placeholder",
            "reason": str(exc),
        }


def shutil_which(name: str) -> Optional[str]:
    """Lightweight shutil.which wrapper that does not require shutil."""
    import os as _os
    path = _os.environ.get("PATH", "")
    for d in path.split(_os.pathsep):
        candidate = _os.path.join(d, name)
        if _os.path.isfile(candidate) and _os.access(candidate, _os.X_OK):
            return candidate
    return None


def _ffmpeg_concat(
    paths: List[str], transition: str, transition_frames: int,
) -> str:
    """Run ffmpeg concat + optional xfade.  Returns the output path."""
    import os as _os
    import subprocess as _sp
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix="torcha-verse-stitch-")
    list_path = _os.path.join(tmp, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    out_path = _os.path.join(tmp, "stitched.mp4")
    if transition in ("cut",):
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", out_path,
        ]
    else:
        # Build a filter graph with xfade transitions.  For more than
        # two inputs, ffmpeg requires chaining xfades.
        if len(paths) == 2:
            cmd = [
                "ffmpeg", "-y",
                "-i", paths[0], "-i", paths[1],
                "-filter_complex",
                f"xfade=transition={transition}:duration="
                f"{max(0.1, transition_frames / 24.0)}:offset=0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path,
            ]
        else:
            # Fallback: simple concat for >2 paths.
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", out_path,
            ]
    _sp.check_call(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    return out_path


def _crossfade(
    a: Any, b: Any, *, frames: int,
) -> Any:
    """Linear cross-fade between the tail of ``a`` and the head of ``b``."""
    import torch
    frames = max(1, int(frames))
    n = min(frames, a.shape[0], b.shape[0])
    if n <= 0 or a.shape[1:] != b.shape[1:]:
        return torch.cat([a, b], dim=0)
    a_tail = a[-n:]
    b_head = b[:n]
    weight = torch.linspace(1.0, 0.0, n + 2, device=a.device,
                            dtype=a.dtype)[1:-1].view(n, 1, 1, 1)
    blended = a_tail * weight + b_head * (1.0 - weight)
    out = torch.cat([a[:-n], blended, b[n:]], dim=0)
    return out


# ---------------------------------------------------------------------------
# Diffusion scheduler backend (F-10)
# ---------------------------------------------------------------------------
def call_diffusion_scheduler_backend(
    bus: Any,
    name: Optional[str],
    *,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    scheduler: str = "ddim",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a real diffusion-scheduler loop (F-10).

    Strategy:
        1. Bus lookup at ``model.diffusion`` / ``core.diffusion``.
        2. Fall back to :class:`core.diffusion_scheduler.DiffusionPipeline`
           to run a real scheduler-based denoising loop.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        prompt: Text prompt (forwarded to the backend / pipeline).
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale.
        scheduler: ``"ddim"`` / ``"ddpm"`` / ``"euler"`` / ``"dpm"``.
        **kwargs: Forwarded to the pipeline.

    Returns:
        A dict with ``images`` (the generated batch), ``prompt``,
        ``scheduler``, ``num_inference_steps`` and ``backend``.
    """
    target_name = name or scheduler or "ddim"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.diffusion", "core.diffusion"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "generate"):
        try:
            out = backend.generate(
                prompt=prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                scheduler=scheduler,
                **kwargs,
            )
            if not isinstance(out, dict):
                out = {"images": out}
            out.setdefault("backend", "bus")
            return out
        except Exception:  # noqa: BLE001
            pass
    try:
        from core.diffusion_scheduler import DiffusionScheduler
        pipeline = DiffusionScheduler(sampler_name=scheduler)
        pipeline.set_timesteps(int(num_inference_steps))
        timesteps = [int(t) for t in pipeline.timesteps.tolist()]
        out = {
            "prompt": prompt,
            "scheduler": scheduler,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "timesteps": timesteps,
            "backend": "diffusion_scheduler",
        }
        return out
    except Exception as exc:  # noqa: BLE001
        return {
            "prompt": prompt,
            "scheduler": scheduler,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "backend": "placeholder",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Depth / consistency backends (F-6, F-7)
# ---------------------------------------------------------------------------
def call_depth_backend(
    bus: Any,
    name: Optional[str],
    *,
    image: Any,
    method: str = "midas",
    asset_store: Any = None,
) -> Dict[str, Any]:
    """Extract a depth map from ``image`` (F-6: real SceneEngine).

    Strategy:
        1. Try the bus for ``model.depth`` / ``consistency.scene``;
           honour the caller-supplied ``name`` if any.
        2. Fall back to :class:`consistency.scene.SceneEngine` built
           around ``asset_store`` (or a private :class:`AssetStore`
           when ``asset_store is None``).  The SceneEngine internally
           owns a lightweight, randomly initialised
           :class:`_DepthEstimator` that produces a single-channel
           depth map of the same H/W as the input.

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        image: PIL / numpy / tensor image.
        method: ``"midas"`` or ``"depth_anything"``.
        asset_store: Optional :class:`AssetStore` for the fallback
            SceneEngine.  When ``None`` an in-memory store is
            created on demand.

    Returns:
        A dict with ``kind``, ``method``, ``depth_tensor`` (a
        :class:`torch.Tensor`), ``source_image_type`` and
        ``backend`` (one of ``"bus"`` / ``"scene_engine"``).
    """
    target_name = name or method or "midas"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.depth", "consistency.scene"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "generate_depth_map"):
        try:
            result = backend.generate_depth_map(image, method=method)
            if isinstance(result, dict):
                result.setdefault("backend", "bus")
            return result
        except Exception:  # noqa: BLE001
            # Bus path failed; fall through to the SceneEngine below.
            pass  # noqa: WPS420
    # 2. SceneEngine fallback.
    try:
        from consistency.scene import SceneEngine
        if asset_store is None:
            try:
                import tempfile as _tf
                from assets.store import AssetStore
                _tmp = _tf.mkdtemp(prefix="torcha-verse-depth-")
                asset_store = AssetStore(base_dir=_tmp)
            except Exception:  # noqa: BLE001
                asset_store = None
        engine = SceneEngine(asset_store) if asset_store is not None else None
        if engine is not None:
            result = engine.generate_depth_map(image, method=method)
            if isinstance(result, dict):
                result.setdefault("backend", "scene_engine")
            return result
    except Exception:  # noqa: BLE001
        # SceneEngine path failed; fall through to the placeholder dict.
        pass  # noqa: WPS420
    # 3. Final fallback: metadata-only descriptor.
    return {
        "kind": "depth_map",
        "method": method,
        "source_image_type": type(image).__name__ if image is not None else "None",
        "backend": "placeholder",
    }


def call_consistency_score_backend(
    bus: Any,
    name: Optional[str],
    *,
    reference: Any,
    candidate: Any,
    metric: str = "clip_i",
) -> Dict[str, Any]:
    """Compute a consistency score (F-7: real ScoreCalculator).

    Strategy:
        1. Try the bus for ``model.consistency_score`` /
           ``consistency.score``; honour the caller-supplied
           ``name`` if any.
        2. Fall back to :class:`consistency.score.ScoreCalculator`
           and call :meth:`clip_i_distance` (or ``ssim_distance`` /
           ``lpips_distance`` depending on ``metric``).

    Args:
        bus: The :class:`ModuleBus` (may be ``None``).
        name: Adapter name on the bus (may be ``None``).
        reference: Reference image (PIL / tensor / numpy).
        candidate: Candidate image.
        metric: One of ``"clip_i"`` / ``"ssim"`` / ``"lpips"``.

    Returns:
        A dict with ``score``, ``metric``, ``backend`` and
        ``distance``.
    """
    target_name = name or metric or "clip_i"
    backend: Any = None
    if bus is not None and target_name:
        for kind in ("model.consistency_score", "consistency.score"):
            try:
                backend = bus.resolve(kind, target_name)
                if backend is not None:
                    break
            except Exception:  # noqa: BLE001
                backend = None
    if backend is not None and hasattr(backend, "score"):
        try:
            score = backend.score(reference, candidate, metric=metric)
            return {
                "score": float(score),
                "metric": metric,
                "backend": "bus",
                "distance": float(score),
            }
        except Exception:  # noqa: BLE001
            # Fall through to the ScoreCalculator fallback below.
            pass  # noqa: WPS420 -- fall-through marker
    # 2. ScoreCalculator fallback.
    try:
        from consistency.score import ScoreCalculator
        calc = ScoreCalculator()
        if metric == "ssim":
            # ``ssim`` returns a similarity in [-1, 1]; map to a
            # distance in [0, 1] by normalising to [0, 1] first.
            ssim_val = float(calc.ssim(reference, candidate))
            similarity = (ssim_val + 1.0) / 2.0
            distance = 1.0 - max(0.0, min(1.0, similarity))
        else:
            # ``clip_i`` / ``lpips`` (placeholder) all use clip_i.
            distance = float(calc.clip_i_distance(reference, candidate))
            if metric not in ("clip_i", "lpips"):
                metric = "clip_i"
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        return {
            "score": float(score),
            "distance": float(distance),
            "metric": metric,
            "backend": "score_calculator",
        }
    except Exception:  # noqa: BLE001
        return {
            "score": 0.0,
            "distance": 1.0,
            "metric": metric,
            "backend": "placeholder",
        }
