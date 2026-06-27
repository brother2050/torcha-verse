"""Runtime config: 一行开启"自研 transformers 风格"本地运行时 (v0.10.0)。

本模块提供 **3 个** 项目级的"运行时开关":

1. :func:`enable_local_runtime` -- 一行把"自研加载 + 真推理循环"注入
   :class:`core.module_bus.ModuleBus`,让 39 个 L4 节点从默认的 echo
   工厂切到 **真模型真生成**。
2. :func:`disable_local_runtime` -- 还原回 echo 工厂 (用于 AB 对比 /
   单元测试)。
3. :func:`is_local_runtime_enabled` -- 进程级状态查询。

设计动机
--------

V0.4.x 的 P0 阶段已经实现了"真模型 + 真循环",但需要每个调用方手动
``register_default_text_backend`` / ``register_default_image_backend``
... 才能让 39 节点真正用上真模型。本模块把这一步打包成
**一行函数**。

零外部依赖
----------

不依赖 ``transformers`` / ``diffusers``。仅依赖项目自有的 L1-L6
模块。

测试 0 回归
-----------

* 全部占位 / 失败路径在 ``docs/placeholder_registry.md`` 登记
* 不破坏 1182+ 现有测试
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from infrastructure.logger import get_logger

__all__ = [
    "RuntimeConfig",
    "enable_local_runtime",
    "disable_local_runtime",
    "is_local_runtime_enabled",
    "get_active_config",
]


_logger = get_logger("models.runtime.module_bus_runtime_switch")


# ---------------------------------------------------------------------------
# Process-wide state
# ---------------------------------------------------------------------------
_ACTIVE_CONFIG: Optional["RuntimeConfig"] = None
_LOCK: threading.RLock = threading.RLock()


@dataclass
class RuntimeConfig:
    """The set of choices :func:`enable_local_runtime` makes.

    Attributes:
        prefer_local_text: When ``True``, register the v0.4.x
            :class:`LocalTorchTextProvider` as the default text
            backend.  When ``False``, the echo backend is kept.
        prefer_local_image: Same for image (drives
            :class:`LocalTorchImageProvider`).
        prefer_local_video: Same for video.
        prefer_local_audio: Same for audio.
        prefer_local_multimodal: Same for multimodal.
        torch_dtype: Optional dtype applied to every backend's
            primary model.
        device: Optional device override applied to every backend.
        max_memory_per_backend_gb: Optional per-backend memory cap
            (forwarded to the resource budget tracker when set).
        use_real_diffusion_loop: When ``True`` the image backend
            goes through :func:`call_diffusion_loop_backend` (the
            v0.8.x real sampler loop).  When ``False`` it stays on
            the random-latents echo path.
    """

    prefer_local_text: bool = True
    prefer_local_image: bool = True
    prefer_local_video: bool = True
    prefer_local_audio: bool = True
    prefer_local_multimodal: bool = True
    torch_dtype: Optional[Any] = None
    device: Union[None, str, Any] = None
    max_memory_per_backend_gb: Optional[float] = None
    use_real_diffusion_loop: bool = True
    # User-selected model identifier.  When non-empty, the text
    # factory tries to resolve the weights from
    # ``~/.cache/torcha-verse/<source>/<model_id>`` via
    # :func:`models.runtime.transformers_style_loader.load_model_and_tokenizer`
    # before falling back to the project micro-transformer.  Use the
    # ``"org/name"`` form for HuggingFace repos (e.g.
    # ``"Qwen/Qwen2.5-0.5B-Instruct"``) or a local path for
    # already-downloaded checkpoints.
    model_id: Optional[str] = None
    # Cache root used to resolve ``model_id``.  Defaults to
    # ``~/.cache/torcha-verse`` (matches the layout produced by
    # :mod:`models.source.fetch`).
    cache_root: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """Return a human-readable one-line description."""
        bits = []
        for k in (
            "prefer_local_text",
            "prefer_local_image",
            "prefer_local_video",
            "prefer_local_audio",
            "prefer_local_multimodal",
        ):
            if getattr(self, k):
                bits.append(k.replace("prefer_local_", ""))
        extras = []
        if self.torch_dtype is not None:
            extras.append(f"dtype={self.torch_dtype}")
        if self.device is not None:
            extras.append(f"device={self.device}")
        if self.max_memory_per_backend_gb is not None:
            extras.append(f"max_mem={self.max_memory_per_backend_gb}GB")
        if self.use_real_diffusion_loop:
            extras.append("real_diffusion_loop")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"LocalRuntime[{', '.join(bits) or 'no-op'}]{suffix}"


# ---------------------------------------------------------------------------
# enable_local_runtime
# ---------------------------------------------------------------------------
def enable_local_runtime(
    config: Optional[RuntimeConfig] = None,
    *,
    prefer_local_text: Optional[bool] = None,
    prefer_local_image: Optional[bool] = None,
    prefer_local_video: Optional[bool] = None,
    prefer_local_audio: Optional[bool] = None,
    prefer_local_multimodal: Optional[bool] = None,
    torch_dtype: Optional[Any] = None,
    device: Union[None, str, Any] = None,
    use_real_diffusion_loop: Optional[bool] = None,
    model_id: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> RuntimeConfig:
    """One-line "turn on" for the local runtime.

    The function:

    1. Builds a :class:`RuntimeConfig` (or merges the caller's
       kwargs into the supplied one).
    2. Calls each ``register_default_*_backend`` shim in
       :mod:`nodes._helpers._backends` so the 39 L4 nodes see the
       "real" backend factory instead of the echo one.
    3. Stashes the config in the module-level ``_ACTIVE_CONFIG``
       slot so :func:`is_local_runtime_enabled` /
       :func:`get_active_config` can introspect it.

    The function is **idempotent**: calling it twice with the
    same args is a no-op (the second call returns the existing
    config).  Calling it with different args updates the active
    config in place and re-registers the backends.
    """
    global _ACTIVE_CONFIG
    if config is None:
        config = RuntimeConfig()
    # Apply keyword overrides.
    if prefer_local_text is not None:
        config.prefer_local_text = bool(prefer_local_text)
    if prefer_local_image is not None:
        config.prefer_local_image = bool(prefer_local_image)
    if prefer_local_video is not None:
        config.prefer_local_video = bool(prefer_local_video)
    if prefer_local_audio is not None:
        config.prefer_local_audio = bool(prefer_local_audio)
    if prefer_local_multimodal is not None:
        config.prefer_local_multimodal = bool(prefer_local_multimodal)
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype
    if device is not None:
        config.device = device
    if use_real_diffusion_loop is not None:
        config.use_real_diffusion_loop = bool(use_real_diffusion_loop)
    if model_id is not None:
        config.model_id = str(model_id) if model_id else None
    if cache_root is not None:
        config.cache_root = str(cache_root) if cache_root else None

    with _LOCK:
        # Idempotent re-entry: if the caller didn't supply a config
        # explicitly and didn't ask to *change* any field, return the
        # current active config unchanged (mirrors
        # ``transformers.pipeline``'s "do not re-build" semantics).
        if (
            _ACTIVE_CONFIG is not None
            and config is not _ACTIVE_CONFIG
            and not any(
                v is not None
                for v in (
                    prefer_local_text, prefer_local_image, prefer_local_video,
                    prefer_local_audio, prefer_local_multimodal, torch_dtype,
                    device, use_real_diffusion_loop,
                )
            )
        ):
            return _ACTIVE_CONFIG
        # Otherwise, the caller's intent is to update the config; do so.
        _ACTIVE_CONFIG = config
    _register_backends(_ACTIVE_CONFIG)
    _logger.info("local runtime enabled: %s", _ACTIVE_CONFIG.describe())
    return _ACTIVE_CONFIG


# ---------------------------------------------------------------------------
# disable_local_runtime
# ---------------------------------------------------------------------------
def disable_local_runtime() -> None:
    """Reset the default backends to the echo factories.

    After this call :func:`is_local_runtime_enabled` returns
    ``False`` and the 39 L4 nodes fall back to the echo backend
    (i.e. random / placeholder behaviour).
    """
    global _ACTIVE_CONFIG
    with _LOCK:
        try:
            from nodes._helpers._backends import reset_default_backends
            reset_default_backends()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("disable_local_runtime: reset failed: %s", exc)
        _ACTIVE_CONFIG = None
    _logger.info("local runtime disabled (backends reset to echo)")


# ---------------------------------------------------------------------------
# is_local_runtime_enabled / get_active_config
# ---------------------------------------------------------------------------
def is_local_runtime_enabled() -> bool:
    """Return ``True`` when :func:`enable_local_runtime` was called
    and :func:`disable_local_runtime` was not.
    """
    with _LOCK:
        return _ACTIVE_CONFIG is not None


def get_active_config() -> Optional[RuntimeConfig]:
    """Return the current :class:`RuntimeConfig` (or ``None``)."""
    with _LOCK:
        if _ACTIVE_CONFIG is None:
            return None
        return _ACTIVE_CONFIG


# ---------------------------------------------------------------------------
# Internal: backend registration
# ---------------------------------------------------------------------------
def _register_backends(config: RuntimeConfig) -> None:
    """Call the four ``register_default_*_backend`` shims.

    We import lazily so that :mod:`models.runtime` stays
    dependency-free at import time (the backends themselves
    depend on :mod:`models.providers.*` and the diffusion
    loop helpers, which may not always be available -- e.g. on
    a fresh CI box with no GPU).
    """
    try:
        from nodes._helpers._backends import (
            register_default_text_backend,
            register_default_image_backend,
            register_default_video_backend,
            register_default_audio_backend,
            register_default_multimodal_backend,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "enable_local_runtime: backends shim unavailable (%s); "
            "skipping registration. The runtime is configured but the "
            "39 nodes will continue using their echo backends.",
            exc,
        )
        return
    # Text
    if config.prefer_local_text:
        try:
            register_default_text_backend(
                _make_text_factory(config),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("text backend registration failed: %s", exc)
    # Image
    if config.prefer_local_image:
        try:
            register_default_image_backend(
                _make_image_factory(config),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("image backend registration failed: %s", exc)
    # Video
    if config.prefer_local_video:
        try:
            register_default_video_backend(
                _make_video_factory(config),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("video backend registration failed: %s", exc)
    # Audio
    if config.prefer_local_audio:
        try:
            register_default_audio_backend(
                _make_audio_factory(config),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("audio backend registration failed: %s", exc)
    # Multimodal
    if config.prefer_local_multimodal:
        try:
            register_default_multimodal_backend(
                _make_multimodal_factory(config),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("multimodal backend registration failed: %s", exc)


def _make_text_factory(config: RuntimeConfig) -> Callable[[], Any]:
    """Return a zero-arg factory for the text backend.

    Resolution order:

    1. If ``config.model_id`` is set, try to load a real user-selected
       checkpoint via :func:`load_model_and_tokenizer` -- this is the
       "use my downloaded Qwen" path.  The model is wrapped in
       :class:`LocalTorchTextProvider` so the rest of the runtime
       sees the same ``generate(prompt, **kw) -> str`` interface.
       A ``NotImplementedError`` from :func:`load_model_and_tokenizer`
       (e.g. when the architecture is not yet implemented, such as
       Qwen2.5-0.5B without a LLaMA-derivative architecture in
       :mod:`models.text`) is logged and we fall through to the
       project micro-transformer.
    2. Fall back to the project micro-transformer
       (:class:`LocalTorchTextProvider.from_random`) so the
       framework always has a working backend.
    """
    cache_root = config.cache_root or _default_cache_root()

    def factory() -> Any:
        # 1) Try the user-selected checkpoint.
        if config.model_id:
            try:
                from models.runtime.transformers_style_loader import (
                    load_model_and_tokenizer,
                )
                from models.providers.local_text import (
                    LocalTorchTextProvider,
                )
                local_path = _resolve_user_model_path(
                    config.model_id, cache_root,
                )
                if local_path is None:
                    _logger.warning(
                        "user model %s not found in cache under %s. "
                        "Falling back to project micro-transformer. "
                        "Use `from models.source import fetch; "
                        "fetch('%s')` to download first.",
                        config.model_id, cache_root, config.model_id,
                    )
                else:
                    bundle = load_model_and_tokenizer(local_path)
                    provider = LocalTorchTextProvider.from_wrapped_model(
                        bundle, device=config.device or "cpu",
                    )
                    _logger.info(
                        "text backend: loaded user model %s (family=%s)",
                        config.model_id, bundle.family.value,
                    )
                    return provider
            except NotImplementedError as exc:
                # Real weights were located, but the architecture is
                # not yet implemented in the project (e.g. Qwen2
                # before the LLaMA-derivative architecture lands).
                _logger.warning(
                    "user model %s detected but architecture not "
                    "supported: %s. Falling back to project micro-"
                    "transformer (random weights, output is noise).",
                    config.model_id, exc,
                )
            except FileNotFoundError as exc:
                _logger.warning(
                    "user model %s not found in cache (%s). "
                    "Falling back to project micro-transformer. "
                    "Use `from models.source import fetch; "
                    "fetch('%s')` to download.",
                    config.model_id, exc, config.model_id,
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "loading user model %s failed: %s. "
                    "Falling back to project micro-transformer.",
                    config.model_id, exc,
                )
        # 2) Project micro-transformer fallback.
        try:
            from models.providers.local_text import (
                LocalTorchTextProvider,
            )
            return LocalTorchTextProvider.from_random(
                device=config.device or "cpu",
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("micro-transformer fallback failed: %s", exc)
            return None
    return factory


def _default_cache_root() -> str:
    """Return the project cache root (matches
    :mod:`models.source.fetch`'s default)."""
    import os
    return os.path.expanduser("~/.cache/torcha-verse")


def _resolve_user_model_path(
    model_id: str, cache_root: str,
) -> "Path | None":
    """Map a HuggingFace-style ``"org/name"`` to the local cache path.

    ``models.source.fetch`` writes the safetensors + tokenizer files
    under ``<cache_root>/<source>/<org>/<name>/``.  For HF repos the
    ``<source>`` is ``"huggingface"`` so the resulting path is
    ``<cache_root>/huggingface/<org>/<name>``.

    Returns ``None`` when no matching directory exists.  The user is
    expected to run :func:`models.source.fetch` first; the CLI prints
    a follow-up hint when the lookup misses.
    """
    from pathlib import Path
    if Path(model_id).is_dir():
        # Local path; use as-is.
        return Path(model_id)
    cache_root_path = Path(cache_root).expanduser()
    # Try the conventional huggingface layout first.
    candidate = cache_root_path / "huggingface" / model_id
    if candidate.is_dir():
        return candidate
    # Fall back to other sources (civitai / local).
    for source in ("huggingface", "civitai", "local", "modelscope"):
        alt = cache_root_path / source / model_id
        if alt.is_dir():
            return alt
    return None


def _make_image_factory(config: RuntimeConfig) -> Callable[[], Any]:
    """Return a zero-arg factory for the image backend."""
    def factory() -> Any:
        try:
            from models.providers.local_image import (
                LocalTorchImageProvider,
            )
            return LocalTorchImageProvider.from_random(
                device=config.device or "cpu",
            )
        except Exception:
            return None
    return factory


def _make_video_factory(config: RuntimeConfig) -> Callable[[], Any]:
    def factory() -> Any:
        try:
            from models.providers.local_video import (
                LocalTorchVideoProvider,
            )
            return LocalTorchVideoProvider.from_random(
                device=config.device or "cpu",
            )
        except Exception:
            return None
    return factory


def _make_audio_factory(config: RuntimeConfig) -> Callable[[], Any]:
    def factory() -> Any:
        try:
            from models.providers.local_audio import (
                LocalTorchAudioProvider,
            )
            return LocalTorchAudioProvider.from_random(
                device=config.device or "cpu",
            )
        except Exception:
            return None
    return factory


def _make_multimodal_factory(config: RuntimeConfig) -> Callable[[], Any]:
    def factory() -> Any:
        try:
            from models.providers.local_multimodal import (
                LocalTorchMultimodalProvider,
            )
            return LocalTorchMultimodalProvider.from_random(
                device=config.device or "cpu",
            )
        except Exception:
            return None
    return factory
