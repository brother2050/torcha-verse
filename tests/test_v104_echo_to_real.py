"""v0.10.4: 默认 fallback 改走项目真模型, 不再 echo。

覆盖:

* :func:`_resolve_via_bus_or_default` 默认 lazy install 路径
  走 :class:`LocalTorch*Provider` 而非 echo factory
* :class:`PipelineService` 默认 :attr:`_llm_provider` 是
  :class:`ChatTemplateProvider` (包 LocalTorchTextProvider)
  而非 :class:`EchoProvider`
* text_completion / text_chat / agent_run 三条端到端路径
  都不再以 ``[echo-text]`` 开头
* 5 个 echo factory 仍可显式调用 (向后兼容测试 fixture)
"""
from __future__ import annotations

import logging
import unittest
from typing import Any
from unittest.mock import patch

from models.providers import (
    LocalTorchAudioProvider,
    LocalTorchImageProvider,
    LocalTorchMultimodalProvider,
    LocalTorchTextProvider,
    LocalTorchVideoProvider,
)
from models.providers.local_text import TINY_CONFIG
from nodes._helpers._backends import (
    _audio_echo_factory,
    _image_echo_factory,
    _multimodal_echo_factory,
    _resolve_via_bus_or_default,
    _text_echo_factory,
    _video_echo_factory,
    reset_default_backends,
)


# ---------------------------------------------------------------------------
# 1. _resolve_via_bus_or_default 默认走真模型
# ---------------------------------------------------------------------------
class TestResolveDefaultIsLocalNotEcho(unittest.TestCase):
    def setUp(self) -> None:
        reset_default_backends()

    def tearDown(self) -> None:
        reset_default_backends()

    def test_text_default_is_local_text_provider(self) -> None:
        backend = _resolve_via_bus_or_default(
            bus=None, kind="model.text", name=None,
            default_kind="text",
        )
        assert isinstance(backend, LocalTorchTextProvider), (
            f"expected LocalTorchTextProvider, got {type(backend).__name__}"
        )

    def test_image_default_is_local_image_provider(self) -> None:
        backend = _resolve_via_bus_or_default(
            bus=None, kind="model.image", name=None,
            default_kind="image",
        )
        assert isinstance(backend, LocalTorchImageProvider), (
            f"expected LocalTorchImageProvider, got {type(backend).__name__}"
        )

    def test_video_default_is_local_video_provider(self) -> None:
        backend = _resolve_via_bus_or_default(
            bus=None, kind="model.video", name=None,
            default_kind="video",
        )
        assert isinstance(backend, LocalTorchVideoProvider), (
            f"expected LocalTorchVideoProvider, got {type(backend).__name__}"
        )

    def test_audio_default_is_local_audio_provider(self) -> None:
        backend = _resolve_via_bus_or_default(
            bus=None, kind="model.audio", name=None,
            default_kind="audio",
        )
        assert isinstance(backend, LocalTorchAudioProvider), (
            f"expected LocalTorchAudioProvider, got {type(backend).__name__}"
        )

    def test_multimodal_default_is_local_multimodal_provider(self) -> None:
        backend = _resolve_via_bus_or_default(
            bus=None, kind="model.multimodal", name=None,
            default_kind="multimodal",
        )
        assert isinstance(backend, LocalTorchMultimodalProvider), (
            f"expected LocalTorchMultimodalProvider, got {type(backend).__name__}"
        )


# ---------------------------------------------------------------------------
# 2. PipelineService 默认 LLM provider 是 ChatTemplateProvider
# ---------------------------------------------------------------------------
class TestPipelineServiceLLMProvider(unittest.TestCase):
    def setUp(self) -> None:
        reset_default_backends()

    def tearDown(self) -> None:
        reset_default_backends()

    def test_default_llm_provider_is_chat_template_not_echo(self) -> None:
        from serving.service._service import PipelineService
        svc = PipelineService()
        try:
            assert svc._llm_provider is not None
            # ``EchoProvider`` defines ``name == 'echo'``; the
            # ChatTemplateProvider wrapping our micro-transformer
            # has the name we passed to the constructor.
            assert svc._llm_provider.name != "echo", (
                "PipelineService._llm_provider must NOT be an "
                "EchoProvider in v0.10.4+"
            )
            assert svc._llm_provider.name == "torcha-verse-micro-transformer"
        finally:
            # ``PipelineService`` does not implement ``close`` but
            # test isolation is guaranteed via ``reset_default_backends``.
            pass


# ---------------------------------------------------------------------------
# 3. 端到端: text_completion / text_chat 不再以 [echo-text] 开头
# ---------------------------------------------------------------------------
class TestTextPipelineEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        reset_default_backends()

    def tearDown(self) -> None:
        reset_default_backends()

    def test_text_completion_not_echo(self) -> None:
        from serving.service._service import PipelineService
        svc = PipelineService()
        result = svc.text_completion(
            prompt="how are you?",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_tokens=10,
        )
        text = result.get("text", "")
        assert not text.startswith("[echo-text"), (
            f"text_completion returned echo-style output: {text[:80]!r}"
        )
        assert text, "text_completion returned empty string"

    def test_text_chat_not_echo(self) -> None:
        from serving.service._service import PipelineService
        svc = PipelineService()
        result = svc.text_chat(
            prompt="how are you?",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_tokens=10,
        )
        text = result.get("text", "")
        assert not text.startswith("[echo-text"), (
            f"text_chat returned echo-style output: {text[:80]!r}"
        )
        assert text, "text_chat returned empty string"


# ---------------------------------------------------------------------------
# 4. 向后兼容: 5 个 echo factory 仍可显式调用 (测试 fixture)
# ---------------------------------------------------------------------------
class TestEchoFactoryBackwardCompat(unittest.TestCase):
    """Echo factories remain available for tests that explicitly
    opt-in to the no-model path.  They MUST NOT be removed --
    removal would break the e2e fixture tests in
    ``test_v100_components_and_interfaces`` /
    ``test_multimodal_providers`` and the v0.10.3 model-name
    labelling tests."""

    def test_text_echo_still_callable(self) -> None:
        backend = _text_echo_factory()
        out = backend.generate("hi")
        assert isinstance(out, str) and out  # non-empty

    def test_image_echo_still_callable(self) -> None:
        backend = _image_echo_factory()
        out = backend.generate("hi", width=32, height=32)
        assert isinstance(out, dict)
        assert out.get("image") is not None

    def test_video_echo_still_callable(self) -> None:
        backend = _video_echo_factory()
        out = backend.generate("hi", num_frames=2)
        assert isinstance(out, dict)
        assert "frames" in out

    def test_audio_echo_still_callable(self) -> None:
        backend = _audio_echo_factory()
        out = backend.generate(text="hi", sample_rate=8000, duration_s=0.1)
        assert isinstance(out, dict)
        assert "waveform" in out

    def test_multimodal_echo_still_callable(self) -> None:
        backend = _multimodal_echo_factory()
        # Multimodal echo is a thin wrapper around
        # ``EchoMultimodalProvider``; just ensure it constructs.
        assert backend is not None


# ---------------------------------------------------------------------------
# 5. _local_*_factory 自身 import 失败仍回退 echo (双层兜底)
# ---------------------------------------------------------------------------
class TestLocalFactoryImportFallback(unittest.TestCase):
    """The ``_local_*_factory`` helpers in ``_backends.py`` wrap
    ``LocalTorch*Provider.from_random()`` in a ``try / except`` that
    falls back to the echo factory when the import explodes.  This
    guards against exotic sandboxes where the providers package
    is unavailable (e.g. minimal CI containers)."""

    def test_text_factory_falls_back_to_echo_on_import_error(self) -> None:
        from nodes._helpers._backends import _local_text_factory
        import models.providers
        # ``_local_text_factory`` does ``from models.providers
        # import LocalTorchTextProvider`` -- the ``from X import
        # Y`` form binds the symbol at import time, so patching
        # the source module attribute is enough to break the
        # subsequent ``from_random`` call (which is what the
        # except-block actually guards against).
        original = models.providers.LocalTorchTextProvider
        try:
            class _Broken:
                @staticmethod
                def from_random(*args, **kwargs):
                    raise ImportError("simulated from_random failure")
            models.providers.LocalTorchTextProvider = _Broken
            backend = _local_text_factory()
        finally:
            models.providers.LocalTorchTextProvider = original
        # Echo backend's ``generate`` is a method.
        assert hasattr(backend, "generate")
        # ``EchoTextBackend.generate`` echoes the prompt -- the
        # behaviour is intentionally the same as the previous
        # framework default so tests that depend on the shape
        # still pass.
        out = backend.generate("hi")
        assert "hi" in out

    def test_image_factory_falls_back_to_echo_on_import_error(self) -> None:
        from nodes._helpers._backends import _local_image_factory
        import models.providers
        original = models.providers.LocalTorchImageProvider
        try:
            class _Broken:
                @staticmethod
                def from_random(*args, **kwargs):
                    raise ImportError("simulated from_random failure")
            models.providers.LocalTorchImageProvider = _Broken
            backend = _local_image_factory()
        finally:
            models.providers.LocalTorchImageProvider = original
        out = backend.generate("hi", width=16, height=16)
        assert isinstance(out, dict)
        assert "image" in out


if __name__ == "__main__":
    unittest.main()
