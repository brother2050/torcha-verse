"""v0.10.3 修复:用户下载的 HF 模型 (Qwen2 / LLaMA 等) 端到端可用性。

覆盖:

* :func:`enable_local_runtime` 接受 ``model_id`` / ``cache_root`` 参数
* :class:`ModelFamily` 识别 Qwen2 (LLaMA-derivative) 模型签名
* echo factory 输出带 ``[echo-text: no model registered for NAME]`` 标识
* :func:`_resolve_user_model_path` 在 cache 缺失 / 存在时的行为
* ``_FAMILY_PREFERENCE`` 在 QWEN2 / LLAMA 同分时倾向 QWEN2
"""
from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.runtime import (
    ModelFamily,
    detect_model_family,
    enable_local_runtime,
    get_active_config,
    is_local_runtime_enabled,
    disable_local_runtime,
)
from models.runtime.module_bus_runtime_switch import (
    RuntimeConfig,
    _resolve_user_model_path,
    _default_cache_root,
)
from nodes._helpers._echo import _text_echo_factory


# ---------------------------------------------------------------------------
# 1. ModelFamily 枚举完整性
# ---------------------------------------------------------------------------
class TestModelFamilyEnum(unittest.TestCase):
    def test_qwen2_enum_present(self) -> None:
        assert hasattr(ModelFamily, "QWEN2"), "ModelFamily.QWEN2 missing"
        assert ModelFamily.QWEN2.value == "qwen2"

    def test_llama_enum_present(self) -> None:
        assert hasattr(ModelFamily, "LLAMA"), "ModelFamily.LLAMA missing"
        assert ModelFamily.LLAMA.value == "llama"

    def test_qwen2_comes_after_musicgen(self) -> None:
        # Order matters for the iteration in detect_model_family.
        members = list(ModelFamily)
        idx_musicgen = members.index(ModelFamily.MUSICGEN)
        idx_qwen2 = members.index(ModelFamily.QWEN2)
        assert idx_qwen2 > idx_musicgen


# ---------------------------------------------------------------------------
# 2. detect_model_family 识别 QWEN2
# ---------------------------------------------------------------------------
class TestDetectQwen2(unittest.TestCase):
    def test_qwen2_signature(self) -> None:
        # A minimal Qwen2-0.5B state-dict header.  ``detect_model_family``
        # only inspects the first ``sample_size`` keys (default 32), so
        # this is enough to drive the heuristic.
        state_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        with patch(
            "models.runtime.transformers_style_loader."
            "_resolve_checkpoint_file",
            lambda p: Path("/fake/model.safetensors"),
        ), patch(
            "models.runtime.transformers_style_loader.load_safetensors",
            lambda p, device="cpu": {k: object() for k in state_keys},
        ):
            family = detect_model_family("/fake/Qwen2.5-0.5B-Instruct")
        assert family == ModelFamily.QWEN2, (
            f"expected QWEN2, got {family}"
        )

    def test_qwen2_wins_over_llama_on_tie(self) -> None:
        # QWEN2 and LLAMA share the same signature tokens, so the
        # preference map must order QWEN2 ahead of LLAMA.
        from models.runtime.transformers_style_loader import (
            _FAMILY_PREFERENCE,
        )
        assert _FAMILY_PREFERENCE[ModelFamily.QWEN2] > _FAMILY_PREFERENCE[
            ModelFamily.LLAMA
        ]


# ---------------------------------------------------------------------------
# 3. RuntimeConfig + enable_local_runtime 新增字段
# ---------------------------------------------------------------------------
class TestRuntimeConfigModelId(unittest.TestCase):
    def setUp(self) -> None:
        # ensure clean state across tests
        if is_local_runtime_enabled():
            disable_local_runtime()

    def tearDown(self) -> None:
        if is_local_runtime_enabled():
            disable_local_runtime()

    def test_default_config_has_model_id_none(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.model_id is None
        assert cfg.cache_root is None

    def test_enable_local_runtime_accepts_model_id_kwarg(self) -> None:
        cfg = enable_local_runtime(model_id="Qwen/Qwen2.5-0.5B-Instruct")
        assert cfg.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        # cache_root should default to None (so the factory uses
        # _default_cache_root()).
        assert cfg.cache_root is None

    def test_enable_local_runtime_accepts_cache_root_kwarg(self) -> None:
        cfg = enable_local_runtime(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            cache_root="/tmp/custom-cache",
        )
        assert cfg.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.cache_root == "/tmp/custom-cache"

    def test_get_active_config_reflects_model_id(self) -> None:
        enable_local_runtime(model_id="Qwen/Qwen2.5-0.5B-Instruct")
        assert get_active_config().model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_empty_model_id_is_normalised_to_none(self) -> None:
        # ``--model ""`` from a CLI should be treated as "no model".
        cfg = enable_local_runtime(model_id="")
        assert cfg.model_id is None


# ---------------------------------------------------------------------------
# 4. _resolve_user_model_path cache 行为
# ---------------------------------------------------------------------------
class TestResolveUserModelPath(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_local_path_passthrough(self) -> None:
        # An already-resolved directory is returned as-is.
        sub = self.tmp_path / "checkpoint"
        sub.mkdir()
        result = _resolve_user_model_path(str(sub), str(self.tmp_path))
        assert result == sub

    def test_missing_org_name_returns_none(self) -> None:
        # When the cache has no matching directory, the resolver
        # returns ``None`` so the factory can fall back gracefully.
        result = _resolve_user_model_path(
            "Qwen/Qwen2.5-0.5B-Instruct", str(self.tmp_path),
        )
        assert result is None

    def test_finds_huggingface_layout(self) -> None:
        target = self.tmp_path / "huggingface" / "Qwen" / "Qwen2.5-0.5B-Instruct"
        target.mkdir(parents=True)
        # Add a sentinel file so it is unambiguously a directory.
        (target / "config.json").write_text("{}")
        result = _resolve_user_model_path(
            "Qwen/Qwen2.5-0.5B-Instruct", str(self.tmp_path),
        )
        assert result == target

    def test_default_cache_root_is_under_home(self) -> None:
        root = _default_cache_root()
        assert root.startswith("~") or "/.cache/torcha-verse" in root


# ---------------------------------------------------------------------------
# 5. echo factory 输出含 model name + 提示
# ---------------------------------------------------------------------------
class TestEchoFactorySignals(unittest.TestCase):
    def test_echo_includes_model_name_when_provided(self) -> None:
        backend = _text_echo_factory()
        text = backend.generate(
            "how are you?",
            _echo_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        )
        assert "Qwen/Qwen2.5-0.5B-Instruct" in text
        assert "no model registered" in text
        assert "how are you?" in text

    def test_echo_without_model_name_omits_quotes(self) -> None:
        backend = _text_echo_factory()
        text = backend.generate("hi")
        assert "[echo-text: no model registered]" in text
        assert "hi" in text

    def test_echo_does_not_invoke_user_model_kwarg_in_real_backends(self) -> None:
        # The kwarg is consumed by echo; a real backend should be
        # able to ignore it without raising.  Here we just ensure
        # the call doesn't blow up for the placeholder path.
        backend = _text_echo_factory()
        result = backend.generate("hi", max_new_tokens=5)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6. _make_text_factory 在 cache 缺失时的 fallback 行为
# ---------------------------------------------------------------------------
class TestTextFactoryFallback(unittest.TestCase):
    def setUp(self) -> None:
        if is_local_runtime_enabled():
            disable_local_runtime()

    def tearDown(self) -> None:
        if is_local_runtime_enabled():
            disable_local_runtime()

    def test_missing_cache_warns_and_falls_back(self) -> None:
        # Point the factory at a non-existent cache.  We expect
        # the user-model path to warn and the micro-transformer
        # to be returned.
        cfg = RuntimeConfig(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            cache_root="/nonexistent/cache",
        )
        # ``is_local_runtime_enabled()`` is still False here, so
        # we instantiate the factory directly via the helper.
        from models.runtime.module_bus_runtime_switch import (
            _make_text_factory,
        )
        factory = _make_text_factory(cfg)
        with self.assertLogs(
            "models.runtime.module_bus_runtime_switch",
            level=logging.WARNING,
        ) as log_ctx:
            provider = factory()
        # The warning is either "not found in cache" or similar.
        assert any("cache" in msg.lower() for msg in log_ctx.output), (
            f"expected a cache warning, got {log_ctx.output}"
        )
        # Micro-transformer is a real provider (or None if torch
        # is unavailable in the sandbox -- we only assert type).
        if provider is not None:
            from models.providers.local_text import (
                LocalTorchTextProvider,
            )
            assert isinstance(provider, LocalTorchTextProvider)


if __name__ == "__main__":
    unittest.main()
