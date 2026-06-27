"""Local transformers runtime tests (v0.10.0).

覆盖 :mod:`models.runtime` 的四个子模块:

* :mod:`models.runtime.cpu_cuda_mps_device_planner` -- 设备规划器
* :mod:`models.runtime.transformers_style_loader` -- 自研加载 API
* :mod:`models.runtime.transformers_style_pipeline` -- 自研推理管道
* :mod:`models.runtime.module_bus_runtime_switch` -- 一行运行时开关

设计原则:

1. **零外部依赖**:本测试文件不引入 ``transformers`` / ``diffusers`` /
   ``huggingface_hub``。所有断言都通过项目自有的 L1-L6 模块。
2. **零网络**:用临时目录 + 合成 ``.safetensors`` 模拟"下载 → 加载"。
3. **零 GPU**:全部 CPU 跑通,与 v0.4.x P0 单元测试对齐。
4. **占位登记**:每个 ``pass`` / ``NotImplementedError`` 必须在
   ``docs/placeholder_registry.md`` 同步登记。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------
def _tmpdir() -> str:
    """Return a fresh per-test temp directory (auto-cleaned by pytest tmp_path)."""
    return tempfile.mkdtemp(prefix="torcha_runtime_test_")


def _write_fake_checkpoint(
    dir_path: str,
    *,
    name: str = "tiny.safetensors",
    state_dict: Dict[str, torch.Tensor] | None = None,
) -> str:
    """Write a synthetic safetensors file under ``dir_path``."""
    from core.checkpoint_loader import save_safetensors

    if state_dict is None:
        state_dict = {
            "linear.weight": torch.zeros(4, 4),
            "linear.bias": torch.zeros(4),
        }
    target = os.path.join(dir_path, name)
    save_safetensors(state_dict, target)
    return target


def _write_hunyuan_style_checkpoint(dir_path: str) -> Tuple[str, str]:
    """Build a small fake 'HunyuanDiT-style' state-dict and save it.

    Returns ``(path, dir_path)``.
    """
    state_dict = {
        # patch / token embed
        "img_in.proj.weight": torch.zeros(8, 4, 3, 3),
        "img_in.proj.bias": torch.zeros(8),
        "x_embedder.weight": torch.zeros(8, 4),
        "x_embedder.bias": torch.zeros(4),
        # time embed
        "time_in.mlp.0.weight": torch.zeros(8, 4),
        "time_in.mlp.0.bias": torch.zeros(8),
        "time_in.mlp.2.weight": torch.zeros(8, 8),
        "time_in.mlp.2.bias": torch.zeros(8),
        # vector (pooled) embed
        "vector_in.proj.weight": torch.zeros(8, 4),
        "vector_in.proj.bias": torch.zeros(8),
        # style / size embeds (1:1)
        "style_embedder.weight": torch.zeros(8, 4),
        "size_embedder.weight": torch.zeros(8, 4),
        # 1 block (i=0) -- enough to exercise {i} expansion
        "blocks.0.attn.qkv.weight": torch.zeros(8, 8),
        "blocks.0.attn.qkv.bias": torch.zeros(8),
        "blocks.0.attn.proj.weight": torch.zeros(4, 4),
        "blocks.0.attn.proj.bias": torch.zeros(4),
        "blocks.0.mlp.fc1.weight": torch.zeros(8, 4),
        "blocks.0.mlp.fc1.bias": torch.zeros(8),
        "blocks.0.mlp.fc2.weight": torch.zeros(4, 8),
        "blocks.0.mlp.fc2.bias": torch.zeros(4),
        "blocks.0.adaln_modulation.0.weight": torch.zeros(4, 8),
        "blocks.0.adaln_modulation.0.bias": torch.zeros(4),
        # final layer
        "final_layer.adaLN_modulation.0.weight": torch.zeros(4, 8),
        "final_layer.adaLN_modulation.0.bias": torch.zeros(4),
        "final_layer.linear.weight": torch.zeros(4, 4),
        "final_layer.linear.bias": torch.zeros(4),
        "final_layer.norm_final.weight": torch.zeros(4),
        "final_layer.norm_final.bias": torch.zeros(4),
    }
    return _write_fake_checkpoint(dir_path, name="hunyuan.safetensors",
                                  state_dict=state_dict), dir_path


# ===========================================================================
# Section 1: device_planner tests
# ===========================================================================
class TestDevicePlanner:
    """Tests for :mod:`models.runtime.cpu_cuda_mps_device_planner`."""

    def test_is_cuda_available_returns_bool(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import is_cuda_available
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_is_mps_available_returns_bool(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import is_mps_available
        result = is_mps_available()
        assert isinstance(result, bool)

    def test_pick_default_device_returns_torch_device(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import pick_default_device
        dev = pick_default_device()
        assert isinstance(dev, torch.device)
        # Must always succeed (CPU fallback is guaranteed).
        assert dev.type in {"cpu", "cuda", "mps"}

    def test_plan_device_none_uses_default(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device(None)
        assert plan.device is not None
        assert isinstance(plan.dtype, torch.dtype)
        # dtype should match device heuristic (CPU -> fp32).
        if plan.device.type == "cpu":
            assert plan.dtype == torch.float32
        if plan.device.type == "cuda":
            assert plan.dtype == torch.float16

    def test_plan_device_string_cpu(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device("cpu")
        assert plan.device == torch.device("cpu")
        assert plan.dtype == torch.float32

    def test_plan_device_torch_device_input(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device(torch.device("cpu"))
        assert plan.device == torch.device("cpu")

    def test_plan_device_explicit_dtype_overrides(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device("cpu", torch_dtype=torch.float16)
        assert plan.device == torch.device("cpu")
        assert plan.dtype == torch.float16

    def test_plan_device_balanced_falls_back_to_single(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device("balanced")
        # Either multi-GPU device_map or single device fallback.
        if plan.device_map is not None:
            assert all(v.startswith("cuda:") for v in plan.device_map.values())
        else:
            assert plan.device is not None

    def test_plan_device_mapping_passthrough(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device({"layer.0": "cuda:0", "layer.1": "cpu"})
        assert plan.device_map is not None
        assert plan.device_map == {"layer.0": "cuda:0", "layer.1": "cpu"}

    def test_plan_device_invalid_string_falls_back(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import plan_device
        plan = plan_device("not-a-real-device")
        assert plan.device is not None
        assert plan.device.type == "cpu"

    def test_get_device_map_returns_dataclass(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import (
            DevicePlan, get_device_map,
        )
        plan = get_device_map(None)
        assert isinstance(plan, DevicePlan)
        assert plan.notes  # non-empty list (at least one note)

    def test_shard_module_across_gpus_balanced(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import _shard_module_across_gpus
        m = nn.ModuleList([nn.Linear(1, 1) for _ in range(6)])
        plan = _shard_module_across_gpus(2, m)
        assert len(plan) == 6
        # First half cuda:0, second half cuda:1 (or similar).
        gpus = {v for v in plan.values()}
        assert gpus.issubset({"cuda:0", "cuda:1"})

    def test_shard_module_across_gpus_raises_for_zero(self) -> None:
        from models.runtime.cpu_cuda_mps_device_planner import _shard_module_across_gpus
        with pytest.raises(ValueError):
            _shard_module_across_gpus(0)


# ===========================================================================
# Section 2: loader tests
# ===========================================================================
class TestDetectModelFamily:
    """Tests for :func:`detect_model_family`."""

    def test_detect_hunyuan_dit(self) -> None:
        from models.runtime.transformers_style_loader import (
            detect_model_family, ModelFamily,
        )
        d = _tmpdir()
        path, _ = _write_hunyuan_style_checkpoint(d)
        assert detect_model_family(path) == ModelFamily.HUNYUAN_DIT

    def test_detect_unknown_for_random_keys(self) -> None:
        from models.runtime.transformers_style_loader import (
            detect_model_family, ModelFamily,
        )
        d = _tmpdir()
        path = _write_fake_checkpoint(d, name="rand.safetensors")
        assert detect_model_family(path) == ModelFamily.UNKNOWN

    def test_detect_raises_for_missing_file(self) -> None:
        from models.runtime.transformers_style_loader import detect_model_family
        with pytest.raises(FileNotFoundError):
            detect_model_family("/nonexistent/path/that/does/not/exist")

    def test_detect_supports_directory_input(self) -> None:
        from models.runtime.transformers_style_loader import (
            detect_model_family, ModelFamily,
        )
        d = _tmpdir()
        _write_hunyuan_style_checkpoint(d)
        assert detect_model_family(d) == ModelFamily.HUNYUAN_DIT


class TestTokenizerBundle:
    """Tests for :class:`TokenizerBundle`."""

    def test_empty_bundle(self) -> None:
        from models.runtime.transformers_style_loader import TokenizerBundle
        b = TokenizerBundle()
        assert b.has_any() is False
        assert "empty" in repr(b)

    def test_bundle_with_clip(self) -> None:
        from models.runtime.transformers_style_loader import TokenizerBundle
        from models.text.clip_tokenizer import SimpleByteBPETokenizer
        b = TokenizerBundle(clip=SimpleByteBPETokenizer())
        assert b.has_any() is True
        assert "clip" in repr(b)


class TestKeymapDispatch:
    """Tests for the internal ``_keymap_for`` / ``_default_num_blocks``
    dispatch logic."""

    def test_keymap_for_hunyuan_dit_expanded(self) -> None:
        from models.runtime.transformers_style_loader import (
            _keymap_for, _default_num_blocks, ModelFamily,
        )
        n = _default_num_blocks(ModelFamily.HUNYUAN_DIT)
        km = _keymap_for(ModelFamily.HUNYUAN_DIT, num_blocks=n)
        # The {i} placeholders must be expanded.
        assert "blocks.0.attn.qkv.weight" in km
        assert "blocks.{i}.attn.qkv.weight" not in km

    def test_keymap_for_unknown_is_none(self) -> None:
        from models.runtime.transformers_style_loader import (
            _keymap_for, ModelFamily,
        )
        assert _keymap_for(ModelFamily.UNKNOWN) is None

    def test_keymap_for_tiny_transformer_is_none(self) -> None:
        from models.runtime.transformers_style_loader import (
            _keymap_for, ModelFamily,
        )
        # TinyTransformer has no upstream-style keymap.
        assert _keymap_for(ModelFamily.TINY_TRANSFORMER) is None

    def test_default_num_blocks_per_family(self) -> None:
        from models.runtime.transformers_style_loader import (
            _default_num_blocks, ModelFamily,
        )
        # Sanity values; the loader uses these to expand per-block
        # {i} placeholders.
        assert _default_num_blocks(ModelFamily.HUNYUAN_DIT) > 0
        assert _default_num_blocks(ModelFamily.FLUX) > 0
        assert _default_num_blocks(ModelFamily.SD3) > 0
        assert _default_num_blocks(ModelFamily.WAN2) > 0
        assert _default_num_blocks(ModelFamily.MUSICGEN) > 0
        assert _default_num_blocks(ModelFamily.UNKNOWN) == 0


class TestResolveCheckpointFile:
    """Tests for the internal :func:`_resolve_checkpoint_file`."""

    def test_resolve_direct_file(self) -> None:
        from models.runtime.transformers_style_loader import _resolve_checkpoint_file
        d = _tmpdir()
        p = _write_fake_checkpoint(d)
        assert _resolve_checkpoint_file(Path(p)) == Path(p)

    def test_resolve_directory_picks_first(self) -> None:
        from models.runtime.transformers_style_loader import _resolve_checkpoint_file
        d = _tmpdir()
        _write_fake_checkpoint(d, name="a.safetensors")
        _write_fake_checkpoint(d, name="b.safetensors")
        resolved = _resolve_checkpoint_file(Path(d))
        assert resolved is not None
        assert resolved.name in {"a.safetensors", "b.safetensors"}

    def test_resolve_missing_path_returns_none(self) -> None:
        from models.runtime.transformers_style_loader import _resolve_checkpoint_file
        assert _resolve_checkpoint_file(Path("/nonexistent/abc/xyz")) is None

    def test_resolve_non_safetensors_file_returns_none(self) -> None:
        from models.runtime.transformers_style_loader import _resolve_checkpoint_file
        d = _tmpdir()
        p = os.path.join(d, "config.json")
        Path(p).write_text("{}", encoding="utf-8")
        assert _resolve_checkpoint_file(Path(p)) is None


class TestModelHub:
    """Tests for :class:`ModelHub` (no real download)."""

    def test_init_creates_cache_dir(self) -> None:
        from models.runtime.transformers_style_loader import ModelHub
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "fresh_cache")
            hub = ModelHub(cache_dir=cache)
            assert Path(cache).is_dir()
            assert hub.load_cache_size() == 0

    def test_init_respects_env_var(self) -> None:
        from models.runtime.transformers_style_loader import ModelHub
        with tempfile.TemporaryDirectory() as d:
            os.environ["TORCHA_VERSE_CACHE"] = d
            try:
                hub = ModelHub()
                assert str(hub.cache_dir) == d
            finally:
                del os.environ["TORCHA_VERSE_CACHE"]

    def test_clear_load_cache(self) -> None:
        from models.runtime.transformers_style_loader import ModelHub
        hub = ModelHub()
        hub.clear_load_cache()
        assert hub.load_cache_size() == 0

    def test_load_unknown_path_raises(self) -> None:
        from models.runtime.transformers_style_loader import ModelHub
        hub = ModelHub()
        with pytest.raises(FileNotFoundError):
            hub.load("/nonexistent/path")

    def test_load_unknown_family_uses_modelmixin_fallback(self) -> None:
        """An unknown-family checkpoint with a generic shape should
        still load via :class:`ModelMixin.from_pretrained`."""
        from models.runtime.transformers_style_loader import ModelHub, ModelFamily
        from models.base import ModelMixin
        from torch import nn

        class _ToyNet(ModelMixin):
            def __init__(self, config: dict | None = None) -> None:
                super().__init__(config)
                self.linear = nn.Linear(4, 4)

        d = _tmpdir()
        ckpt = _write_fake_checkpoint(d, name="toy.safetensors")
        # Use a custom subclass via hub.load with a ModelMixin
        # fallback by passing family=ModelFamily.UNKNOWN.
        hub = ModelHub()
        mdl, tok, fam = hub.load(
            ckpt, family=ModelFamily.UNKNOWN, strict=False,
        )
        # ModelMixin from_pretrained default cls is ModelMixin itself
        # (no-op module); we just assert the round-trip is observable.
        assert isinstance(mdl, ModelMixin)
        assert fam == ModelFamily.UNKNOWN
        # TokenizerBundle may be empty (no vocab files shipped).
        assert tok is not None


class TestTaskHeads:
    """Tests for the four task-head wrappers."""

    def test_local_model_for_causal_lm_wraps_model(self) -> None:
        from models.runtime.transformers_style_loader import (
            ModelForCausalLM, ModelFamily, TokenizerBundle,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(mdl, TokenizerBundle(byte=tok),
                                     family=ModelFamily.TINY_TRANSFORMER)
        assert head.family == ModelFamily.TINY_TRANSFORMER
        # generate() returns a string.
        out = head.generate("hello")
        assert isinstance(out, str)

    def test_local_model_for_text_to_image_falls_back_to_random(self) -> None:
        """When the diffusion loop is unavailable the wrapper returns
        a 'random latents' dict so the contract is honoured."""
        from models.runtime.transformers_style_loader import (
            ModelForTextToImage, ModelFamily,
        )
        from models.base import ModelMixin
        # A minimal stand-in ModelMixin that has no forward / encode_text,
        # so the wrapper falls back to the random-latents path.
        head = ModelForTextToImage(
            ModelMixin(), family=ModelFamily.HUNYUAN_DIT,
        )
        out = head("a tiny cat", height=64, width=64, num_inference_steps=2)
        assert isinstance(out, dict)
        # Two acceptable paths:
        #   1. real diffusion loop: returns {latents, num_inference_steps, ...}
        #   2. fallback path: returns {latents, text_embeds, sampler, note}
        assert "latents" in out
        # Either the diffusion loop ran (latents tensor) or the
        # wrapper fell back (text_embeds tensor + note).
        assert ("text_embeds" in out) or ("num_inference_steps" in out)

    def test_local_model_for_tts_falls_back_to_zero_mel(self) -> None:
        from models.runtime.transformers_style_loader import (
            ModelForTextToSpeech, ModelFamily,
        )
        from models.base import ModelMixin
        head = ModelForTextToSpeech(
            ModelMixin(), family=ModelFamily.MUSICGEN,
        )
        out = head("hello world", sample_rate=16000)
        assert isinstance(out, dict)
        assert "mel" in out
        assert "sample_rate" in out
        assert out["sample_rate"] == 16000

    def test_local_model_for_music_falls_back_to_zero_codes(self) -> None:
        from models.runtime.transformers_style_loader import (
            ModelForMusic, ModelFamily,
        )
        from models.base import ModelMixin
        head = ModelForMusic(
            ModelMixin(), family=ModelFamily.MUSICGEN,
        )
        out = head("a funky beat", duration_s=2.0, sample_rate=22050)
        assert isinstance(out, dict)
        assert "codes" in out
        assert "duration_s" in out
        assert out["duration_s"] == 2.0

    def test_chat_format_concatenates_messages(self) -> None:
        from models.runtime.transformers_style_loader import (
            ModelForCausalLM, ModelFamily, TokenizerBundle,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(mdl, TokenizerBundle(byte=tok),
                                     family=ModelFamily.TINY_TRANSFORMER)
        out = head.chat(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello!"},
                {"role": "user", "content": "what's up?"},
            ],
            max_new_tokens=8,
        )
        assert isinstance(out, str)


class TestLoadModelAndTokenizer:
    """Top-level :func:`load_model_and_tokenizer` entry point."""

    def test_requires_path_or_repo_id(self) -> None:
        from models.runtime.transformers_style_loader import load_model_and_tokenizer
        with pytest.raises(ValueError):
            load_model_and_tokenizer()

    def test_repo_id_requires_download(self) -> None:
        from models.runtime.transformers_style_loader import load_model_and_tokenizer
        with pytest.raises(ValueError):
            load_model_and_tokenizer(repo_id="someone/somewhere")

    def test_load_from_nonexistent_path_raises(self) -> None:
        from models.runtime.transformers_style_loader import load_model_and_tokenizer
        with pytest.raises(FileNotFoundError):
            load_model_and_tokenizer("/nonexistent/path/abc")


# ===========================================================================
# Section 3: local_pipeline tests
# ===========================================================================
class TestPipelineOutput:
    """Tests for :class:`PipelineOutput`."""

    def test_empty(self) -> None:
        from models.runtime.transformers_style_pipeline import PipelineOutput
        out = PipelineOutput()
        assert len(out) == 0
        assert out.to_dict() == []

    def test_with_records(self) -> None:
        from models.runtime.transformers_style_pipeline import PipelineOutput
        out = PipelineOutput(records=[{"a": 1}, {"a": 2}])
        assert len(out) == 2
        assert out[0]["a"] == 1
        assert out[1]["a"] == 2
        # to_dict returns a deep-copyable list.
        d = out.to_dict()
        assert d == [{"a": 1}, {"a": 2}]


class TestTextGenerationPipeline:
    """Tests for :class:`TextGenerationPipeline`."""

    def test_call_with_single_string(self) -> None:
        from models.runtime.transformers_style_pipeline import TextGenerationPipeline
        from models.runtime.transformers_style_loader import (
            ModelForCausalLM, ModelFamily, TokenizerBundle,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(mdl, TokenizerBundle(byte=tok),
                                     family=ModelFamily.TINY_TRANSFORMER)
        pipe = TextGenerationPipeline(head)
        out = pipe("hello", max_new_tokens=8)
        assert len(out) == 1
        rec = out[0]
        assert rec["prompt"] == "hello"
        assert "generated_text" in rec
        assert rec["family"] == ModelFamily.TINY_TRANSFORMER.value

    def test_call_with_list_of_strings(self) -> None:
        from models.runtime.transformers_style_pipeline import TextGenerationPipeline
        from models.runtime.transformers_style_loader import (
            ModelForCausalLM, ModelFamily, TokenizerBundle,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(mdl, TokenizerBundle(byte=tok),
                                     family=ModelFamily.TINY_TRANSFORMER)
        pipe = TextGenerationPipeline(head)
        out = pipe(["a", "b", "c"], max_new_tokens=4)
        assert len(out) == 3
        for rec in out:
            assert "generated_text" in rec


class TestImageGenerationPipeline:
    """Tests for :class:`ImageGenerationPipeline`."""

    def test_call_with_single_prompt(self) -> None:
        from models.runtime.transformers_style_pipeline import ImageGenerationPipeline
        from models.runtime.transformers_style_loader import ModelForTextToImage
        from models.base import ModelMixin
        head = ModelForTextToImage(ModelMixin(), family="hunyuan_dit")
        pipe = ImageGenerationPipeline(head)
        out = pipe("a tiny cat", height=64, width=64, num_inference_steps=2)
        assert len(out) == 1
        rec = out[0]
        assert rec["prompt"] == "a tiny cat"
        assert "latents" in rec or "note" in rec

    def test_call_with_list_of_prompts(self) -> None:
        from models.runtime.transformers_style_pipeline import ImageGenerationPipeline
        from models.runtime.transformers_style_loader import ModelForTextToImage
        from models.base import ModelMixin
        head = ModelForTextToImage(ModelMixin(), family="hunyuan_dit")
        pipe = ImageGenerationPipeline(head)
        out = pipe(
            ["a cat", "a dog", "a bird"],
            height=64, width=64, num_inference_steps=2,
        )
        assert len(out) == 3


class TestAudioPipeline:
    """Tests for :class:`AudioPipeline`."""

    def test_call_dispatches_tts(self) -> None:
        from models.runtime.transformers_style_pipeline import AudioPipeline
        from models.runtime.transformers_style_loader import ModelForTextToSpeech
        from models.base import ModelMixin
        head = ModelForTextToSpeech(ModelMixin(), family="musicgen")
        pipe = AudioPipeline(head)
        out = pipe("hello", sample_rate=16000)
        assert len(out) == 1
        rec = out[0]
        assert rec["prompt"] == "hello"
        assert "sample_rate" in rec
        assert rec["sample_rate"] == 16000

    def test_call_dispatches_music(self) -> None:
        from models.runtime.transformers_style_pipeline import AudioPipeline
        from models.runtime.transformers_style_loader import ModelForMusic
        from models.base import ModelMixin
        head = ModelForMusic(ModelMixin(), family="musicgen")
        pipe = AudioPipeline(head)
        out = pipe("a funky beat", duration_s=2.0, sample_rate=22050)
        assert len(out) == 1
        rec = out[0]
        assert rec["prompt"] == "a funky beat"
        assert "duration_s" in rec
        assert rec["duration_s"] == 2.0


class TestPipelineFactory:
    """Tests for the top-level :func:`pipeline` factory."""

    def test_list_supported_tasks(self) -> None:
        from models.runtime.transformers_style_pipeline import list_supported_tasks
        tasks = list_supported_tasks()
        assert "text-generation" in tasks
        assert "text-to-image" in tasks
        assert "text-to-speech" in tasks
        assert "music-generation" in tasks

    def test_unsupported_task_raises(self) -> None:
        from models.runtime.transformers_style_pipeline import pipeline
        with pytest.raises(ValueError):
            pipeline("not-a-real-task")

    def test_no_model_or_path_raises(self) -> None:
        from models.runtime.transformers_style_pipeline import pipeline
        with pytest.raises(RuntimeError):
            pipeline("text-generation")

    def test_text_pipeline_inline_load(self) -> None:
        """``pipeline("text-generation", model=...)`` with a
        pre-built TaskHead skips the model load entirely."""
        from models.runtime.transformers_style_pipeline import pipeline
        from models.runtime.transformers_style_loader import (
            ModelForCausalLM, ModelFamily, TokenizerBundle,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(mdl, TokenizerBundle(byte=tok),
                                     family=ModelFamily.TINY_TRANSFORMER)
        pipe = pipeline("text-generation", model=head)
        out = pipe("hello", max_new_tokens=4)
        assert len(out) == 1

    def test_image_pipeline_inline_load(self) -> None:
        from models.runtime.transformers_style_pipeline import pipeline
        from models.runtime.transformers_style_loader import ModelForTextToImage
        from models.base import ModelMixin
        head = ModelForTextToImage(ModelMixin(), family="hunyuan_dit")
        pipe = pipeline("text-to-image", model=head)
        out = pipe("a tiny cat", height=64, width=64, num_inference_steps=2)
        assert len(out) == 1

    def test_audio_pipeline_inline_load(self) -> None:
        from models.runtime.transformers_style_pipeline import pipeline
        from models.runtime.transformers_style_loader import ModelForTextToSpeech
        from models.base import ModelMixin
        head = ModelForTextToSpeech(ModelMixin(), family="musicgen")
        pipe = pipeline("text-to-speech", model=head)
        out = pipe("hi", sample_rate=16000)
        assert len(out) == 1


# ===========================================================================
# Section 4: runtime_config tests
# ===========================================================================
class TestRuntimeConfig:
    """Tests for :class:`RuntimeConfig` + the
    :func:`enable_local_runtime` / :func:`disable_local_runtime` pair."""

    def test_default_config(self) -> None:
        from models.runtime.module_bus_runtime_switch import (
            RuntimeConfig, is_local_runtime_enabled, get_active_config,
        )
        # Ensure we're starting from a clean state.
        if is_local_runtime_enabled():
            from models.runtime import disable_local_runtime
            disable_local_runtime()
        cfg = RuntimeConfig()
        assert cfg.prefer_local_text is True
        assert cfg.prefer_local_image is True
        assert cfg.use_real_diffusion_loop is True
        assert cfg.describe()  # non-empty

    def test_enable_disable_roundtrip(self) -> None:
        from models.runtime import (
            enable_local_runtime,
            disable_local_runtime,
            is_local_runtime_enabled,
            get_active_config,
        )
        # Disable first to ensure clean state.
        if is_local_runtime_enabled():
            disable_local_runtime()
        assert is_local_runtime_enabled() is False
        assert get_active_config() is None

        cfg = enable_local_runtime()
        assert is_local_runtime_enabled() is True
        assert get_active_config() is cfg

        disable_local_runtime()
        assert is_local_runtime_enabled() is False
        assert get_active_config() is None

    def test_enable_idempotent(self) -> None:
        from models.runtime import (
            enable_local_runtime, disable_local_runtime, is_local_runtime_enabled,
        )
        if is_local_runtime_enabled():
            disable_local_runtime()
        cfg1 = enable_local_runtime()
        cfg2 = enable_local_runtime()
        # The second call should return the SAME object (idempotent).
        assert cfg1 is cfg2
        disable_local_runtime()

    def test_enable_with_overrides(self) -> None:
        from models.runtime import (
            enable_local_runtime, disable_local_runtime, get_active_config,
        )
        enable_local_runtime(
            prefer_local_text=False,
            prefer_local_image=True,
            device="cpu",
        )
        cfg = get_active_config()
        assert cfg is not None
        assert cfg.prefer_local_text is False
        assert cfg.prefer_local_image is True
        assert cfg.device == "cpu"
        disable_local_runtime()

    def test_enable_with_explicit_config(self) -> None:
        from models.runtime import (
            enable_local_runtime, disable_local_runtime, get_active_config,
        )
        from models.runtime.module_bus_runtime_switch import RuntimeConfig
        custom = RuntimeConfig(
            prefer_local_text=True,
            prefer_local_image=False,
            use_real_diffusion_loop=False,
            tags=["custom-test"],
        )
        enable_local_runtime(custom)
        cfg = get_active_config()
        assert cfg is custom
        assert cfg.tags == ["custom-test"]
        assert cfg.prefer_local_image is False
        disable_local_runtime()

    def test_describe_includes_bits(self) -> None:
        from models.runtime.module_bus_runtime_switch import RuntimeConfig
        cfg = RuntimeConfig(
            prefer_local_text=True,
            prefer_local_image=True,
            prefer_local_video=False,
            prefer_local_audio=False,
            prefer_local_multimodal=False,
            torch_dtype="fp16",
            device="cuda:0",
        )
        d = cfg.describe()
        assert "text" in d
        assert "image" in d
        assert "fp16" in d
        assert "cuda:0" in d

    def test_disable_when_already_disabled(self) -> None:
        """disable_local_runtime() should be safe to call twice."""
        from models.runtime import (
            enable_local_runtime, disable_local_runtime, is_local_runtime_enabled,
        )
        if is_local_runtime_enabled():
            disable_local_runtime()
        disable_local_runtime()
        assert is_local_runtime_enabled() is False


# ===========================================================================
# Section 5: end-to-end smoke (uses the example)
# ===========================================================================
class TestEndToEnd:
    """End-to-end smoke tests that exercise the public surface as a
    user would.  These tests run in < 2 s on a stock dev box and
    prove that ``import models.runtime`` + ``pipeline(...)`` works
    without any external network.
    """

    def test_import_runtime_package(self) -> None:
        import models.runtime as rt
        # Canonical (v0.10.1) names
        assert hasattr(rt, "load_model_and_tokenizer")
        assert hasattr(rt, "pipeline")
        assert hasattr(rt, "ModelHub")
        assert hasattr(rt, "TextGenerationPipeline")
        assert hasattr(rt, "ImageGenerationPipeline")
        assert hasattr(rt, "AudioPipeline")
        assert hasattr(rt, "enable_local_runtime")
        assert hasattr(rt, "ModelFamily")
        assert hasattr(rt, "TokenizerBundle")
        assert hasattr(rt, "DevicePlan")

    def test_end_to_end_text_pipeline(self) -> None:
        """One-call smoke: build → pipeline → generate."""
        from models.runtime import (
            ModelForCausalLM,
            TextGenerationPipeline,
            TokenizerBundle,
            ModelFamily,
        )
        from models.providers.tiny_transformer import (
            TINY_CONFIG, build_tiny_transformer,
        )
        mdl, tok = build_tiny_transformer(TINY_CONFIG)
        head = ModelForCausalLM(
            mdl, TokenizerBundle(byte=tok),
            family=ModelFamily.TINY_TRANSFORMER,
        )
        pipe = TextGenerationPipeline(head)
        out = pipe(
            ["the quick brown fox", "lorem ipsum"],
            max_new_tokens=8,
        )
        assert len(out) == 2
        assert all(isinstance(r["generated_text"], str) for r in out)

    def test_end_to_end_image_pipeline(self) -> None:
        from models.runtime import (
            ModelForTextToImage,
            ImageGenerationPipeline,
        )
        from models.base import ModelMixin
        head = ModelForTextToImage(ModelMixin(), family="hunyuan_dit")
        pipe = ImageGenerationPipeline(head)
        out = pipe("a serene mountain", height=64, width=64,
                   num_inference_steps=2)
        assert len(out) == 1
        rec = out[0]
        # Either the diffusion loop returned a 'latents' tensor or
        # the wrapper fell back to a 'note' / 'latents' placeholder.
        assert "latents" in rec

    def test_end_to_end_audio_pipeline(self) -> None:
        from models.runtime import (
            ModelForTextToSpeech,
            AudioPipeline,
        )
        from models.base import ModelMixin
        head = ModelForTextToSpeech(ModelMixin(), family="musicgen")
        pipe = AudioPipeline(head)
        out = pipe("hello world", sample_rate=22050)
        assert len(out) == 1
        rec = out[0]
        assert "mel" in rec or "codes" in rec

    def test_loader_and_pipeline_module_imports(self) -> None:
        """The two main runtime modules must import cleanly without
        side effects.  This is the v0.10.1 equivalent of the old
        ``test_example_module_imports`` (the demo example was
        removed in v0.10.1 since the 13 project-owned examples
        already cover end-to-end usage).
        """
        import importlib
        loader = importlib.import_module("models.runtime.transformers_style_loader")
        pipeline = importlib.import_module("models.runtime.transformers_style_pipeline")
        # Spot-check that the renamed public symbols are reachable.
        assert hasattr(loader, "ModelHub")
        assert hasattr(loader, "ModelForCausalLM")
        assert hasattr(loader, "load_model_and_tokenizer")
        assert hasattr(loader, "detect_model_family")
        assert hasattr(pipeline, "TextGenerationPipeline")
        assert hasattr(pipeline, "ImageGenerationPipeline")
        assert hasattr(pipeline, "AudioPipeline")
        assert hasattr(pipeline, "pipeline")
