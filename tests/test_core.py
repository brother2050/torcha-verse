"""Tests for the Core Layer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.model_registry import ModelRegistry, BaseModel
from core.tokenizer_hub import TokenizerHub, TextTokenizer
from core.kv_cache_manager import KVCacheManager
from core.diffusion_scheduler import DiffusionScheduler, NoiseSchedule
from core.vocoder_manager import VocoderManager, HiFiGANVocoder
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry


class TestModelRegistry:
    """Test ModelRegistry."""

    def test_register_and_list(self):
        """register() then list_available() includes the model."""
        registry = ModelRegistry()
        registry.reset()

        class TestModel(BaseModel):
            def __init__(self, config=None):
                super().__init__(config=config)
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x)
            def generate(self, *args, **kwargs):
                return torch.zeros(1, 1)

        registry.register("test_model_core", TestModel, {})
        assert "test_model_core" in registry.list_available()

    def test_is_registered(self):
        """is_registered() returns correct boolean."""
        registry = ModelRegistry()

        class TestModel2(BaseModel):
            def __init__(self, config=None):
                super().__init__(config=config)
            def forward(self, x):
                return x
            def generate(self, *args, **kwargs):
                return torch.zeros(1, 1)

        registry.register("test_model_is_reg", TestModel2, {})
        assert registry.is_registered("test_model_is_reg") is True
        assert registry.is_registered("nonexistent") is False


class TestTokenizerHub:
    """Test TokenizerHub."""

    def test_register_and_get(self):
        """register_tokenizer then get_tokenizer returns instance."""
        hub = TokenizerHub()
        hub.reset()
        hub.register_tokenizer("test_text_tok", TextTokenizer)
        assert hub.is_registered("test_text_tok") is True

    def test_text_tokenizer_encode_decode(self):
        """TextTokenizer encode/decode round-trip."""
        tok = TextTokenizer(vocab_size=256)
        tokens = tok.encode("hello")
        assert isinstance(tokens, (list, torch.Tensor))


class TestKVCacheManager:
    """Test KVCacheManager."""

    def test_static_allocate_and_update(self):
        """Static strategy allocates and updates cache."""
        mgr = KVCacheManager(
            strategy="static",
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_batch_size=2,
            max_seq_len=16,
        )
        mgr.allocate(batch_size=1, seq_len=4)
        key = torch.randn(1, 4, 4, 8)
        value = torch.randn(1, 4, 4, 8)
        mgr.update(layer_idx=0, key=key, value=value)
        k, v = mgr.get(layer_idx=0, batch_idx=0)
        assert k is not None

    def test_paged_allocate(self):
        """Paged strategy allocates pages."""
        mgr = KVCacheManager(
            strategy="paged",
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_batch_size=2,
            max_seq_len=16,
            page_size=4,
            max_pages=32,
        )
        mgr.allocate(batch_size=1, seq_len=4)
        key = torch.randn(1, 4, 4, 8)
        value = torch.randn(1, 4, 4, 8)
        mgr.update(layer_idx=0, key=key, value=value)


class TestDiffusionScheduler:
    """Test DiffusionScheduler."""

    def test_noise_schedule_linear(self):
        """Linear noise schedule produces valid betas."""
        ns = NoiseSchedule(num_timesteps=100, strategy="linear")
        assert ns.betas.shape[0] == 100
        assert (ns.betas >= 0).all()

    def test_add_noise(self):
        """add_noise returns same shape as input."""
        sched = DiffusionScheduler(num_timesteps=100, device="cpu")
        x = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(x)
        t = torch.tensor([50])
        noisy = sched.add_noise(x, t, noise)
        assert noisy.shape == x.shape

    def test_step(self):
        """step() returns same shape as input."""
        sched = DiffusionScheduler(num_timesteps=100, sampler_name="ddim", device="cpu")
        sched.set_timesteps(5)
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        t = sched.timesteps[0]
        prev = sched.step(model_output, t, sample)
        assert prev.shape == sample.shape


class TestVocoderManager:
    """Test VocoderManager."""

    def test_register_and_list(self):
        """register_vocoder then list_available includes it."""
        mgr = VocoderManager()
        mgr.reset()
        mgr.register_vocoder("test_vocoder", HiFiGANVocoder)
        assert "test_vocoder" in mgr.list_available()


class TestMemoryManager:
    """Test MemoryManager."""

    def test_estimate_memory(self):
        """estimate_memory returns a positive value."""
        mgr = MemoryManager()
        est = mgr.estimate_memory(model_params=1_000_000, batch_size=1)
        assert est > 0

    def test_gc_collect(self):
        """gc_collect runs without error."""
        mgr = MemoryManager()
        mgr.gc_collect()


class TestToolRegistry:
    """Test ToolRegistry."""

    def test_register_and_execute(self):
        """register_tool then execute_tool works."""
        registry = ToolRegistry()
        registry.reset()

        def adder(a, b):
            return a + b

        registry.register_tool(
            name="add",
            func=adder,
            description="Add two numbers",
            parameter_schema={"a": {"type": "integer"}, "b": {"type": "integer"}},
        )
        result = registry.execute_tool("add", {"a": 3, "b": 4})
        assert result.success is True
        assert result.output == 7

    def test_get_tool_descriptions(self):
        """get_tool_descriptions returns list of dicts."""
        registry = ToolRegistry()
        registry.reset()
        registry.register_tool(
            name="noop",
            func=lambda: None,
            description="Does nothing",
            parameter_schema={},
        )
        descs = registry.get_tool_descriptions()
        assert len(descs) >= 1
        assert descs[0]["name"] == "noop"
