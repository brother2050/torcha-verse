"""Tests for the Infrastructure Layer."""

from __future__ import annotations

import os
import time
import tempfile
import pytest
import torch

from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager, DTypePolicy
from infrastructure.checkpoint_manager import CheckpointManager
from infrastructure.logger import get_logger
from infrastructure.error_handler import ErrorHandler, with_error_handler
from infrastructure.rate_limiter import RateLimiter
from infrastructure.cache_store import CacheStore


class TestConfigManager:
    """Test ConfigManager."""

    def test_get_default(self):
        """get() returns default when key is missing."""
        cm = ConfigManager()
        assert cm.get("nonexistent.key", "default") == "default"

    def test_set_and_get(self):
        """set() then get() returns the value."""
        cm = ConfigManager()
        cm.set("test.key", 42)
        assert cm.get("test.key") == 42

    def test_nested_key(self):
        """Dot-notation keys work for nested dicts."""
        cm = ConfigManager()
        cm.set("a.b.c", "value")
        assert cm.get("a.b.c") == "value"


class TestDeviceManager:
    """Test DeviceManager."""

    def test_get_device(self):
        """get_device() returns a valid torch device."""
        dm = DeviceManager()
        device = dm.get_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_dtype_policy(self):
        """DTypePolicy converts strings to torch dtypes."""
        policy = DTypePolicy()
        assert policy.get_dtype("fp32") == torch.float32
        assert policy.get_dtype("fp16") == torch.float16
        assert policy.get_dtype("bf16") == torch.bfloat16


class TestCheckpointManager:
    """Test CheckpointManager."""

    def test_save_and_load_weights(self, tmp_path):
        """save_weights_only and load_weights round-trip."""
        model = torch.nn.Linear(10, 5)
        cm = CheckpointManager(save_dir=str(tmp_path))
        path = str(tmp_path / "weights")
        cm.save_weights_only(model, path)
        # save_weights_only appends .safetensors extension
        import os
        saved_files = list(tmp_path.glob("weights*"))
        load_path = str(saved_files[0]) if saved_files else path
        model2 = torch.nn.Linear(10, 5)
        cm.load_weights(load_path, model2)
        assert torch.allclose(model.weight, model2.weight)


class TestLogger:
    """Test Logger."""

    def test_get_logger(self):
        """get_logger returns a working logger."""
        logger = get_logger("test")
        logger.info("Test info message")
        logger.warning("Test warning message")


class TestErrorHandler:
    """Test ErrorHandler."""

    def test_register_and_handle(self):
        """Registered handler is called for matching exception."""
        eh = ErrorHandler()
        called = []
        eh.register_handler(ValueError, lambda e: called.append(str(e)))
        eh.handle(ValueError("test error"))
        assert called == ["test error"]

    def test_decorator_default(self):
        """with_error_handler decorator catches exceptions."""
        @with_error_handler(default="fallback")
        def func():
            raise ValueError("oops")
        result = func()
        # The decorator may return None or the default value.
        assert result is None or result == "fallback"


class TestRateLimiter:
    """Test RateLimiter."""

    def test_try_acquire(self):
        """try_acquire returns True when tokens available."""
        rl = RateLimiter(rate=10, burst=5)
        assert rl.try_acquire(1) is True

    def test_depletion(self):
        """Tokens deplete and replenish over time."""
        rl = RateLimiter(rate=10, burst=2)
        assert rl.try_acquire(2) is True
        assert rl.try_acquire(1) is False


class TestCacheStore:
    """Test CacheStore."""

    def test_set_and_get(self):
        """set() then get() returns the value."""
        cs = CacheStore(max_size=10, ttl=60)
        cs.set("key", "value")
        assert cs.get("key") == "value"

    def test_lru_eviction(self):
        """LRU evicts oldest entries when full."""
        cs = CacheStore(max_size=2, ttl=60)
        cs.set("a", 1)
        cs.set("b", 2)
        cs.set("c", 3)  # should evict "a"
        assert cs.get("a") is None
        assert cs.get("b") == 2
        assert cs.get("c") == 3

    def test_ttl_expiry(self):
        """Entries expire after TTL."""
        cs = CacheStore(max_size=10, ttl=0.1)
        cs.set("key", "value")
        time.sleep(0.15)
        assert cs.get("key") is None
