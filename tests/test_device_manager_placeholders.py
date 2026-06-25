"""Tests for the safe-call wrappers in :mod:`infrastructure.device_manager`.

The :class:`DeviceManager` exposes ``tensor_parallel`` and
``pipeline_parallel`` as **placeholders** (see :doc:`/docs/DEFERRED_TASKS`
D3).  In single-GPU development environments they must NOT raise;
they should fall back to the original model with a logged warning.

These tests assert the safe-call contract:

* calling TP / PP on any ``nn.Module`` returns the same instance;
* a warning is emitted (but no exception escapes);
* the underlying placeholder is still importable for documentation.
"""
from __future__ import annotations

import logging

import pytest
import torch.nn as nn

from infrastructure.device_manager import (
    DeviceManager,
    _pipeline_parallel_impl,
    _tensor_parallel_impl,
)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class TestParallelPlaceholders:
    def test_tensor_parallel_returns_same_model(self, caplog: pytest.LogCaptureFixture) -> None:
        model = nn.Linear(4, 4)
        dm = DeviceManager()
        with caplog.at_level(logging.WARNING, logger="infrastructure.error_helper"):
            out = dm.tensor_parallel(model, num_devices=2)
        assert out is model, "TP placeholder must return the original model"
        assert any("tensor_parallel" in r.message for r in caplog.records)

    def test_pipeline_parallel_returns_same_model(self, caplog: pytest.LogCaptureFixture) -> None:
        model = nn.Linear(4, 4)
        dm = DeviceManager()
        with caplog.at_level(logging.WARNING, logger="infrastructure.error_helper"):
            out = dm.pipeline_parallel(model, num_stages=2, devices=None)
        assert out is model, "PP placeholder must return the original model"
        assert any("pipeline_parallel" in r.message for r in caplog.records)

    def test_tensor_parallel_does_not_raise(self) -> None:
        model = nn.Linear(4, 4)
        dm = DeviceManager()
        # Must not raise — single-GPU development environments rely on it.
        dm.tensor_parallel(model, num_devices=2)

    def test_pipeline_parallel_does_not_raise(self) -> None:
        model = nn.Linear(4, 4)
        dm = DeviceManager()
        dm.pipeline_parallel(model, num_stages=2, devices=None)


# ---------------------------------------------------------------------------
# Underlying placeholders (raise — used by the wrappers above)
# ---------------------------------------------------------------------------
class TestUnderlyingPlaceholders:
    def test_tensor_parallel_impl_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="DEFERRED_TASKS D3"):
            _tensor_parallel_impl(nn.Linear(4, 4), num_devices=2)

    def test_pipeline_parallel_impl_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="DEFERRED_TASKS D3"):
            _pipeline_parallel_impl(nn.Linear(4, 4), num_stages=2, devices=None)
