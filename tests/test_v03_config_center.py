"""Tests for v0.3.0 ConfigCenter."""
import pytest
from infrastructure.config_center import ConfigCenter

def test_config_center_singleton():
    cc1 = ConfigCenter()
    cc2 = ConfigCenter()
    assert cc1 is cc2

def test_config_center_get_set():
    cc = ConfigCenter()
    cc.reset()
    cc.set("test.key", "value")
    assert cc.get("test.key") == "value"
    assert cc.has("test.key")
    assert not cc.has("test.nonexistent")

def test_config_center_snapshot():
    cc = ConfigCenter()
    cc.reset()
    cc.set("test.snap", {"nested": True})
    snap = cc.snapshot()
    assert isinstance(snap, dict)
    assert snap["test"]["snap"]["nested"] is True
