"""Smoke tests for :mod:`training.synthetic_data`.

Validates the public API of :class:`SyntheticDataGenerator`.  The
generator depends on a ``TextEngine`` callable; we inject a deterministic
stub so tests do not need any real model.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from training.synthetic_data import (
    SyntheticDataConfig,
    SyntheticDataGenerator,
)


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------
class _StubTextEngine:
    """Deterministic text engine for tests."""

    def __init__(self, reply: str = "stub-response") -> None:
        self._reply = reply
        self.calls: List[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self._reply


# ---------------------------------------------------------------------------
# SyntheticDataConfig
# ---------------------------------------------------------------------------
class TestSyntheticDataConfig:
    def test_defaults(self) -> None:
        cfg = SyntheticDataConfig()
        # The config exposes generation parameters; assert the fields
        # exist and are sane.
        assert cfg.max_tokens >= 16
        assert 0.0 <= cfg.temperature <= 2.0
        assert 0.0 <= cfg.top_p <= 1.0
        assert isinstance(cfg.seed, int)


# ---------------------------------------------------------------------------
# SyntheticDataGenerator.filter_quality
# ---------------------------------------------------------------------------
def _make() -> SyntheticDataGenerator:
    return SyntheticDataGenerator(text_engine=_StubTextEngine())


class TestFilterQuality:
    def test_keeps_high_quality_samples(self) -> None:
        gen = _make()
        data: List[Dict[str, Any]] = [
            {"instruction": "Q1", "response": "the quick brown fox jumps over the lazy dog " * 3},
            {"instruction": "Q2", "response": "lorem ipsum dolor sit amet consectetur adipiscing " * 3},
        ]
        kept = gen.filter_quality(data, min_quality=0.3)
        assert len(kept) == 2

    def test_drops_short_samples(self) -> None:
        """Samples whose only text is too short should be filtered out.

        The quality heuristic rewards length: a single-word response gets
        a low score.  We set the threshold high enough that short
        samples fall through.
        """
        gen = _make()
        data: List[Dict[str, Any]] = [
            # 1 word -- very short -- quality score is 0.0 (returns
            # early because ``_longest_text`` finds only single-word
            # strings).
            {"instruction": "Q1", "response": "yes"},
            # ~30 unique words -- passes the 0.7 threshold.
            {
                "instruction": "Q2",
                "response": "the answer to your question involves many "
                            "considerations including safety efficiency "
                            "scalability reliability and maintainability",
            },
        ]
        kept = gen.filter_quality(data, min_quality=0.7)
        # Only the long sample survives.
        assert len(kept) == 1
        assert kept[0]["instruction"] == "Q2"

    def test_respects_existing_quality_score(self) -> None:
        gen = _make()
        data: List[Dict[str, Any]] = [
            {"instruction": "Q1", "response": "x", "quality": 0.9},
            {"instruction": "Q2", "response": "x", "quality": 0.1},
        ]
        kept = gen.filter_quality(data, min_quality=0.5)
        assert len(kept) == 1
        assert kept[0]["instruction"] == "Q1"

    def test_returns_empty_list_for_all_bad(self) -> None:
        gen = _make()
        data: List[Dict[str, Any]] = [
            {"instruction": "Q1", "response": ""},
            {"instruction": "Q2", "response": ""},
        ]
        kept = gen.filter_quality(data, min_quality=0.99)
        assert kept == []


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------
class TestJSONLRoundTrip:
    def test_save_and_load(self, tmp_path) -> None:
        path = tmp_path / "synthetic.jsonl"
        data: List[Dict[str, Any]] = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        SyntheticDataGenerator.save_to_jsonl(data, str(path))
        loaded = SyntheticDataGenerator.load_from_jsonl(str(path))
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "synthetic.jsonl"
        data: List[Dict[str, Any]] = [{"x": 1}]
        SyntheticDataGenerator.save_to_jsonl(data, str(path))
        assert path.exists()
        loaded = SyntheticDataGenerator.load_from_jsonl(str(path))
        assert loaded == data
