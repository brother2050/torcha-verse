"""Tests for the v0.4.2 evaluation leaderboard skeleton.

Covers:

* Construction / validation of :class:`LeaderboardEntry` and
  rejection of negative / empty fields.
* :meth:`LeaderboardEntry.from_report` wiring against a hand-built
  :class:`EvaluationReport`.
* :class:`Leaderboard` ranking for both lower-is-better and
  higher-is-better primary metrics, including custom-metric paths.
* JSON round-trip and on-disk save / load helpers.
* Markdown rendering producing a pipe-table with a header row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.leaderboard import (
    LEADERBOARD_FORMAT_VERSION,
    Leaderboard,
    LeaderboardEntry,
    load_leaderboard,
    save_leaderboard,
)
from evaluation.runner import EvaluationReport


def _entry(model_id: str, fid: float, recall: float = 0.5, n: int = 10, t: float = 1.0):
    return LeaderboardEntry(
        model_id=model_id,
        config_hash="hash-" + model_id,
        prompt_set="basic",
        n_prompts=n,
        metrics={"fid": fid, "prompt_recall": recall},
        throughput_prompts_per_sec=n / t,
        runtime_seconds=t,
        notes="",
    )


# ---------------------------------------------------------------------------
# LeaderboardEntry validation
# ---------------------------------------------------------------------------
def test_entry_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="model_id"):
        LeaderboardEntry(
            model_id="",
            config_hash="h",
            prompt_set="p",
            n_prompts=1,
            metrics={},
            throughput_prompts_per_sec=1.0,
            runtime_seconds=1.0,
        )
    with pytest.raises(ValueError, match="config_hash"):
        LeaderboardEntry(
            model_id="m",
            config_hash="",
            prompt_set="p",
            n_prompts=1,
            metrics={},
            throughput_prompts_per_sec=1.0,
            runtime_seconds=1.0,
        )
    with pytest.raises(ValueError, match="prompt_set"):
        LeaderboardEntry(
            model_id="m",
            config_hash="h",
            prompt_set="",
            n_prompts=1,
            metrics={},
            throughput_prompts_per_sec=1.0,
            runtime_seconds=1.0,
        )


def test_entry_rejects_negative_scalars() -> None:
    base = dict(
        model_id="m",
        config_hash="h",
        prompt_set="p",
        metrics={},
        throughput_prompts_per_sec=1.0,
        runtime_seconds=1.0,
    )
    with pytest.raises(ValueError, match="n_prompts"):
        LeaderboardEntry(**base, n_prompts=-1)
    with pytest.raises(ValueError, match="throughput_prompts_per_sec"):
        LeaderboardEntry(**{**base, "n_prompts": 1, "throughput_prompts_per_sec": -1.0})
    with pytest.raises(ValueError, match="runtime_seconds"):
        LeaderboardEntry(**{**base, "n_prompts": 1, "runtime_seconds": -1.0})


def test_from_report_extracts_metrics() -> None:
    report = EvaluationReport(
        fid=7.5,
        prompt_recall={"mean": 0.3, "std": 0.05, "scores": [0.3, 0.3]},
        n_real=2,
        n_generated=4,
    )
    entry = LeaderboardEntry.from_report(
        report,
        model_id="tiny",
        config_hash="abc",
        prompt_set="basic",
        runtime_seconds=2.0,
    )
    assert entry.metrics["fid"] == 7.5
    assert entry.metrics["prompt_recall"] == pytest.approx(0.3)
    assert entry.n_prompts == 4
    assert entry.throughput_prompts_per_sec == pytest.approx(2.0)
    assert entry.runtime_seconds == 2.0


def test_from_report_handles_zero_runtime() -> None:
    report = EvaluationReport(fid=0.0, prompt_recall=None, n_real=0, n_generated=0)
    entry = LeaderboardEntry.from_report(
        report,
        model_id="tiny",
        config_hash="abc",
        prompt_set="basic",
        runtime_seconds=0.0,
    )
    # Throughput is 0.0 when there is no wall-clock time, not NaN.
    assert entry.throughput_prompts_per_sec == 0.0


# ---------------------------------------------------------------------------
# Leaderboard ranking
# ---------------------------------------------------------------------------
def test_ranked_fid_lower_is_better() -> None:
    board = Leaderboard()
    board.extend(
        [
            _entry("best", fid=5.0),
            _entry("worst", fid=20.0),
            _entry("middle", fid=12.0),
        ]
    )
    ranked = board.ranked("fid")
    assert [e.model_id for e in ranked] == ["best", "middle", "worst"]


def test_ranked_recall_higher_is_better() -> None:
    board = Leaderboard()
    board.extend(
        [
            _entry("low", fid=10.0, recall=0.1),
            _entry("high", fid=10.0, recall=0.9),
        ]
    )
    ranked = board.ranked("prompt_recall")
    assert [e.model_id for e in ranked] == ["high", "low"]


def test_ranked_throughput_higher_is_better() -> None:
    board = Leaderboard()
    board.extend(
        [
            _entry("slow", fid=10.0, t=10.0),  # throughput 1.0
            _entry("fast", fid=10.0, t=1.0),   # throughput 10.0
        ]
    )
    ranked = board.ranked("throughput_prompts_per_sec")
    assert [e.model_id for e in ranked] == ["fast", "slow"]


def test_ranked_with_missing_metric_puts_entry_last() -> None:
    board = Leaderboard()
    entry = LeaderboardEntry(
        model_id="no-metric",
        config_hash="h",
        prompt_set="p",
        n_prompts=1,
        metrics={},
        throughput_prompts_per_sec=1.0,
        runtime_seconds=1.0,
    )
    board.extend([_entry("winner", fid=1.0), entry])
    ranked = board.ranked("fid")
    assert [e.model_id for e in ranked] == ["winner", "no-metric"]


def test_ranked_with_custom_metric_and_descending() -> None:
    board = Leaderboard()
    board.extend(
        [
            LeaderboardEntry(
                model_id="a",
                config_hash="h",
                prompt_set="p",
                n_prompts=1,
                metrics={"novelty": 0.4},
                throughput_prompts_per_sec=1.0,
                runtime_seconds=1.0,
            ),
            LeaderboardEntry(
                model_id="b",
                config_hash="h",
                prompt_set="p",
                n_prompts=1,
                metrics={"novelty": 0.7},
                throughput_prompts_per_sec=1.0,
                runtime_seconds=1.0,
            ),
        ]
    )
    ranked = board.ranked("novelty", descending=True)
    assert [e.model_id for e in ranked] == ["b", "a"]


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------
def test_to_dict_from_dict_round_trip() -> None:
    board = Leaderboard(name="v0.4.2 smoke", description="manual entry")
    board.add(_entry("x", fid=1.0))
    board.add(_entry("y", fid=2.0))
    payload = board.to_dict()
    assert payload["format_version"] == LEADERBOARD_FORMAT_VERSION
    assert payload["name"] == "v0.4.2 smoke"
    rebuilt = Leaderboard.from_dict(payload)
    assert len(rebuilt) == 2
    assert rebuilt.entries[0].model_id == "x"


def test_from_json_rejects_unknown_format_version() -> None:
    payload = {
        "format_version": LEADERBOARD_FORMAT_VERSION + 99,
        "name": "future",
        "description": "",
        "entries": [],
    }
    with pytest.raises(ValueError, match="Unsupported leaderboard format_version"):
        Leaderboard.from_dict(payload)


def test_to_json_and_back() -> None:
    board = Leaderboard()
    board.add(_entry("z", fid=3.0))
    text = board.to_json()
    # Round-trip via raw JSON to be sure the text is parseable.
    parsed = json.loads(text)
    assert parsed["entries"][0]["model_id"] == "z"
    rebuilt = Leaderboard.from_json(text)
    assert rebuilt.entries[0].model_id == "z"


def test_save_load_round_trip(tmp_path: Path) -> None:
    board = Leaderboard(name="disk-round-trip")
    board.add(_entry("persisted", fid=8.0))
    path = tmp_path / "leaderboard.json"
    save_leaderboard(board, path)
    loaded = load_leaderboard(path)
    assert len(loaded) == 1
    assert loaded.entries[0].model_id == "persisted"
    assert loaded.name == "disk-round-trip"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def test_to_markdown_contains_header_and_rows() -> None:
    board = Leaderboard()
    board.extend([_entry("a", fid=1.0), _entry("b", fid=2.0)])
    text = board.to_markdown()
    assert "| model | prompt_set | n | fid | throughput | runtime |" in text
    assert "a" in text and "b" in text
    # The Markdown table should be sorted by ``fid`` ascending.
    assert text.index("a") < text.index("b")


def test_to_markdown_empty_board() -> None:
    text = Leaderboard().to_markdown()
    assert "| model |" in text
    # No body rows.
    assert text.count("\n|") == 1
