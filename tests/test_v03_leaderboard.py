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


# ---------------------------------------------------------------------------
# HTML rendering (v0.4.3)
# ---------------------------------------------------------------------------
def test_to_html_includes_doctype_and_table() -> None:
    board = Leaderboard(name="v0.4.3 html", description="html smoke")
    board.add(_entry("a", fid=1.0))
    text = board.to_html(title="custom title")
    assert text.startswith("<!DOCTYPE html>")
    assert "<title>custom title</title>" in text
    assert "<h1>custom title</h1>" in text
    assert "<table>" in text
    assert "<th>model</th>" in text
    assert ">a<" in text


def test_to_html_escapes_user_supplied_data() -> None:
    board = Leaderboard()
    # model_id contains a payload that would inject script tags if
    # not escaped; ``_html_escape`` should neutralise them.
    board.add(
        LeaderboardEntry(
            model_id='<script>alert("xss")</script>',
            config_hash="h",
            prompt_set="p&q",
            n_prompts=1,
            metrics={"fid": 1.0},
            throughput_prompts_per_sec=1.0,
            runtime_seconds=1.0,
        )
    )
    text = board.to_html()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    # Double quotes are HTML-escaped too, so the original payload
    # is neutralised in the rendered output.
    assert "&quot;xss&quot;" in text
    assert "p&amp;q" in text


def test_to_html_uses_self_name_when_no_title() -> None:
    board = Leaderboard(name="my-board")
    text = board.to_html()
    assert "<title>my-board</title>" in text


def test_to_html_empty_board_still_renders() -> None:
    text = Leaderboard().to_html()
    assert text.startswith("<!DOCTYPE html>")
    # No data rows.
    assert text.count("<tr>") == 1  # header row only


# ---------------------------------------------------------------------------
# Compare (v0.4.3)
# ---------------------------------------------------------------------------
def _board(*entries: LeaderboardEntry) -> Leaderboard:
    board = Leaderboard()
    board.extend(entries)
    return board


def test_compare_finds_common_deltas() -> None:
    a = _entry("model-a", fid=10.0, recall=0.5, n=10, t=1.0)
    b = _entry("model-b", fid=20.0, recall=0.4, n=10, t=2.0)
    a_again = LeaderboardEntry(
        model_id=a.model_id,
        config_hash=a.config_hash,
        prompt_set=a.prompt_set,
        n_prompts=a.n_prompts,
        metrics={"fid": 8.0, "prompt_recall": 0.6},
        throughput_prompts_per_sec=20.0,
        runtime_seconds=0.5,
    )
    board_old = _board(a, b)
    board_new = _board(a_again, b)
    result = board_new.compare(board_old, metric="fid")
    # model-a appears in both: delta = 8 - 10 = -2 (improvement).
    common_a = [c for c in result["common"] if c["key"][0] == "model-a"][0]
    assert common_a["delta"] == pytest.approx(-2.0)
    assert common_a["delta_pct"] == pytest.approx(-20.0)
    # Throughput delta = 20 - 10 = +10.
    assert common_a["throughput_delta"] == pytest.approx(10.0)
    # Sorted ascending: most-improved first.
    assert result["common"][0]["key"][0] == "model-a"
    # Nothing appears in only-one list.
    assert result["only_in_self"] == []
    assert result["only_in_other"] == []


def test_compare_handles_only_in_self_and_other() -> None:
    old = _board(_entry("x", fid=1.0))
    new = _board(_entry("x", fid=1.0), _entry("y", fid=2.0))
    result = new.compare(old, metric="fid")
    assert [e.model_id for e in result["only_in_self"]] == ["y"]
    assert result["only_in_other"] == []


def test_compare_handles_only_in_other() -> None:
    old = _board(_entry("x", fid=1.0), _entry("z", fid=3.0))
    new = _board(_entry("x", fid=1.0))
    result = new.compare(old, metric="fid")
    assert result["only_in_self"] == []
    assert [e.model_id for e in result["only_in_other"]] == ["z"]


def test_compare_delta_pct_handles_zero_baseline() -> None:
    # When the baseline metric is 0.0, ``delta_pct`` is +inf rather
    # than ZeroDivisionError.
    old = _board(_entry("m", fid=0.0))
    new = _board(_entry("m", fid=5.0))
    result = new.compare(old, metric="fid")
    assert result["common"][0]["delta_pct"] == float("inf")
