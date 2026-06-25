"""Evaluation leaderboard (v1.0.0 M3b, shipped as a v0.4.2 skeleton).

A leaderboard tracks a list of :class:`LeaderboardEntry` records --
each one summarises the result of running :class:`EvaluationRunner`
against a (model, prompt-set, config) triple.  Entries can be:

* loaded from a JSON / JSONL file via :func:`load_leaderboard`,
* appended in memory via :meth:`Leaderboard.add`,
* sorted / ranked by any of the primary metrics (FID, prompt-recall,
  PSNR, SSIM, LPIPS, throughput),
* serialised back to JSON via :meth:`Leaderboard.to_dict` /
  :meth:`Leaderboard.to_json`,
* and rendered as a Markdown table for quick paste into PRs.

The format is deliberately simple so it can be consumed both by the
HTTP API and by ``pytest -m eval`` runs in CI; the goal is a working
skeleton for the v0.4.x → v1.0.0 bridge, not a full multi-tenant
benchmarking platform.

Example
-------

    >>> from evaluation.leaderboard import Leaderboard, LeaderboardEntry
    >>> board = Leaderboard()
    >>> board.add(LeaderboardEntry(
    ...     model_id="tiny-transformer",
    ...     config_hash="abc123",
    ...     prompt_set="basic",
    ...     n_prompts=10,
    ...     metrics={"fid": 12.3, "prompt_recall": 0.28},
    ...     throughput_prompts_per_sec=4.2,
    ...     runtime_seconds=2.4,
    ... ))
    >>> board.to_markdown()  # doctest: +ELLIPSIS
    '| model | ... |'
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .runner import EvaluationReport

__all__ = [
    "LeaderboardEntry",
    "Leaderboard",
    "load_leaderboard",
    "save_leaderboard",
    "LEADERBOARD_FORMAT_VERSION",
]

#: Bumped on breaking changes to the JSON schema; downstream tooling
#: should refuse to load a file whose ``format_version`` it does not
#: understand.  v0.4.2 ships 1; v1.0.0 may bump to 2 if it adds
#: per-model sub-leaderboards.
LEADERBOARD_FORMAT_VERSION: int = 1

#: Metrics used as the *primary* ranking criteria.  Lower-is-better
#: metrics (FID, LPIPS) are tagged with a negative sort sign at the
#: call site.
PRIMARY_METRICS: tuple[str, ...] = (
    "fid",
    "prompt_recall",
    "psnr",
    "ssim",
    "lpips",
    "throughput_prompts_per_sec",
)


@dataclass
class LeaderboardEntry:
    """A single leaderboard row.

    Attributes:
        model_id: Stable identifier of the model that produced the
            run (e.g. ``"qwen2.5-7b-instruct"``).
        config_hash: Hash of the config snapshot used for the run.
            Pairs with the on-disk config archive to make runs
            reproducible.
        prompt_set: Name of the prompt / dataset used
            (e.g. ``"mscoco-1k"``, ``"basic"``).
        n_prompts: Number of prompts scored.
        metrics: Per-metric results.  Recognised keys are listed in
            :data:`PRIMARY_METRICS`; any other key is preserved as-is
            in the JSON output.
        throughput_prompts_per_sec: Wall-clock throughput, in prompts
            per second.
        runtime_seconds: Total wall-clock time for the run.
        notes: Free-form annotation (e.g. "torch 2.1.0 + bf16").
        created_at: ISO-8601 timestamp of when the entry was
            constructed.  Defaults to "now (UTC)".
        git_commit: Optional commit hash of the codebase that
            produced the run.
    """

    model_id: str
    config_hash: str
    prompt_set: str
    n_prompts: int
    metrics: Dict[str, float]
    throughput_prompts_per_sec: float
    runtime_seconds: float
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    git_commit: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("LeaderboardEntry.model_id must be a non-empty string.")
        if not self.config_hash:
            raise ValueError("LeaderboardEntry.config_hash must be a non-empty string.")
        if not self.prompt_set:
            raise ValueError("LeaderboardEntry.prompt_set must be a non-empty string.")
        if self.n_prompts < 0:
            raise ValueError(f"n_prompts must be >= 0, got {self.n_prompts}.")
        if self.throughput_prompts_per_sec < 0:
            raise ValueError(
                f"throughput_prompts_per_sec must be >= 0, got "
                f"{self.throughput_prompts_per_sec}."
            )
        if self.runtime_seconds < 0:
            raise ValueError(
                f"runtime_seconds must be >= 0, got {self.runtime_seconds}."
            )

    @classmethod
    def from_report(
        cls,
        report: EvaluationReport,
        *,
        model_id: str,
        config_hash: str,
        prompt_set: str,
        runtime_seconds: float,
        notes: str = "",
        git_commit: Optional[str] = None,
    ) -> "LeaderboardEntry":
        """Build an entry from an :class:`EvaluationReport`.

        This is the canonical "I just ran an evaluation, append it to
        the board" path; it extracts ``n_generated``, ``fid`` and
        ``prompt_recall`` from the report and computes throughput.
        """
        prompt_recall = report.prompt_recall or {}
        prompt_recall_mean = float(prompt_recall.get("mean", 0.0))
        metrics: Dict[str, float] = {
            "fid": float(report.fid),
            "prompt_recall": prompt_recall_mean,
        }
        n_prompts_value = int(report.n_generated)
        throughput = (
            float(n_prompts_value) / runtime_seconds if runtime_seconds > 0 else 0.0
        )
        return cls(
            model_id=model_id,
            config_hash=config_hash,
            prompt_set=prompt_set,
            n_prompts=n_prompts_value,
            metrics=metrics,
            throughput_prompts_per_sec=throughput,
            runtime_seconds=float(runtime_seconds),
            notes=notes,
            git_commit=git_commit,
        )

    def primary_metric(self, name: str) -> float:
        """Return the value of the primary metric ``name`` or 0.0."""
        return float(self.metrics.get(name, 0.0))


@dataclass
class Leaderboard:
    """An ordered collection of :class:`LeaderboardEntry`.

    Entries are stored in insertion order; :meth:`ranked` returns a
    new list sorted by a chosen primary metric.
    """

    entries: List[LeaderboardEntry] = field(default_factory=list)
    name: str = "torcha-verse-leaderboard"
    description: str = ""

    def add(self, entry: LeaderboardEntry) -> None:
        """Append ``entry`` to the leaderboard."""
        self.entries.append(entry)

    def extend(self, entries: Iterable[LeaderboardEntry]) -> None:
        """Append every entry from ``entries`` in order."""
        for entry in entries:
            self.add(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary view of the board."""
        return {
            "format_version": LEADERBOARD_FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "entries": [asdict(e) for e in self.entries],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Return the JSON text representation."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Leaderboard":
        """Build a :class:`Leaderboard` from a parsed JSON dict."""
        version = int(payload.get("format_version", LEADERBOARD_FORMAT_VERSION))
        if version != LEADERBOARD_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported leaderboard format_version {version}; this build "
                f"understands version {LEADERBOARD_FORMAT_VERSION}."
            )
        entries_raw = payload.get("entries", [])
        entries = [LeaderboardEntry(**e) for e in entries_raw]
        return cls(
            entries=entries,
            name=str(payload.get("name", "torcha-verse-leaderboard")),
            description=str(payload.get("description", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "Leaderboard":
        """Parse a :class:`Leaderboard` from JSON text."""
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def ranked(
        self,
        metric: str = "fid",
        *,
        descending: bool = False,
    ) -> List[LeaderboardEntry]:
        """Return a new list of entries sorted by ``metric``.

        Args:
            metric: One of :data:`PRIMARY_METRICS` or any custom key
                present in :attr:`LeaderboardEntry.metrics`.  The
                default ``"fid"`` is *lower-is-better* in the
                literature so ``descending=False`` keeps the natural
                ordering.
            descending: Sort in descending order (use for
                higher-is-better metrics like ``prompt_recall`` /
                ``ssim`` / ``throughput``).

        Returns:
            A new list of entries, sorted by the chosen metric.
            Entries missing the metric are placed last regardless of
            ``descending`` to avoid giving them a phantom best rank.
        """
        if metric not in PRIMARY_METRICS and metric not in (
            "throughput_prompts_per_sec", "runtime_seconds"
        ):
            # Custom metrics: trust the caller's intent on direction.
            sentinel = (float("-inf") if descending else float("inf"))

            def key(entry: LeaderboardEntry):
                if metric in entry.metrics:
                    return entry.metrics[metric]
                if metric == "throughput_prompts_per_sec":
                    return entry.throughput_prompts_per_sec
                if metric == "runtime_seconds":
                    return entry.runtime_seconds
                return sentinel

            return sorted(
                self.entries, key=key, reverse=descending
            )

        # Primary-metric path: pull from ``metrics`` (or alias).
        def key(entry: LeaderboardEntry):
            if metric == "throughput_prompts_per_sec":
                return entry.throughput_prompts_per_sec
            return entry.metrics.get(metric, float("inf") if not descending else float("-inf"))

        # For higher-is-better primary metrics (ssim, psnr, prompt_recall,
        # throughput) invert the sign for stable tie-breaking.
        higher_is_better = metric in (
            "prompt_recall",
            "psnr",
            "ssim",
            "throughput_prompts_per_sec",
        )
        if higher_is_better:
            descending = True
        return sorted(self.entries, key=key, reverse=descending)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def to_markdown(self, metric: str = "fid") -> str:
        """Render the leaderboard as a Markdown table.

        Columns: ``model``, ``prompt_set``, ``n_prompts``, the chosen
        ``metric``, ``throughput`` (prompts/s), and ``runtime``.
        Sorted by the chosen metric using the natural direction
        (lower-is-better for ``fid`` / ``lpips``).
        """
        ranked = self.ranked(metric)
        header = (
            f"| model | prompt_set | n | {metric} | throughput | runtime |\n"
            "|---|---|---:|---:|---:|---:|"
        )
        rows: List[str] = []
        for entry in ranked:
            value = entry.primary_metric(metric)
            rows.append(
                "| "
                + " | ".join(
                    [
                        entry.model_id,
                        entry.prompt_set,
                        str(entry.n_prompts),
                        f"{value:.4f}",
                        f"{entry.throughput_prompts_per_sec:.2f}",
                        f"{entry.runtime_seconds:.2f}s",
                    ]
                )
                + " |"
            )
        return "\n".join([header, *rows]) + ("\n" if rows else "")


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------
def load_leaderboard(path: Union[str, Path]) -> Leaderboard:
    """Load a :class:`Leaderboard` from ``path`` (JSON)."""
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return Leaderboard.from_json(text)


def save_leaderboard(
    board: Leaderboard, path: Union[str, Path], *, indent: Optional[int] = 2
) -> None:
    """Serialise ``board`` to ``path`` as JSON."""
    Path(path).expanduser().write_text(
        board.to_json(indent=indent), encoding="utf-8"
    )
