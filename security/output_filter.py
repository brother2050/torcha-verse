"""Output filtering for the TorchaVerse security layer (Gate 3).

This module is the **third gate** of the defence-in-depth pipeline.  It
inspects model *outputs* (text, images, audio) for harmful content
before they are returned to the user or persisted.

Components
----------
* :class:`FilterResult` -- a dataclass capturing the verdict, a numeric
  score, the offending categories and a recommended action
  (``"pass"``, ``"block"`` or ``"flag"``).
* :class:`OutputFilter` -- the main filter.  It supports three media:

  - :meth:`filter_text` -- toxicity detection.  Uses the optional
    ``Detoxify`` package when available; otherwise falls back to a
    rule-based blocklist matcher.
  - :meth:`filter_image` -- NSFW detection.  Uses the optional
    ``NudeNet`` package when available; otherwise returns a permissive
    pass.
  - :meth:`filter_audio` -- audio content screening.  A pluggable hook
    that returns a permissive pass by default.

The module is **pure Python** (no ``torch`` dependency).  Optional
dependencies (``Detoxify``, ``NudeNet``) are imported lazily with
``try/except`` guards.  All public methods are thread-safe.

Example:
    >>> f = OutputFilter()
    >>> f.set_custom_blocklist(["badword"])
    >>> result = f.filter_text("this is a badword")
    >>> result.passed
    False
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

__all__ = [
    "OutputFilter",
    "FilterResult",
]

#: 模块级日志器，用于记录可选依赖缺失及过滤失败等警告信息。
_logger: logging.Logger = logging.getLogger("security.output_filter")

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from detoxify import Detoxify as _Detoxify  # type: ignore

    _HAS_DETOXIFY: bool = True
except Exception:  # pragma: no cover - Detoxify not installed
    _logger.debug("Detoxify 未安装，文本过滤将使用回退模式。", exc_info=True)
    _HAS_DETOXIFY: bool = False

try:  # pragma: no cover - import guard
    from nudenet import NudeDetector as _NudeDetector  # type: ignore

    _HAS_NUDENET: bool = True
except Exception:  # pragma: no cover - NudeNet not installed
    _logger.debug("NudeNet 未安装，图像过滤将使用回退模式。", exc_info=True)
    _HAS_NUDENET: bool = False


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Toxicity score at or above which text is blocked.
_DEFAULT_TOXICITY_THRESHOLD: float = 0.8

#: Toxicity score at or above which text is flagged (but not blocked).
_DEFAULT_TOXICITY_FLAG_THRESHOLD: float = 0.5

#: NSFW score at or above which an image is blocked.
_DEFAULT_NSFW_THRESHOLD: float = 0.8

#: NSFW score at or above which an image is flagged.
_DEFAULT_NSFW_FLAG_THRESHOLD: float = 0.5

#: Default toxicity score returned by the blocklist fallback when a
#: blocked word is found.
_BLOCKLIST_MATCH_SCORE: float = 1.0

#: Default toxicity score returned by the blocklist fallback when the
#: text is clean.
_BLOCKLIST_CLEAN_SCORE: float = 0.0

#: Toxicity categories recognised by the Detoxify fallback mapping.
_TOXICITY_CATEGORIES: tuple[str, ...] = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)

#: Default blocklist of profanity / harmful terms used when Detoxify is
#: unavailable.  Stored as lowercase substrings.
_DEFAULT_BLOCKLIST: tuple[str, ...] = (
    "hate",
    "kill",
    "bomb",
    "terrorist",
    "racist",
    "nazi",
)

#: NSFW label categories returned by NudeNet that are considered explicit.
_EXPLICIT_LABELS: tuple[str, ...] = (
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    """Outcome of an output filter check.

    Attributes:
        passed: ``True`` when the content is safe to release.
        score: Numeric severity in ``[0.0, 1.0]`` (higher = worse).
        categories: List of offending category names.
        action: Recommended action -- ``"pass"``, ``"block"`` or
            ``"flag"``.
    """

    passed: bool
    score: float
    categories: List[str] = field(default_factory=list)
    action: str = "pass"

    def __post_init__(self) -> None:
        if self.action not in ("pass", "block", "flag"):
            raise ValueError(
                f"action must be 'pass', 'block' or 'flag', got {self.action!r}."
            )


# ---------------------------------------------------------------------------
# OutputFilter
# ---------------------------------------------------------------------------
class OutputFilter:
    """Thread-safe output content filter (security Gate 3).

    Inspects text, image and audio outputs for harmful content.  When
    the optional ``Detoxify`` / ``NudeNet`` packages are installed they
    are used for ML-based detection; otherwise rule-based fallbacks are
    used so the filter always returns a verdict.

    Args:
        toxicity_threshold: Score at or above which text is blocked.
        toxicity_flag_threshold: Score at or above which text is flagged.
        nsfw_threshold: Score at or above which an image is blocked.
        nsfw_flag_threshold: Score at or above which an image is flagged.
        blocklist: Initial custom blocklist of words/phrases.

    Example:
        >>> f = OutputFilter()
        >>> f.filter_text("hello world").passed
        True
    """

    def __init__(
        self,
        toxicity_threshold: float = _DEFAULT_TOXICITY_THRESHOLD,
        toxicity_flag_threshold: float = _DEFAULT_TOXICITY_FLAG_THRESHOLD,
        nsfw_threshold: float = _DEFAULT_NSFW_THRESHOLD,
        nsfw_flag_threshold: float = _DEFAULT_NSFW_FLAG_THRESHOLD,
        blocklist: Optional[Sequence[str]] = None,
    ) -> None:
        if not 0.0 <= toxicity_flag_threshold <= toxicity_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= toxicity_flag_threshold <= toxicity_threshold <= 1."
            )
        if not 0.0 <= nsfw_flag_threshold <= nsfw_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= nsfw_flag_threshold <= nsfw_threshold <= 1."
            )

        self._tox_threshold: float = float(toxicity_threshold)
        self._tox_flag_threshold: float = float(toxicity_flag_threshold)
        self._nsfw_threshold: float = float(nsfw_threshold)
        self._nsfw_flag_threshold: float = float(nsfw_flag_threshold)
        self._lock: threading.Lock = threading.Lock()
        self._blocklist: list[str] = list(blocklist) if blocklist else list(_DEFAULT_BLOCKLIST)
        self._detoxify: Any = None
        self._nude_detector: Any = None
        self._audio_analyser: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def has_detoxify(self) -> bool:
        """``True`` when the Detoxify backend is available."""
        return _HAS_DETOXIFY

    @property
    def has_nudenet(self) -> bool:
        """``True`` when the NudeNet backend is available."""
        return _HAS_NUDENET

    # ------------------------------------------------------------------
    # Text filtering
    # ------------------------------------------------------------------
    def filter_text(self, text: str) -> FilterResult:
        """Screen ``text`` for toxicity.

        When Detoxify is installed the maximum score across its
        categories is used; otherwise a blocklist substring match is
        performed.

        Args:
            text: The text to inspect.

        Returns:
            A :class:`FilterResult`.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")

        if _HAS_DETOXIFY:
            return self._filter_text_detoxify(text)
        return self._filter_text_blocklist(text)

    # ------------------------------------------------------------------
    # Image filtering
    # ------------------------------------------------------------------
    def filter_image(self, image: Union[bytes, str, Any]) -> FilterResult:
        """Screen ``image`` for NSFW content.

        When NudeNet is installed the maximum detection score among
        explicit labels is used; otherwise a **fail-closed** verdict is
        returned (``passed=False, action="flag"``) so that un-screened
        images are never silently released.

        Args:
            image: A path, raw bytes or PIL image.

        Returns:
            A :class:`FilterResult`.
        """
        if _HAS_NUDENET:
            return self._filter_image_nudenet(image)
        # Fail-closed: without a backend we cannot guarantee the image is
        # safe, so we flag it rather than passing it through.
        return FilterResult(
            passed=False,
            score=0.0,
            categories=[],
            action="flag",
        )

    # ------------------------------------------------------------------
    # Audio filtering
    # ------------------------------------------------------------------
    def filter_audio(self, audio: Union[bytes, str, Any]) -> FilterResult:
        """Screen ``audio`` content.

        A pluggable hook that returns a permissive pass by default.
        Subclasses or callers can override the behaviour by providing a
        custom analyser via :meth:`set_audio_analyser`.

        Args:
            audio: A path, raw bytes or waveform tensor.

        Returns:
            A :class:`FilterResult`.
        """
        analyser = self._audio_analyser
        if analyser is not None:
            return analyser(audio)
        return FilterResult(
            passed=True,
            score=0.0,
            categories=[],
            action="pass",
        )

    # ------------------------------------------------------------------
    # Blocklist management
    # ------------------------------------------------------------------
    def set_custom_blocklist(self, words: Sequence[str]) -> None:
        """Replace the entire blocklist.

        Args:
            words: New list of words/phrases (case-insensitive).
        """
        with self._lock:
            self._blocklist = [str(w).lower() for w in words if w]

    def add_to_blocklist(self, word: str) -> None:
        """Append a single word to the blocklist."""
        with self._lock:
            self._blocklist.append(str(word).lower())

    def set_audio_analyser(self, analyser: Any) -> None:
        """Install a custom callable for :meth:`filter_audio`.

        Args:
            analyser: A callable ``audio -> FilterResult``.
        """
        with self._lock:
            self._audio_analyser = analyser

    # ------------------------------------------------------------------
    # Internals -- text
    # ------------------------------------------------------------------
    def _filter_text_detoxify(self, text: str) -> FilterResult:
        """Use Detoxify to score the text."""
        model = self._get_detoxify()
        try:
            results = model.predict(text)
        except Exception as exc:
            _logger.warning("Detoxify 预测失败，回退到黑名单匹配: %s", exc)
            return self._filter_text_blocklist(text)

        scores: dict[str, float] = {}
        for key, value in results.items():
            cat = key.lower()
            if isinstance(value, (list, tuple)) and value:
                scores[cat] = float(value[0])
            elif isinstance(value, (int, float)):
                scores[cat] = float(value)

        max_score = max(scores.values()) if scores else 0.0
        flagged = [cat for cat, sc in scores.items() if sc >= self._tox_flag_threshold]
        return self._build_text_result(max_score, flagged)

    def _filter_text_blocklist(self, text: str) -> FilterResult:
        """Rule-based fallback: substring match against the blocklist."""
        with self._lock:
            blocklist = list(self._blocklist)
        lowered = text.lower()
        matched = [w for w in blocklist if w and w in lowered]
        if matched:
            return self._build_text_result(_BLOCKLIST_MATCH_SCORE, matched)
        return FilterResult(
            passed=True,
            score=_BLOCKLIST_CLEAN_SCORE,
            categories=[],
            action="pass",
        )

    def _build_text_result(self, score: float, categories: List[str]) -> FilterResult:
        """Translate a numeric score into a :class:`FilterResult`."""
        if score >= self._tox_threshold:
            return FilterResult(
                passed=False,
                score=score,
                categories=categories,
                action="block",
            )
        if score >= self._tox_flag_threshold:
            return FilterResult(
                passed=True,
                score=score,
                categories=categories,
                action="flag",
            )
        return FilterResult(
            passed=True,
            score=score,
            categories=[],
            action="pass",
        )

    # ------------------------------------------------------------------
    # Internals -- image
    # ------------------------------------------------------------------
    def _filter_image_nudenet(self, image: Union[bytes, str, Any]) -> FilterResult:
        """Use NudeNet to score the image."""
        detector = self._get_nude_detector()
        try:
            detections = detector.detect(image)
        except Exception as exc:
            _logger.warning("NudeNet detection failed: %s; blocking image as fail-closed", exc)
            return FilterResult(
                passed=False,
                score=1.0,
                categories=[],
                action="block",
            )

        max_score = 0.0
        flagged: list[str] = []
        for det in detections or []:
            label = det.get("label", "")
            score = float(det.get("score", 0.0))
            if label in _EXPLICIT_LABELS and score > max_score:
                max_score = score
            if label in _EXPLICIT_LABELS:
                flagged.append(label)

        return self._build_image_result(max_score, flagged)

    def _build_image_result(self, score: float, categories: List[str]) -> FilterResult:
        """Translate an NSFW score into a :class:`FilterResult`."""
        if score >= self._nsfw_threshold:
            return FilterResult(
                passed=False,
                score=score,
                categories=categories,
                action="block",
            )
        if score >= self._nsfw_flag_threshold:
            return FilterResult(
                passed=True,
                score=score,
                categories=categories,
                action="flag",
            )
        return FilterResult(
            passed=True,
            score=score,
            categories=[],
            action="pass",
        )

    # ------------------------------------------------------------------
    # Lazy model singletons
    # ------------------------------------------------------------------
    def _get_detoxify(self) -> Any:
        """Lazily instantiate the Detoxify model (thread-safe)."""
        if self._detoxify is None:
            with self._lock:
                if self._detoxify is None:
                    self._detoxify = _Detoxify("original")  # type: ignore[name-defined]
        return self._detoxify

    def _get_nude_detector(self) -> Any:
        """Lazily instantiate the NudeNet detector (thread-safe)."""
        if self._nude_detector is None:
            with self._lock:
                if self._nude_detector is None:
                    self._nude_detector = _NudeDetector()  # type: ignore[name-defined]
        return self._nude_detector

    def __repr__(self) -> str:
        return (
            f"OutputFilter(detoxify={_HAS_DETOXIFY}, nudenet={_HAS_NUDENET}, "
            f"tox_threshold={self._tox_threshold}, "
            f"nsfw_threshold={self._nsfw_threshold})"
        )
