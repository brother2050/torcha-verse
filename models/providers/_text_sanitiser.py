"""Garbled-output sanitisation helpers (v0.10.5).

The project micro-transformer is a randomly-initialised
``TransformerDecoder`` + :class:`ByteTokenizer`.  When the model
samples ids outside the byte vocabulary (``>= 256``), the
default ``errors="replace"`` UTF-8 decode produces a single
U+FFFD per *skipped* id, which can truncate the surrounding
multi-byte sequence (e.g. a 3-byte CJK character that
partially overlaps the gap becomes a single FFFD + the
remaining two bytes -- classic "half character" garble).

The two helpers in this module clean up that output:

* :func:`sanitise_generation` runs at the model boundary and
  strips ASCII control characters, collapses runaway FFFD
  runs, and trims trailing whitespace.  The result is a
  human-readable string the CLI can render without polluting
  the terminal.
* :func:`quality_metrics` computes simple text-quality stats
  (FFFD ratio, control-char ratio, printable ratio) that the
  service layer uses to attach a diagnostic note to the
  response without altering the raw model output.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: U+FFFD is the standard UTF-8 replacement character.  We treat it
#: as the canonical "this byte is not a real character" signal.
_REPLACEMENT_CHAR = "\ufffd"

#: A *run* of 8+ consecutive FFFDs is a strong signal that the model
#: has entered a "garbage" regime (e.g. tail of a sampling run that
#: collapsed onto a single high-id token).  Truncate at the start of
#: the run.
_MAX_FFFD_RUN = 8

#: ASCII control characters we *strip* (except newline + tab + CR
#: which are sometimes legitimate).  ``\x7f`` is DEL which most
#: terminals render as ``^?``.
_CONTROL_CHARS = "".join(
    chr(c) for c in range(0, 32) if c not in (0x09, 0x0A, 0x0D)
) + "\x7f"
_CONTROL_CHAR_TRANS = str.maketrans("", "", _CONTROL_CHARS)

#: FFFD run detector (8+ in a row).
_FFFD_RUN_RE = re.compile(re.escape(_REPLACEMENT_CHAR) + "{" + str(_MAX_FFFD_RUN) + r",}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def sanitise_generation(raw: str) -> str:
    """Return a human-readable rendering of ``raw``.

    The transformation is intentionally conservative -- it only
    touches characters that the user *would never* want to see
    in a CLI:

    1. Strip ASCII control characters except ``\\n``, ``\\t``,
       ``\\r``.  These leak in when the model samples ids in
       ``[0, 32)`` (e.g. ``\\x00`` is a real byte value in the
       ByteTokenizer vocab).
    2. Truncate at the first run of
       :data:`_MAX_FFFD_RUN` consecutive U+FFFD characters.  Long
       FFFD runs are a strong signal that the model has entered
       a garbage regime; keeping the prefix preserves whatever
       legible text was emitted before the run.
    3. Normalise ``\r\n`` -> ``\n`` and trim trailing
       whitespace.

    The function never raises.  An empty input maps to an
    empty string.  The function is idempotent: applying it
    twice yields the same result as applying it once.
    """
    if not raw:
        return ""
    # 1. Strip control chars.
    out = raw.translate(_CONTROL_CHAR_TRANS)
    # 2. Truncate at long FFFD runs.
    match = _FFFD_RUN_RE.search(out)
    if match is not None:
        out = out[: match.start()]
    # 3. Normalise line endings + trim.
    out = out.replace("\r\n", "\n").rstrip()
    return out


def quality_metrics(text: str) -> Dict[str, float]:
    """Return diagnostic text-quality stats for ``text``.

    Returns a dict with:

    * ``length`` -- number of code points
    * ``printable_ratio`` -- fraction of code points that are
      letters / digits / punctuation (not whitespace, not
      control, not FFFD)
    * ``fffd_ratio`` -- fraction of code points that are
      U+FFFD
    * ``control_ratio`` -- fraction of code points that are
      ASCII control characters (excluding the
      whitespace subset)

    The service layer uses ``fffd_ratio`` to decide whether
    to attach a ``quality_warning`` to the response without
    altering the raw model output -- callers that want the
    sanitised view should pipe the result through
    :func:`sanitise_generation` separately.
    """
    if not text:
        return {
            "length": 0.0,
            "printable_ratio": 0.0,
            "fffd_ratio": 0.0,
            "control_ratio": 0.0,
        }
    n = float(len(text))
    fffd = sum(1 for c in text if c == _REPLACEMENT_CHAR)
    control = sum(
        1
        for c in text
        if (ord(c) < 0x20 and c not in "\t\n\r")
        or ord(c) == 0x7F
    )
    printable = sum(
        1
        for c in text
        if c.isprintable() and c != _REPLACEMENT_CHAR
    )
    return {
        "length": n,
        "printable_ratio": printable / n,
        "fffd_ratio": fffd / n,
        "control_ratio": control / n,
    }


def garble_assessment(
    text: str,
    *,
    printable_threshold: float = 0.5,
    fffd_threshold: float = 0.2,
) -> Tuple[str, str]:
    """Return ``(level, reason)`` describing the garble level of ``text``.

    Levels:

    * ``"ok"`` -- ``printable_ratio`` >= ``printable_threshold``
      and ``fffd_ratio`` < ``fffd_threshold``.
    * ``"warn"`` -- at least one threshold is just outside its
      nominal range (printed text is mostly there but noisy).
    * ``"garbled"`` -- both thresholds are violated, or the
      text is dominated by control / FFFD characters.

    The service layer maps ``"garbled"`` -> ``quality_warning``
    in the response.
    """
    m = quality_metrics(text)
    pr, fr = m["printable_ratio"], m["fffd_ratio"]
    if pr >= printable_threshold and fr < fffd_threshold:
        return "ok", ""
    if pr < printable_threshold and fr >= fffd_threshold:
        return "garbled", (
            "model output is mostly non-printable bytes "
            "(printable_ratio={pr:.2f}, fffd_ratio={fr:.2f}); "
            "the project micro-transformer has random weights and "
            "will produce noise until a real checkpoint is loaded "
            "(see docs/local_transformers.md § Loading a real model)."
        ).format(pr=pr, fr=fr)
    if fr >= fffd_threshold:
        return "warn", (
            "model output contains fffd_ratio={fr:.2f} U+FFFD "
            "replacement characters; some bytes fell outside the "
            "byte vocabulary.  Consider loading a real checkpoint."
        ).format(fr=fr)
    return "warn", (
        "model output printable_ratio={pr:.2f} below nominal "
        "threshold {thr:.2f}."
    ).format(pr=pr, thr=printable_threshold)


__all__ = [
    "sanitise_generation",
    "quality_metrics",
    "garble_assessment",
]
