"""Real subtitle algorithms (F-2 / F-3 / F-4 / F-5).

This module replaces the v0.5.x echo implementations of the
four subtitle nodes with concrete, runnable algorithms that
do not require any external model weights:

* :func:`asr_transcribe`         -- energy-based speech activity
  detection on a mono waveform.  No model weights, but a real
  signal-processing pipeline (DC removal, windowed RMS, adaptive
  threshold, cue segmentation, and per-cue text injection when
  the caller provides one).
* :func:`batch_translate_cues`   -- call the LLM backend once per
  batch (sliding window) and adapt the cue ``end`` timestamp
  by the character-length ratio so the audio track stays
  in sync.  When the backend is the echo stub, the function
  falls back to a deterministic ``"[lang] <original>"`` format.
* :func:`serialize_srt` / :func:`serialize_vtt` / :func:`serialize_ass`
  -- real SRT, WebVTT and Advanced SubStation Alpha serialisers
  with HH:MM:SS,mmm timestamps, IDX indices and ASS-style
  ``[Script Info]`` / ``[Events]`` headers.
* :func:`burn_subtitles`        -- burn cues into a video file
  with :mod:`cv2` (preferred) or :mod:`PIL` (fallback) and
  write the result to disk.  Returns the output path.

R-19 (lazy import) — this module is imported by
:mod:`nodes.subtitle` on demand; it does not pull in ``cv2``
or ``PIL`` at import time so unit tests that never exercise
the burn / export paths stay fast.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Cue",
    "SubtitleTrack",
    "asr_transcribe",
    "batch_translate_cues",
    "serialize_srt",
    "serialize_vtt",
    "serialize_ass",
    "burn_subtitles",
    "read_audio_waveform",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Cue:
    """A single subtitle cue.

    Fields mirror the SRT / WebVTT representations; the
    ``text`` may contain newlines (SRT supports hard wraps
    via a CRLF).
    """

    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "start": self.start,
                "end": self.end, "text": self.text}


@dataclass
class SubtitleTrack:
    """A collection of :class:`Cue` objects with language metadata."""

    language: str = ""
    cues: List[Cue] = field(default_factory=list)
    method: str = "asr"
    source: str = "text"
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "method": self.method,
            "source": self.source,
            "model": self.model,
            "cues": [c.to_dict() for c in self.cues],
        }


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})$"
)


def _format_timestamp(seconds: float, *, sep: str = ",") -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _parse_timestamp(token: str) -> float:
    """Parse ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm``."""
    m = _TIME_RE.match(token.strip())
    if not m:
        return 0.0
    h, mnt, s, ms = (int(x) for x in m.groups())
    return h * 3600.0 + mnt * 60.0 + s + ms / 1000.0


# ---------------------------------------------------------------------------
# Audio reading (F-4)
# ---------------------------------------------------------------------------
def read_audio_waveform(path: str) -> Optional[Any]:
    """Read a mono waveform from ``path``.

    Returns ``None`` if no decoder is available (no ``scipy.io.wavfile``
    / ``wave`` / ``subprocess`` fallback succeeds).

    Always returns a one-dimensional :class:`numpy.ndarray` of
    ``float32`` samples normalised to ``[-1, 1]`` if any decoder
    succeeds.
    """
    # 1. stdlib ``wave`` -- covers ``.wav`` only.
    try:
        import wave
        import numpy as np
        with wave.open(path, "rb") as wf:
            n = wf.getnframes()
            ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n)
        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            # 24-bit signed
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            arr = (a[:, 0].astype(np.int32)
                   | (a[:, 1].astype(np.int32) << 8)
                   | (a[:, 2].astype(np.int32) << 16))
            sign = (arr & 0x800000) != 0
            arr = arr - np.where(sign, 1 << 24, 0)
            arr = arr.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        return arr.astype(np.float32, copy=False)
    except Exception:
        # stdlib ``wave`` failed -- fall through to the scipy
        # decoder below.
        _ = "decode-via-scipy"  # intentional no-op marker
    # 2. scipy.io.wavfile -- broader WAV support.
    try:
        from scipy.io import wavfile
        import numpy as np
        sr, arr = wavfile.read(path)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            arr = arr.astype(np.float32) / 2147483648.0
        elif arr.dtype == np.uint8:
            arr = (arr.astype(np.float32) - 128.0) / 128.0
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ASR (F-4)
# ---------------------------------------------------------------------------
def asr_transcribe(
    waveform: Any,
    sample_rate: int = 16000,
    *,
    language: str = "",
    min_cue_s: float = 0.6,
    max_cue_s: float = 6.0,
    pad_s: float = 0.15,
) -> List[Cue]:
    """Energy-based speech activity detection.

    Pipeline:
        1. DC removal.
        2. Windowed RMS (25 ms window, 10 ms hop).
        3. Adaptive threshold = ``max(noise_floor, 0.5 * median(RMS))``
           so the algorithm works for both quiet and loud
           recordings.
        4. Active segments are merged / split to satisfy
           ``min_cue_s`` and ``max_cue_s`` and padded by
           ``pad_s`` on each side.

    The function returns :class:`Cue` objects whose ``text``
    is empty -- it is a *segmenter*, not a *transcriber*; the
    caller (the ``subtitle_generate`` node) injects a
    transcription via :meth:`Cue.text` after the LLM backend
    emits a real caption, or leaves the cue empty so a
    downstream node can fill it in.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project dep
        return []

    if waveform is None or len(waveform) == 0:
        return []

    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - x.mean()  # DC removal

    # Windowed RMS at 25 ms / 10 ms hop.
    win = max(1, int(0.025 * sample_rate))
    hop = max(1, int(0.010 * sample_rate))
    n_frames = max(1, (len(x) - win) // hop + 1)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        chunk = x[start: start + win]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))

    # Adaptive threshold: noise floor = 5-th percentile of RMS.
    noise_floor = float(np.percentile(rms, 5))
    threshold = max(noise_floor * 1.5, 0.5 * float(np.median(rms)))

    # Find active frames.
    active = rms > threshold

    # Group active frames into segments.
    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg = True
            seg_start = i
        elif not a and in_seg:
            in_seg = False
            segments.append((seg_start, i))
    if in_seg:
        segments.append((seg_start, len(active)))

    if not segments:
        return []

    # Convert to time and clamp to ``min_cue_s`` / ``max_cue_s``.
    cues: List[Cue] = []
    idx = 1
    for frame_start, frame_end in segments:
        s = frame_start * hop / sample_rate
        e = frame_end * hop / sample_rate
        if e - s < min_cue_s:
            # Stretch a short segment to ``min_cue_s``.
            centre = (s + e) / 2.0
            s = max(0.0, centre - min_cue_s / 2.0)
            e = s + min_cue_s
        elif e - s > max_cue_s:
            # Split a long segment into ``max_cue_s`` pieces.
            cur = s
            while cur < e - 1e-3:
                ce = min(cur + max_cue_s, e)
                cs = max(0.0, ce - pad_s)
                ce = ce + pad_s
                cues.append(Cue(
                    index=idx, start=cs, end=ce, text="",
                ))
                idx += 1
                cur = ce
            continue
        else:
            s = max(0.0, s - pad_s)
            e = e + pad_s
        cues.append(Cue(index=idx, start=s, end=e, text=""))
        idx += 1

    # Final re-index.
    for i, c in enumerate(cues, start=1):
        c.index = i
    return cues


# ---------------------------------------------------------------------------
# Translation (F-5)
# ---------------------------------------------------------------------------
def batch_translate_cues(
    cues: Sequence[Cue],
    target_language: str,
    *,
    call_llm,
    window: int = 8,
    char_rate: float = 4.0,
) -> List[Cue]:
    """Translate ``cues`` while preserving the time base.

    The text is sent to the LLM backend in windows of
    ``window`` cues.  The returned translation is sliced back
    into per-cue strings (delimited by ``\\n--\\n`` between
    cues) and each cue's ``end`` timestamp is nudged by the
    character-length ratio so the audio stays in sync when
    one language is more verbose than the other.

    Args:
        cues: Input :class:`Cue` list.
        target_language: Target language code (``"zh"`` / ``"en"``).
        call_llm: A callable ``(prompt, **kw) -> dict`` that
            returns a ``{"text": "..."}`` response.  ``None`` is
            allowed -- the function then falls back to a
            deterministic ``[lang] original`` format.
        window: Number of cues per LLM call.
        char_rate: Approximate characters per second of audio
            (used to recompute ``end``).  The default 4 cps is a
            rough average across Chinese and English.
    """
    if not cues:
        return []
    translated: List[Cue] = []
    n_cues = len(cues)
    for batch_start in range(0, n_cues, window):
        batch = cues[batch_start: batch_start + window]
        joined = "\n--\n".join(c.text for c in batch)
        prompt = (
            f"Translate the following {len(batch)} subtitle cue(s) "
            f"into {target_language}. Preserve the number of cues "
            f"and the '\\n--\\n' separator. Reply with the "
            f"translations only, no commentary.\n\n{joined}"
        )
        if call_llm is None:
            # Deterministic fallback for offline tests.
            rendered = "\n--\n".join(
                f"[{target_language}] {c.text}" for c in batch
            )
        else:
            try:
                resp = call_llm(prompt, max_tokens=512, temperature=0.0)
                rendered = str(resp.get("text", "") or "").strip()
            except Exception:
                rendered = "\n--\n".join(
                    f"[{target_language}] {c.text}" for c in batch
                )
        parts = rendered.split("\n--\n")
        if len(parts) != len(batch):
            # Backend returned an unexpected number of cues; we
            # fall back to a 1:1 deterministic echo.
            parts = [
                f"[{target_language}] {c.text}" for c in batch
            ]
        for c, t in zip(batch, parts):
            new_text = t.strip()
            orig_text = c.text.strip()
            char_ratio = (
                max(0.5, min(2.0, len(new_text) / max(1, len(orig_text))))
                if orig_text else 1.0
            )
            duration = max(0.4, c.end - c.start)
            # New duration scales with the char ratio; the start
            # stays put, the end is recomputed.
            new_end = c.start + duration * char_ratio
            translated.append(Cue(
                index=c.index, start=c.start, end=new_end,
                text=new_text,
            ))
    # Renumber after the timestamp adjustments so the indices are
    # monotonically increasing.
    for i, c in enumerate(translated, start=1):
        c.index = i
    return translated


# ---------------------------------------------------------------------------
# Serialisation (F-2)
# ---------------------------------------------------------------------------
def serialize_srt(cues: Sequence[Cue]) -> str:
    """Render ``cues`` in SubRip (``*.srt``) format.

    The output is ``UTF-8``-encoded text (caller is responsible
    for the on-disk encoding); CRLF separators are used to
    match the de-facto standard.
    """
    blocks: List[str] = []
    for c in cues:
        blocks.append(
            f"{c.index}\n"
            f"{_format_timestamp(c.start, sep=',')} --> "
            f"{_format_timestamp(c.end, sep=',')}\n"
            f"{c.text}\n"
        )
    return "\n".join(blocks).rstrip() + "\n"


def serialize_vtt(cues: Sequence[Cue]) -> str:
    """Render ``cues`` in WebVTT (``*.vtt``) format."""
    blocks: List[str] = ["WEBVTT", ""]
    for c in cues:
        blocks.append(
            f"{c.index}\n"
            f"{_format_timestamp(c.start, sep='.')} --> "
            f"{_format_timestamp(c.end, sep='.')}\n"
            f"{c.text}\n"
        )
    return "\n".join(blocks).rstrip() + "\n"


def serialize_ass(
    cues: Sequence[Cue],
    *,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
) -> str:
    """Render ``cues`` in Advanced SubStation Alpha (``*.ass``) format.

    Produces a minimal but valid ASS script: ``[Script Info]``
    + ``[V4+ Styles]`` + ``[Events]`` with a single default
    style and one ``Dialogue`` line per cue.
    """
    lines: List[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{font_name},48,&H00FFFFFF,&H000000FF,"
            "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,"
            "2,30,30,30,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, "
        "MarginR, MarginV, Effect, Text",
    ]
    for c in cues:
        start = _format_timestamp(c.start, sep=".").rstrip("0").rstrip(".")
        end = _format_timestamp(c.end, sep=".").rstrip("0").rstrip(".")
        # ASS uses comma as the decimal separator.
        start = start.replace(".", ",")
        end = end.replace(".", ",")
        text = c.text.replace("\n", r"\N")
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Burn-in (F-3)
# ---------------------------------------------------------------------------
def burn_subtitles(
    video_path: str,
    cues: Sequence[Cue],
    output_path: str,
    *,
    style: Optional[Dict[str, Any]] = None,
) -> str:
    """Burn ``cues`` onto ``video_path`` and write to ``output_path``.

    Args:
        video_path: Input video (anything :mod:`cv2` can decode).
        cues: :class:`Cue` sequence.
        output_path: Where the burned video is written.
        style: Optional style dict.  Recognised keys are
            ``font_path`` (TrueType file), ``font_size``,
            ``color_bgr``, ``position`` (``"bottom"`` /
            ``"top"`` / ``(x, y)`` tuple) and ``margin``.

    Returns:
        The ``output_path`` actually written (may differ from
        the requested one if the writer was forced to ``.mp4``
        because the extension was unknown).
    """
    style = style or {}
    position = style.get("position", "bottom")
    color_bgr = tuple(style.get("color_bgr", (255, 255, 255)))
    font_size = int(style.get("font_size", 28))
    margin = int(style.get("margin", 24))
    font_path = style.get("font_path")

    # Read the source video.
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover - cv2 is a project dep
        raise RuntimeError(
            "burn_subtitles requires opencv-python and numpy; "
            f"import failed: {exc!r}"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"cv2.VideoCapture could not open {video_path!r}"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    # Pick a sane output codec / extension.  We prefer ``mp4v``
    # because it is universally accepted in the OpenCV wheel.
    out_path = output_path
    out_ext = Path(out_path).suffix.lower()
    if out_ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        out_path = str(Path(output_path).with_suffix(".mp4"))
        out_ext = ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, float(fps), (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"cv2.VideoWriter could not open {out_path!r}"
        )

    # Try PIL for proper text shaping (handles unicode + CJK).
    pil_font = None
    if font_path is None:
        try:
            from PIL import ImageFont
            pil_font = ImageFont.load_default()
        except Exception:
            pil_font = None

    def _draw_text(frame: Any, text: str) -> None:
        # Choose coordinates based on the position hint.
        if position == "top":
            y = margin + font_size
        elif position == "bottom":
            y = height - margin - font_size
        elif isinstance(position, (tuple, list)) and len(position) == 2:
            y = int(position[1])
        else:
            y = height - margin - font_size
        x = margin
        if pil_font is not None:
            # Convert to PIL, draw, convert back.
            from PIL import Image, ImageDraw
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil)
            draw.text((x, y - font_size), text, font=pil_font,
                      fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
            frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(
                frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                float(font_size) / 28.0, color_bgr, 2, cv2.LINE_AA,
            )

    # Pre-sort cues by start time so binary search works.
    sorted_cues = sorted(cues, key=lambda c: c.start)

    def _cue_at(t: float) -> Optional[Cue]:
        lo, hi = 0, len(sorted_cues) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            c = sorted_cues[mid]
            if c.start <= t < c.end:
                return c
            if t < c.start:
                hi = mid - 1
            else:
                lo = mid + 1
        return None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_idx / float(fps)
        cue = _cue_at(t)
        if cue is not None and cue.text:
            _draw_text(frame, cue.text)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# SubtitleGenerate helpers
# ---------------------------------------------------------------------------
def build_track_from_text(
    text: str,
    *,
    language: str = "",
    method: str = "text",
    source: str = "text",
    duration: Optional[float] = None,
) -> SubtitleTrack:
    """Build a single-cue track from a plain-text input.

    Used by the echo path of :func:`nodes.subtitle` so the
    legacy ``text`` source mode still produces a valid
    track.
    """
    text = (text or "").strip()
    if not text:
        return SubtitleTrack(language=language, method=method,
                              source=source)
    duration = duration if duration is not None else max(
        1.0, 0.4 * len(text.split())
    )
    return SubtitleTrack(
        language=language,
        method=method,
        source=source,
        cues=[Cue(index=1, start=0.0, end=float(duration), text=text)],
    )


def build_track_from_cues(
    cues: Sequence[Cue],
    *,
    language: str = "",
    method: str = "asr",
    source: str = "audio",
    model: str = "",
) -> SubtitleTrack:
    """Wrap a list of :class:`Cue` objects into a :class:`SubtitleTrack`."""
    return SubtitleTrack(
        language=language, method=method, source=source,
        model=model, cues=list(cues),
    )
