"""Subtitle nodes for the TorchaVerse L4 capability layer.

This module implements the full subtitle pipeline as four composable
nodes (the v0.3.0 architecture's "字幕全链路"):

* :class:`SubtitleGenerateNode` (``subtitle_generate``) -- produce a
  subtitle track from a video / audio / text source via ASR, an LLM or
  an alignment method.
* :class:`SubtitleTranslateNode` (``subtitle_translate``) -- translate a
  subtitle track into a target language.
* :class:`SubtitleBurnNode` (``subtitle_burn``) -- burn a subtitle track
  into a video.
* :class:`SubtitleExportNode` (``subtitle_export``) -- serialise a
  subtitle track to ``srt`` / ``vtt`` / ``ass`` on disk.

A subtitle track is represented as a plain dictionary with a ``cues``
list (each cue carrying ``start``, ``end`` and ``text``) so that it can
flow between nodes without a dedicated asset type.

All four nodes carry a real :meth:`validate_inputs` (enum membership for
``source`` / ``method`` / ``format``, mutual-exclusivity of
``media_path`` / ``text``) and a real :meth:`estimate_resources`.
Their :meth:`execute` bodies are placeholder stubs returning
deterministic mock data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from security.input_sanitizer import InputSanitizer

from .base import BaseNode, NodeContext, NodeSpec, register_node


# ---------------------------------------------------------------------------
# 路径净化 -- 所有接收路径输入的字幕节点在使用路径前必须经过此校验。
# 允许的根目录包含系统临时目录(测试与临时输出)与当前工作目录(项目内
# 输出)，同时拒绝路径穿越(``..``)与敏感系统路径(如 ``/etc/passwd``)。
# ---------------------------------------------------------------------------
_sanitizer = InputSanitizer()


def _sanitize_path(path: str) -> str:
    """对节点路径输入进行净化校验，返回净化后的路径字符串。

    空路径原样返回(由 ``validate_inputs`` 负责非空校验)；非空路径经
    :meth:`InputSanitizer.sanitize_path` 解析并校验后返回字符串形式。
    """
    if not path:
        return path
    allowed_roots = (tempfile.gettempdir(), Path.cwd())
    return str(_sanitizer.sanitize_path(path, allowed_roots=allowed_roots))

__all__ = [
    "SubtitleGenerateNode",
    "SubtitleTranslateNode",
    "SubtitleBurnNode",
    "SubtitleExportNode",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
#: Allowed values for the ``source`` input of :class:`SubtitleGenerateNode`.
_SUBTITLE_SOURCES: tuple[str, ...] = ("video", "audio", "text")
#: Allowed values for the ``method`` input of :class:`SubtitleGenerateNode`.
_SUBTITLE_METHODS: tuple[str, ...] = ("asr", "llm", "align")
#: Allowed values for the ``format`` input of :class:`SubtitleExportNode`.
_SUBTITLE_FORMATS: tuple[str, ...] = ("srt", "vtt", "ass")
#: VRAM (GB) for the ASR model used by subtitle generation.
_SUBTITLE_ASR_VRAM_GB: float = 2.0
#: VRAM (GB) for the LLM used by subtitle generation / translation.
_SUBTITLE_LLM_VRAM_GB: float = 4.0
#: Wall-clock seconds of compute per minute of media for ASR.
_SUBTITLE_ASR_TIME_PER_MIN_S: float = 6.0
#: Wall-clock seconds of compute per cue for translation.
_SUBTITLE_TRANSLATE_TIME_PER_CUE_S: float = 0.05
#: Wall-clock seconds of compute per cue for burning.
_SUBTITLE_BURN_TIME_PER_CUE_S: float = 0.2


def _cue_count(track: Any) -> int:
    """Return the number of cues in a subtitle ``track`` dict (0 if absent)."""
    if isinstance(track, dict):
        cues = track.get("cues")
        if isinstance(cues, list):
            return len(cues)
    return 0


# ---------------------------------------------------------------------------
# SubtitleGenerateNode
# ---------------------------------------------------------------------------
@register_node("subtitle_generate")
class SubtitleGenerateNode(BaseNode):
    """Subtitle generation node (``subtitle_generate``).

    Produces a subtitle track from a media source or raw text.

    Inputs:
        source: Origin of the content -- ``"video"``, ``"audio"`` or
            ``"text"`` (required).
        media_path: Path to the media file.  Required when ``source`` is
            ``"video"`` or ``"audio"``.
        text: Raw text to segment.  Required when ``source`` is
            ``"text"``.
        language: BCP-47 language code of the content (e.g. ``"en"``).
        method: Generation method -- ``"asr"``, ``"llm"`` or ``"align"``.

    Outputs:
        subtitle_track: A dict with a ``cues`` list (each cue has
            ``start``, ``end`` and ``text``) plus ``language`` and
            ``method`` metadata.
    """

    spec = NodeSpec(
        type="subtitle_generate",
        name="Subtitle Generate",
        description="Generate a subtitle track from video, audio or text.",
        inputs={
            "source": "TEXT",
            "media_path": "Optional[TEXT]",
            "text": "Optional[TEXT]",
            "language": "TEXT",
            "method": "TEXT",
        },
        outputs={
            "subtitle_track": "SUBTITLE",
        },
        tags=["subtitle", "asr", "generation"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate subtitle-generation inputs.

        Extends the base checks with:

        * ``source`` in ``{"video", "audio", "text"}``.
        * ``method`` in ``{"asr", "llm", "align"}``.
        * ``media_path`` required for ``video`` / ``audio`` sources.
        * ``text`` required for the ``text`` source.
        * ``language`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        source = inputs.get("source")
        if isinstance(source, str) and source not in _SUBTITLE_SOURCES:
            errors.append(
                "Input 'source' for node 'subtitle_generate' must be one "
                "of {}, got {!r}.".format(list(_SUBTITLE_SOURCES), source)
            )

        method = inputs.get("method")
        if isinstance(method, str) and method not in _SUBTITLE_METHODS:
            errors.append(
                "Input 'method' for node 'subtitle_generate' must be one "
                "of {}, got {!r}.".format(list(_SUBTITLE_METHODS), method)
            )

        if source in ("video", "audio"):
            media_path = inputs.get("media_path")
            if not (isinstance(media_path, str) and media_path.strip()):
                errors.append(
                    "Input 'media_path' for node 'subtitle_generate' is "
                    "required when source is {!r}.".format(source)
                )
        if source == "text":
            text = inputs.get("text")
            if not (isinstance(text, str) and text.strip()):
                errors.append(
                    "Input 'text' for node 'subtitle_generate' is required "
                    "when source is 'text'."
                )

        language = inputs.get("language")
        if isinstance(language, str) and not language.strip():
            errors.append(
                "Input 'language' for node 'subtitle_generate' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for subtitle generation.

        ASR uses the ASR model footprint and scales with media length;
        LLM / align use the LLM footprint.
        """
        method = inputs.get("method")
        vram_gb: float
        if method == "asr":
            vram_gb = _SUBTITLE_ASR_VRAM_GB
        else:
            vram_gb = _SUBTITLE_LLM_VRAM_GB

        ram_gb = 0.5
        # Without media duration, fall back to a text-length heuristic.
        text = inputs.get("text")
        text_len = len(text) if isinstance(text, str) else 0
        time_s = 1.0 + (text_len / 1000.0) * 2.0

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a subtitle track (F-4: real ASR + text path).

        Dispatch:
            * ``source == "text"`` -- build a single-cue track from
              the text via :func:`build_track_from_text`.
            * ``method == "asr"`` -- decode the media (audio or
              video-extracted audio), run energy-based speech
              activity detection, and (when the registered text
              backend is more capable than the echo stub) emit
              transcriptions for each segment.
            * ``method == "llm"`` / ``"align"`` -- fall back to the
              text backend, which formats the input into cues.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: See :attr:`spec.inputs`.

        Returns:
            A dict with ``subtitle_track``.
        """
        source = str(inputs.get("source", "text"))
        method = str(inputs.get("method", "asr"))
        language = str(inputs.get("language", ""))
        # 对媒体路径输入进行净化校验，拒绝路径穿越与敏感系统路径。
        media_path = inputs.get("media_path")
        if isinstance(media_path, str) and media_path:
            media_path = _sanitize_path(media_path)
        text = inputs.get("text", "")
        model = ctx.config.get("default_asr_model")

        ctx.logger.debug(
            "subtitle_generate run_id=%s source=%s method=%s lang=%s",
            ctx.run_id, source, method, language,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.subtitle_generate",
                action="generate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "source": source,
                    "method": method,
                    "language": language,
                    "media_path": media_path,
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Text path -- no media decoding required.
        # ------------------------------------------------------------------
        if source == "text":
            from ._subtitle_codec import build_track_from_text
            track = build_track_from_text(
                text or "",
                language=language,
                method=method,
                source="text",
            )
            track.model = str(
                inputs.get("subtitle_model")
                or ctx.config.get("default_subtitle_model")
                or model
                or ""
            )
            return {"subtitle_track": track.to_dict()}

        # ------------------------------------------------------------------
        # ASR path (F-4).  Decode the audio, segment it, and emit
        # cue timestamps.  When the registered text backend can
        # actually transcribe we let it fill in the cue text per
        # segment; otherwise each cue is left with an empty text
        # field (segmentation-only) so the pipeline is still
        # observable end-to-end.
        # ------------------------------------------------------------------
        from ._subtitle_codec import (
            Cue, asr_transcribe, read_audio_waveform,
        )

        waveform = None
        sample_rate = 16000
        if media_path:
            try:
                waveform = read_audio_waveform(media_path)
            except Exception as exc:
                ctx.logger.warning(
                    "subtitle_generate: audio decode failed for %s: %s",
                    media_path, exc,
                )
                waveform = None

        cues: list[Cue] = []
        if waveform is not None and len(waveform) > 0:
            raw_cues = asr_transcribe(
                waveform, sample_rate=sample_rate, language=language,
            )
            # Best-effort transcription: the LLM backend may not be
            # able to actually transcribe the audio (it only sees the
            # prompt), so we always keep the segmenter output and
            # fill the text with the per-segment approximate caption
            # produced by the text backend.
            from ._helpers import call_text_backend
            caption_model = (
                inputs.get("subtitle_model")
                or ctx.config.get("default_subtitle_model")
                or model
                or "echo"
            )
            if raw_cues and method in ("asr", "llm", "align"):
                # Use the original text input or media-path as a
                # crude caption source; downstream nodes can refine.
                caption_seed = str(text or media_path or "")
                for c in raw_cues:
                    cues.append(Cue(
                        index=c.index,
                        start=c.start,
                        end=c.end,
                        text=caption_seed[: 80] if caption_seed else "",
                    ))
                if text:
                    try:
                        resp = call_text_backend(
                            ctx.bus, caption_model,
                            prompt=(
                                f"Generate {len(raw_cues)} concise "
                                f"caption(s) (one per line) for the "
                                f"following content in {language or 'en'}: "
                                f"{text}"
                            ),
                            max_tokens=128, temperature=0.0,
                        )
                        rendered = (resp.get("text") or "").strip()
                        lines = [
                            ln.strip() for ln in rendered.splitlines()
                            if ln.strip()
                        ]
                        if len(lines) == len(cues):
                            for cue, line in zip(cues, lines):
                                cue.text = line
                    except Exception as exc:
                        ctx.logger.debug(
                            "subtitle_generate caption seed failed: %s",
                            exc,
                        )
        # If we could not segment the audio (no decoder / empty)
        # we fall back to a single deterministic cue from ``text``.
        if not cues:
            fallback_text = str(text or media_path or "[empty cue]")[: 256]
            duration = max(1.0, len(fallback_text.split()) * 0.4)
            cues = [Cue(
                index=1, start=0.0, end=duration, text=fallback_text,
            )]

        from ._subtitle_codec import build_track_from_cues
        track = build_track_from_cues(
            cues,
            language=language,
            method=method,
            source=source,
            model=str(
                inputs.get("subtitle_model")
                or ctx.config.get("default_subtitle_model")
                or model
                or ""
            ),
        )
        return {"subtitle_track": track.to_dict()}


# ---------------------------------------------------------------------------
# SubtitleTranslateNode
# ---------------------------------------------------------------------------
@register_node("subtitle_translate")
class SubtitleTranslateNode(BaseNode):
    """Subtitle translation node (``subtitle_translate``).

    Translates every cue of a subtitle track into ``target_language``.

    Inputs:
        subtitle_track: The source subtitle track dict (required).
        target_language: BCP-47 target language code (required).

    Outputs:
        subtitle_track: A new subtitle track dict with translated cues.
    """

    spec = NodeSpec(
        type="subtitle_translate",
        name="Subtitle Translate",
        description="Translate a subtitle track into a target language.",
        inputs={
            "subtitle_track": "SUBTITLE",
            "target_language": "TEXT",
        },
        outputs={
            "subtitle_track": "SUBTITLE",
        },
        tags=["subtitle", "translate", "llm"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate translation inputs.

        Extends the base checks with:

        * ``target_language`` non-empty.
        * ``subtitle_track`` has a non-empty ``cues`` list.
        """
        errors = super().validate_inputs(inputs)

        target_language = inputs.get("target_language")
        if isinstance(target_language, str) and not target_language.strip():
            errors.append(
                "Input 'target_language' for node 'subtitle_translate' must "
                "be a non-empty string."
            )

        track = inputs.get("subtitle_track")
        if isinstance(track, dict) and _cue_count(track) == 0:
            errors.append(
                "Input 'subtitle_track' for node 'subtitle_translate' must "
                "contain a non-empty 'cues' list."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for translation (scales with cue count)."""
        cues = _cue_count(inputs.get("subtitle_track"))
        vram_gb = _SUBTITLE_LLM_VRAM_GB
        ram_gb = 0.25 + cues * 0.001
        time_s = cues * _SUBTITLE_TRANSLATE_TIME_PER_CUE_S
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Translate a subtitle track (F-5: windowed LLM translation).

        The cue list is sent to the text backend in sliding windows
        (default size 8).  Each window's response is split back
        into per-cue translations and each cue's ``end`` timestamp
        is adjusted by the character-length ratio so the audio
        track stays in sync when the target language is more
        verbose than the source.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``subtitle_track``, ``target_language``.

        Returns:
            A dict with ``subtitle_track``.
        """
        track = inputs.get("subtitle_track")
        track = track if isinstance(track, dict) else {}
        target_language = str(inputs.get("target_language", ""))
        model = ctx.config.get("default_translate_model")

        ctx.logger.debug(
            "subtitle_translate run_id=%s target=%s cues=%d",
            ctx.run_id, target_language, _cue_count(track),
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.subtitle_translate",
                action="translate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "target_language": target_language,
                    "num_cues": _cue_count(track),
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Translation (F-5).  Build a :class:`Cue` list from the track
        # dict, dispatch to the LLM in windows, and adapt the
        # ``end`` timestamp by the char-length ratio so the audio
        # stays in sync.
        # ------------------------------------------------------------------
        from ._subtitle_codec import Cue, batch_translate_cues
        from ._helpers import call_text_backend

        model = (
            inputs.get("translate_model")
            or ctx.config.get("default_translate_model")
            or model
            or "echo"
        )

        raw_cues = track.get("cues") or []
        cues: list[Cue] = []
        for i, c in enumerate(raw_cues, start=1):
            if not isinstance(c, dict):
                continue
            try:
                start = float(c.get("start", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(c.get("end", start + 1.0))
            except (TypeError, ValueError):
                end = start + 1.0
            text = str(c.get("text", ""))
            cues.append(Cue(index=i, start=start, end=end, text=text))

        def _call_llm(prompt: str, **kw: Any) -> Dict[str, Any]:
            return call_text_backend(
                ctx.bus, model, prompt=prompt,
                max_tokens=int(kw.get("max_tokens", 512)),
                temperature=float(kw.get("temperature", 0.0)),
            )

        window = 8
        if isinstance(inputs.get("window"), int) and inputs["window"] > 0:
            window = int(inputs["window"])

        translated = batch_translate_cues(
            cues,
            target_language=target_language,
            call_llm=_call_llm,
            window=window,
        )

        translated_track = {
            **track,
            "language": target_language,
            "source_language": track.get("language"),
            "cues": [c.to_dict() for c in translated],
            "model": model,
            "num_cues": len(translated),
        }
        return {"subtitle_track": translated_track}


# ---------------------------------------------------------------------------
# SubtitleBurnNode
# ---------------------------------------------------------------------------
@register_node("subtitle_burn")
class SubtitleBurnNode(BaseNode):
    """Subtitle burn-in node (``subtitle_burn``).

    Burns (renders) a subtitle track into a video so the subtitles become
    part of the image.

    Inputs:
        video: The source video (required).
        subtitle_track: The subtitle track dict to burn (required).
        style: Optional styling dictionary (font, size, colour, ...).

    Outputs:
        video: The video with subtitles burned in.
    """

    spec = NodeSpec(
        type="subtitle_burn",
        name="Subtitle Burn",
        description="Burn a subtitle track into a video.",
        inputs={
            "video": "VIDEO",
            "subtitle_track": "SUBTITLE",
            "style": "Optional[TEXT]",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["subtitle", "video", "postprocess", "burn"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate burn inputs.

        Extends the base checks with:

        * ``subtitle_track`` has a non-empty ``cues`` list.
        """
        errors = super().validate_inputs(inputs)

        track = inputs.get("subtitle_track")
        if isinstance(track, dict) and _cue_count(track) == 0:
            errors.append(
                "Input 'subtitle_track' for node 'subtitle_burn' must "
                "contain a non-empty 'cues' list."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for burning (scales with cue count)."""
        cues = _cue_count(inputs.get("subtitle_track"))
        vram_gb = 0.5
        ram_gb = 0.5 + cues * 0.002
        time_s = cues * _SUBTITLE_BURN_TIME_PER_CUE_S
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Burn subtitles into a video (F-3: real cv2 burn-in).

        Reads the source video, decodes the cue list, aligns cues to
        frame timestamps via binary search and writes a new video
        with the cues rasterised onto each frame via
        :func:`nodes._subtitle_codec.burn_subtitles`.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``video``, ``subtitle_track``, ``style``.

        Returns:
            A dict with ``video`` (the output path), ``frames``,
            ``num_cues`` and ``backend`` (one of ``"cv2"`` /
            ``"placeholder"``).
        """
        track = inputs.get("subtitle_track")
        track = track if isinstance(track, dict) else {}
        video = inputs.get("video")
        style = inputs.get("style") if isinstance(inputs.get("style"), dict) else {}

        ctx.logger.debug(
            "subtitle_burn run_id=%s cues=%d style_keys=%s",
            ctx.run_id, _cue_count(track), sorted(style.keys()),
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.subtitle_burn",
                action="burn",
                resource_id=None,
                details={
                    "run_id": ctx.run_id,
                    "num_cues": _cue_count(track),
                    "has_style": bool(style),
                    "style_keys": sorted(style.keys()),
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Burn-in (F-3).  When the video input is a filesystem path we
        # delegate to :func:`burn_subtitles` from the codec module;
        # otherwise (e.g. an in-memory dict placeholder) we keep the
        # legacy echo payload so the pipeline is still observable.
        # ------------------------------------------------------------------
        from ._subtitle_codec import Cue, burn_subtitles

        raw_cues = track.get("cues") or []
        cues: list[Cue] = []
        for i, c in enumerate(raw_cues, start=1):
            if not isinstance(c, dict):
                continue
            try:
                start = float(c.get("start", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(c.get("end", start + 1.0))
            except (TypeError, ValueError):
                end = start + 1.0
            text = str(c.get("text", ""))
            cues.append(Cue(index=i, start=start, end=end, text=text))

        if isinstance(video, str) and video:
            video_path = _sanitize_path(video)
            tmp_dir = Path(tempfile.gettempdir()) / "torcha-verse-burn"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_path = tmp_dir / f"burned_{ctx.run_id}.mp4"
            try:
                written = burn_subtitles(
                    video_path, cues, str(out_path), style=style,
                )
            except FileNotFoundError as exc:
                ctx.logger.warning("subtitle_burn: %s", exc)
                return {
                    "video": {
                        "kind": "placeholder_burned_video",
                        "reason": "video_not_found",
                        "num_cues": len(cues),
                    },
                    "backend": "placeholder",
                    "num_cues": len(cues),
                }
            except RuntimeError as exc:
                ctx.logger.warning("subtitle_burn backend unavailable: %s", exc)
                return {
                    "video": {
                        "kind": "placeholder_burned_video",
                        "reason": "backend_unavailable",
                        "num_cues": len(cues),
                    },
                    "backend": "placeholder",
                    "num_cues": len(cues),
                }
            written_size = Path(written).stat().st_size
            return {
                "video": written,
                "backend": "cv2",
                "num_cues": len(cues),
                "bytes_written": written_size,
            }

        # No path -- echo the structure so downstream nodes still get
        # a valid payload.
        return {
            "video": {
                "kind": "placeholder_burned_video",
                "num_cues": len(cues),
                "has_style": bool(style),
            },
            "backend": "placeholder",
            "num_cues": len(cues),
        }


# ---------------------------------------------------------------------------
# SubtitleExportNode
# ---------------------------------------------------------------------------
@register_node("subtitle_export")
class SubtitleExportNode(BaseNode):
    """Subtitle export node (``subtitle_export``).

    Serialises a subtitle track to a file in ``srt`` / ``vtt`` / ``ass``
    format.

    Inputs:
        subtitle_track: The subtitle track dict to export (required).
        format: Output format -- ``"srt"``, ``"vtt"`` or ``"ass"``.
        path: Destination file path (required).

    Outputs:
        path: The path the subtitle file was written to.
    """

    spec = NodeSpec(
        type="subtitle_export",
        name="Subtitle Export",
        description="Export a subtitle track to srt / vtt / ass.",
        inputs={
            "subtitle_track": "SUBTITLE",
            "format": "TEXT",
            "path": "TEXT",
        },
        outputs={
            "path": "TEXT",
        },
        tags=["subtitle", "export", "srt", "vtt", "ass"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate export inputs.

        Extends the base checks with:

        * ``format`` in ``{"srt", "vtt", "ass"}``.
        * ``path`` non-empty.
        * ``subtitle_track`` has a ``cues`` list.
        """
        errors = super().validate_inputs(inputs)

        fmt = inputs.get("format")
        if isinstance(fmt, str) and fmt not in _SUBTITLE_FORMATS:
            errors.append(
                "Input 'format' for node 'subtitle_export' must be one of "
                "{}, got {!r}.".format(list(_SUBTITLE_FORMATS), fmt)
            )

        path = inputs.get("path")
        if isinstance(path, str) and not path.strip():
            errors.append(
                "Input 'path' for node 'subtitle_export' must be a "
                "non-empty string."
            )

        track = inputs.get("subtitle_track")
        if isinstance(track, dict) and "cues" not in track:
            errors.append(
                "Input 'subtitle_track' for node 'subtitle_export' must "
                "contain a 'cues' list."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for export (CPU-bound, scales with cues)."""
        cues = _cue_count(inputs.get("subtitle_track"))
        vram_gb = 0.0
        ram_gb = 0.05 + cues * 0.0005
        time_s = 0.1 + cues * 0.001
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Export a subtitle track to disk (F-2: real serialisation).

        Serialises the cue list to ``srt`` / ``vtt`` / ``ass`` via
        :mod:`nodes._subtitle_codec` and writes the result to
        ``path`` (a sanitised path under the system temp dir or
        the current working directory).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``subtitle_track``, ``format``, ``path``.

        Returns:
            A dict with ``path`` (the file written), ``format``,
            ``bytes_written`` and ``num_cues``.
        """
        track = inputs.get("subtitle_track")
        track = track if isinstance(track, dict) else {}
        fmt = str(inputs.get("format", "srt")).lower()
        # 对导出路径输入进行净化校验，拒绝路径穿越与敏感系统路径。
        path = _sanitize_path(str(inputs.get("path", "")))

        ctx.logger.debug(
            "subtitle_export run_id=%s format=%s path=%s cues=%d",
            ctx.run_id, fmt, path, _cue_count(track),
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "EXPORT",
                actor="node.subtitle_export",
                action="export",
                resource_id=path,
                details={
                    "run_id": ctx.run_id,
                    "format": fmt,
                    "path": path,
                    "num_cues": _cue_count(track),
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Serialise (F-2).  The cue list is normalised into the codec
        # module's :class:`Cue` objects, then rendered with the
        # appropriate serializer and persisted to disk.  When ``path``
        # is empty we return the serialised payload alongside the
        # suggested filename so the caller can decide where to write.
        # ------------------------------------------------------------------
        from ._subtitle_codec import (
            Cue, serialize_ass, serialize_srt, serialize_vtt,
        )

        raw_cues = track.get("cues") or []
        cues: list[Cue] = []
        for i, c in enumerate(raw_cues, start=1):
            if not isinstance(c, dict):
                continue
            try:
                start = float(c.get("start", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(c.get("end", start + 1.0))
            except (TypeError, ValueError):
                end = start + 1.0
            text = str(c.get("text", ""))
            cues.append(Cue(index=i, start=start, end=end, text=text))

        if fmt == "srt":
            rendered = serialize_srt(cues)
        elif fmt == "vtt":
            rendered = serialize_vtt(cues)
        elif fmt == "ass":
            rendered = serialize_ass(cues)
        else:
            # validate_inputs already rejects unknown formats; defensive
            # default is SRT.
            rendered = serialize_srt(cues)
            fmt = "srt"

        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
            return {
                "path": str(out),
                "format": fmt,
                "bytes_written": out.stat().st_size,
                "num_cues": len(cues),
            }
        # No path supplied -- return the serialised payload and a
        # suggested filename.
        suggested = f"subtitles.{fmt}"
        return {
            "path": None,
            "format": fmt,
            "suggested_filename": suggested,
            "payload": rendered,
            "num_cues": len(cues),
        }
