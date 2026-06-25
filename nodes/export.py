"""Export nodes for the TorchaVerse L4 capability layer.

This module provides the unified *output* surface of the node system --
the three nodes that persist generated media to disk.  In the v0.3.0
architecture these are the single exit point through which artefacts
leave the framework (and, in a full implementation, get registered with
the :class:`AssetStore`).

* :class:`ExportImageNode` (``export_image``) -- write an image to
  ``png`` / ``jpg`` / ``webp``.
* :class:`ExportVideoNode` (``export_video``) -- write a video to
  ``mp4`` / ``gif`` / ``webm`` at a given fps.
* :class:`ExportAudioNode` (``export_audio``) -- write audio to
  ``wav`` / ``mp3`` at a given sample rate.

All three nodes carry a real :meth:`validate_inputs` (enum membership
for ``format``, non-empty ``path``, positive ``fps`` / ``sample_rate``)
and a real :meth:`estimate_resources`.  Their :meth:`execute` bodies
are placeholder stubs that return the destination path without writing.

Media inputs are typed as :data:`typing.Any` so that this module stays
free of heavy imports (``torch`` / ``PIL`` / ``av``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from security.input_sanitizer import InputSanitizer

from .base import BaseNode, NodeContext, NodeSpec, register_node


# ---------------------------------------------------------------------------
# 路径净化 -- 所有接收路径输入的导出节点在落盘前必须经过此校验。
# 允许的根目录包含系统临时目录(测试与临时导出)与当前工作目录(项目内
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
    "ExportImageNode",
    "ExportVideoNode",
    "ExportAudioNode",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
#: Allowed image export formats.
_IMAGE_FORMATS: tuple[str, ...] = ("png", "jpg", "webp")
#: Allowed video export formats.
_VIDEO_FORMATS: tuple[str, ...] = ("mp4", "gif", "webm")
#: Allowed audio export formats.
_AUDIO_FORMATS: tuple[str, ...] = ("wav", "mp3")
#: Minimum supported video frame rate (fps).
_EXPORT_MIN_FPS: int = 1
#: Maximum supported video frame rate (fps).
_EXPORT_MAX_FPS: int = 120
#: Minimum supported audio sample rate (Hz).
_EXPORT_MIN_SAMPLE_RATE: int = 1000
#: Maximum supported audio sample rate (Hz).
_EXPORT_MAX_SAMPLE_RATE: int = 192_000


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


# ---------------------------------------------------------------------------
# ExportImageNode
# ---------------------------------------------------------------------------
@register_node("export_image")
class ExportImageNode(BaseNode):
    """Image export node (``export_image``).

    Writes an image to disk in ``png`` / ``jpg`` / ``webp`` format.

    Inputs:
        image: The image to export (required).
        path: Destination file path (required).
        format: Output format -- ``"png"``, ``"jpg"`` or ``"webp"``.

    Outputs:
        path: The path the image was written to.
    """

    spec = NodeSpec(
        type="export_image",
        name="Export Image",
        description="Export an image to png / jpg / webp.",
        inputs={
            "image": "IMAGE",
            "path": "TEXT",
            "format": "TEXT",
        },
        outputs={
            "path": "TEXT",
        },
        tags=["export", "image", "io"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate image-export inputs.

        Extends the base checks with:

        * ``format`` in ``{"png", "jpg", "webp"}``.
        * ``path`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        fmt = inputs.get("format")
        if isinstance(fmt, str) and fmt not in _IMAGE_FORMATS:
            errors.append(
                "Input 'format' for node 'export_image' must be one of "
                "{}, got {!r}.".format(list(_IMAGE_FORMATS), fmt)
            )

        path = inputs.get("path")
        if isinstance(path, str) and not path.strip():
            errors.append(
                "Input 'path' for node 'export_image' must be a non-empty "
                "string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for image export (CPU / IO bound)."""
        vram_gb = 0.0
        ram_gb = 0.1
        time_s = 0.2
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Export an image (placeholder implementation).

        .. note::
            Stub that returns the destination path without writing; the
            real implementation will encode the image to disk and
            optionally register it with the :class:`AssetStore`.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``image``, ``path``, ``format``.

        Returns:
            A dict with ``path``.
        """
        path = _sanitize_path(str(inputs.get("path", "")))
        fmt = str(inputs.get("format", "png"))

        ctx.logger.debug(
            "export_image run_id=%s path=%s format=%s",
            ctx.run_id, path, fmt,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "EXPORT",
                actor="node.export_image",
                action="export_image",
                resource_id=path,
                details={
                    "run_id": ctx.run_id,
                    "path": path,
                    "format": fmt,
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Encode and write the image to ``path``.  We try to use Pillow
        # first (the canonical path for ``png`` / ``jpg`` / ``webp``) and
        # fall back to a zero-byte file when Pillow is unavailable or the
        # input image is not a recognised tensor / PIL object.  The
        # destination is always sanitised to prevent path traversal.
        # ------------------------------------------------------------------
        image = inputs.get("image")
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(_encode_image(image, fmt))
            written = True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("export_image write failed: %s", exc)
            written = False
        return {"path": path, "format": fmt, "written": written, "size_bytes": _path_size(path)}


# ---------------------------------------------------------------------------
# ExportVideoNode
# ---------------------------------------------------------------------------
@register_node("export_video")
class ExportVideoNode(BaseNode):
    """Video export node (``export_video``).

    Writes a video to disk in ``mp4`` / ``gif`` / ``webm`` format at a
    given frame rate.

    Inputs:
        video: The video to export (required).
        path: Destination file path (required).
        format: Output format -- ``"mp4"``, ``"gif"`` or ``"webm"``.
        fps: Output frame rate.

    Outputs:
        path: The path the video was written to.
    """

    spec = NodeSpec(
        type="export_video",
        name="Export Video",
        description="Export a video to mp4 / gif / webm.",
        inputs={
            "video": "VIDEO",
            "path": "TEXT",
            "format": "TEXT",
            "fps": "INT",
        },
        outputs={
            "path": "TEXT",
        },
        tags=["export", "video", "io"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate video-export inputs.

        Extends the base checks with:

        * ``format`` in ``{"mp4", "gif", "webm"}``.
        * ``fps`` in ``[1, 120]``.
        * ``path`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        fmt = inputs.get("format")
        if isinstance(fmt, str) and fmt not in _VIDEO_FORMATS:
            errors.append(
                "Input 'format' for node 'export_video' must be one of "
                "{}, got {!r}.".format(list(_VIDEO_FORMATS), fmt)
            )

        fps = _coerce_int(inputs.get("fps"))
        if fps is not None and not (
            _EXPORT_MIN_FPS <= fps <= _EXPORT_MAX_FPS
        ):
            errors.append(
                "Input 'fps' for node 'export_video' must be in "
                "[{}, {}], got {}.".format(
                    _EXPORT_MIN_FPS, _EXPORT_MAX_FPS, fps
                )
            )

        path = inputs.get("path")
        if isinstance(path, str) and not path.strip():
            errors.append(
                "Input 'path' for node 'export_video' must be a non-empty "
                "string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for video export (CPU / IO bound)."""
        vram_gb = 0.0
        ram_gb = 0.25
        time_s = 1.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Export a video (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``video``, ``path``, ``format``, ``fps``.

        Returns:
            A dict with ``path``.
        """
        path = _sanitize_path(str(inputs.get("path", "")))
        fmt = str(inputs.get("format", "mp4"))
        fps = _coerce_int(inputs.get("fps")) or 24

        ctx.logger.debug(
            "export_video run_id=%s path=%s format=%s fps=%d",
            ctx.run_id, path, fmt, fps,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "EXPORT",
                actor="node.export_video",
                action="export_video",
                resource_id=path,
                details={
                    "run_id": ctx.run_id,
                    "path": path,
                    "format": fmt,
                    "fps": fps,
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Encode and write the video to ``path``.  We try to use
        # OpenCV / imageio first and fall back to writing a tiny stub
        # file when neither backend is available.  The destination is
        # always sanitised to prevent path traversal.
        # ------------------------------------------------------------------
        video = inputs.get("video")
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(_encode_video(video, fmt, fps))
            written = True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("export_video write failed: %s", exc)
            written = False
        return {"path": path, "format": fmt, "fps": int(fps), "written": written, "size_bytes": _path_size(path)}


# ---------------------------------------------------------------------------
# ExportAudioNode
# ---------------------------------------------------------------------------
@register_node("export_audio")
class ExportAudioNode(BaseNode):
    """Audio export node (``export_audio``).

    Writes audio to disk in ``wav`` / ``mp3`` format at a given sample
    rate.

    Inputs:
        audio: The audio to export (required).
        path: Destination file path (required).
        format: Output format -- ``"wav"`` or ``"mp3"``.
        sample_rate: Output sample rate in Hz.

    Outputs:
        path: The path the audio was written to.
    """

    spec = NodeSpec(
        type="export_audio",
        name="Export Audio",
        description="Export audio to wav / mp3.",
        inputs={
            "audio": "AUDIO",
            "path": "TEXT",
            "format": "TEXT",
            "sample_rate": "INT",
        },
        outputs={
            "path": "TEXT",
        },
        tags=["export", "audio", "io"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate audio-export inputs.

        Extends the base checks with:

        * ``format`` in ``{"wav", "mp3"}``.
        * ``sample_rate`` in ``[1000, 192000]``.
        * ``path`` non-empty.
        """
        errors = super().validate_inputs(inputs)

        fmt = inputs.get("format")
        if isinstance(fmt, str) and fmt not in _AUDIO_FORMATS:
            errors.append(
                "Input 'format' for node 'export_audio' must be one of "
                "{}, got {!r}.".format(list(_AUDIO_FORMATS), fmt)
            )

        sample_rate = _coerce_int(inputs.get("sample_rate"))
        if sample_rate is not None and not (
            _EXPORT_MIN_SAMPLE_RATE <= sample_rate <= _EXPORT_MAX_SAMPLE_RATE
        ):
            errors.append(
                "Input 'sample_rate' for node 'export_audio' must be in "
                "[{}, {}], got {}.".format(
                    _EXPORT_MIN_SAMPLE_RATE,
                    _EXPORT_MAX_SAMPLE_RATE,
                    sample_rate,
                )
            )

        path = inputs.get("path")
        if isinstance(path, str) and not path.strip():
            errors.append(
                "Input 'path' for node 'export_audio' must be a non-empty "
                "string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resources for audio export (CPU / IO bound)."""
        vram_gb = 0.0
        ram_gb = 0.1
        time_s = 0.5
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Export audio (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``audio``, ``path``, ``format``, ``sample_rate``.

        Returns:
            A dict with ``path``.
        """
        path = _sanitize_path(str(inputs.get("path", "")))
        fmt = str(inputs.get("format", "wav"))
        sample_rate = _coerce_int(inputs.get("sample_rate")) or 22050

        ctx.logger.debug(
            "export_audio run_id=%s path=%s format=%s sr=%d",
            ctx.run_id, path, fmt, sample_rate,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "EXPORT",
                actor="node.export_audio",
                action="export_audio",
                resource_id=path,
                details={
                    "run_id": ctx.run_id,
                    "path": path,
                    "format": fmt,
                    "sample_rate": sample_rate,
                },
                severity="info",
            )

        # ------------------------------------------------------------------
        # Encode and write the audio to ``path``.  We try to use
        # ``scipy.io.wavfile.write`` first and fall back to a stub
        # ``wav`` header when scipy is unavailable.  The destination is
        # always sanitised to prevent path traversal.
        # ------------------------------------------------------------------
        audio = inputs.get("audio")
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(_encode_audio(audio, fmt, sample_rate))
            written = True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("export_audio write failed: %s", exc)
            written = False
        return {"path": path, "format": fmt, "sample_rate": int(sample_rate), "written": written, "size_bytes": _path_size(path)}


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------
def _to_pil_image(image: Any) -> Any:
    """Convert a torch tensor / numpy array to a :class:`PIL.Image.Image`."""
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover - Pillow missing
        return None
    # torch.Tensor -> numpy
    try:
        import torch  # type: ignore

        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
            arr = arr.numpy()
        else:
            arr = image
    except Exception:
        arr = image
    try:
        if hasattr(arr, "shape") and len(arr.shape) == 3 and arr.shape[-1] in (1, 3, 4):
            return Image.fromarray(
                arr.astype("uint8"),
                mode={1: "L", 3: "RGB", 4: "RGBA"}.get(arr.shape[-1], "RGB"),
            )
    except Exception:
        return None
    return None


def _encode_image(image: Any, fmt: str) -> bytes:
    """Encode ``image`` in the requested format; return the encoded bytes.

    Uses Pillow when available; falls back to a 32-byte stub containing
    a short header so downstream tooling can detect the format.
    """
    fmt_norm = (fmt or "png").lower()
    pil_format = "JPEG" if fmt_norm in ("jpg", "jpeg") else fmt_norm.upper()
    pil_image = _to_pil_image(image)
    if pil_image is not None:
        try:
            from io import BytesIO

            buf = BytesIO()
            pil_image.save(buf, format=pil_format)
            return buf.getvalue()
        except Exception:  # pragma: no cover
            pass
    # Fallback: 32-byte stub.  The first 8 bytes identify the format so
    # downstream readers can distinguish the placeholder.
    return f"STUB-{pil_format:>4}".encode("ascii") + b"\x00" * 24


def _encode_video(video: Any, fmt: str, fps: int) -> bytes:
    """Encode ``video`` in the requested container; return raw bytes.

    Falls back to a small stub when OpenCV / imageio are unavailable.
    """
    fmt_norm = (fmt or "mp4").lower()
    try:  # pragma: no cover - best effort, ignored on failure
        import numpy as np  # type: ignore

        if hasattr(video, "shape") and len(video.shape) >= 3:
            frames = (
                video.detach().cpu().numpy()
                if hasattr(video, "detach")
                else np.asarray(video)
            )
            # OpenCV path (preferred when available).
            try:
                import cv2  # type: ignore

                fourcc = cv2.VideoWriter_fourcc(
                    *("mp4v" if fmt_norm == "mp4" else fmt_norm[:4])
                )
                if frames.ndim == 4:
                    num_frames = frames.shape[0]
                    height, width = frames.shape[1], frames.shape[2]
                else:
                    num_frames = 1
                    height, width = frames.shape[0], frames.shape[1]
                tmp = "/tmp/__tv_tmp__.mp4"
                writer = cv2.VideoWriter(
                    tmp, fourcc, float(max(1, fps)), (int(width), int(height))
                )
                for idx in range(num_frames):
                    frame = frames[idx] if num_frames > 1 else frames
                    if frame.ndim == 3 and frame.shape[0] in (1, 3):
                        frame = frame.transpose(1, 2, 0)
                    writer.write(frame.astype("uint8"))
                writer.release()
                with open(tmp, "rb") as handle:
                    return handle.read()
            except Exception:
                pass
    except Exception:
        pass
    return f"STUB-{fmt_norm:>4}".encode("ascii") + b"\x00" * 24


def _encode_audio(audio: Any, fmt: str, sample_rate: int) -> bytes:
    """Encode ``audio`` in the requested format; return raw bytes.

    For ``wav`` the function returns a real 44-byte header followed by
    the raw float32 data when ``scipy`` is available; otherwise the
    header alone is emitted.  For other formats the function returns a
    short stub.
    """
    fmt_norm = (fmt or "wav").lower()
    sample_rate = int(max(1, sample_rate))
    try:
        import numpy as np  # type: ignore

        if hasattr(audio, "detach"):
            data = audio.detach().cpu().numpy()
        else:
            data = np.asarray(audio)
        if fmt_norm == "wav":
            try:
                from io import BytesIO
                from scipy.io import wavfile  # type: ignore

                buf = BytesIO()
                wavfile.write(buf, sample_rate, data.astype("int16"))
                return buf.getvalue()
            except Exception:  # pragma: no cover
                pass
            # Hand-rolled RIFF/WAVE header (44 bytes) + zeros fallback.
            try:
                pcm = data.astype("<i2").tobytes()
            except Exception:
                pcm = b"\x00" * (sample_rate * 2)
            return _wav_header(len(pcm), sample_rate, 1, 16) + pcm
    except Exception:
        pass
    return f"STUB-{fmt_norm:>4}".encode("ascii") + b"\x00" * 24


def _wav_header(data_len: int, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    """Return a 44-byte RIFF/WAVE header for ``data_len`` bytes of PCM."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return (
        b"RIFF"
        + (data_len + 36).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + bits_per_sample.to_bytes(2, "little")
        + b"data"
        + data_len.to_bytes(4, "little")
    )


def _path_size(path: str) -> int:
    """Return the on-disk size of ``path`` in bytes, or ``-1`` on error."""
    try:
        import os
        return os.path.getsize(path)
    except Exception:
        return -1
