"""Base64 / media serialisation helpers for the serving API (v0.6.x).

These helpers cover two opposite directions:

* :func:`_image_to_b64` / :func:`_audio_to_b64` /
  :func:`_video_to_b64` -- encode a *real* media object (PIL image,
  waveform tensor, frames tensor) as a base64 string so it can
  ride in a JSON response.
* :func:`_decode_b64_image` / :func:`_decode_b64_audio` -- decode
  a base64 string coming back from the client into a real
  media object the node system can consume.

Plus the high-level dispatcher :func:`_media_payload` which
picks the right encoder based on the MIME type and gracefully
falls back to a JSON summary when the node system returns
placeholder dicts (the contract for unimplemented nodes).

The CLI's artefact-savers live in
:mod:`serving.cli._runtime` (writing to disk) and are
intentionally NOT duplicated here -- the two code paths have
different "output" contracts.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

__all__ = [
    "_image_to_b64",
    "_audio_to_b64",
    "_video_to_b64",
    "_media_payload",
    "_decode_b64_image",
    "_decode_b64_audio",
]


def _image_to_b64(image: Any) -> str:
    """Encode a PIL image to a base64 JPEG string."""
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _audio_to_b64(audio: Any) -> str:
    """Encode an audio object to a base64 WAV string.

    Accepts either a real audio object exposing ``numpy`` / ``waveform``
    / ``sample_rate`` attributes or a placeholder dict returned by the
    node system (in which case an empty string is returned).
    """
    import numpy as np
    import wave

    waveform = getattr(audio, "numpy", None)
    if waveform is None:
        waveform = getattr(audio, "waveform", None)
    if waveform is None:
        return ""
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform[0]  # take first channel
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    sample_rate = getattr(audio, "sample_rate", 22050)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_to_b64(video: Any) -> str:
    """Encode a video object to a base64 GIF string.

    Accepts either a real video object exposing ``frames`` / ``fps``
    attributes (a tensor or array of shape ``[T, C, H, W]`` or
    ``[T, H, W, C]``) or a placeholder dict returned by the node system
    (in which case an empty string is returned).
    """
    from PIL import Image as PILImage
    import numpy as np

    frames = getattr(video, "frames", None)
    if frames is None:
        return ""
    frames_np = np.asarray(frames)
    if frames_np.ndim == 5:
        frames_np = frames_np[0]
    # Normalise to [T, H, W, C] uint8.
    if frames_np.ndim == 4 and frames_np.shape[-1] not in (1, 3, 4):
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
    frames_np = (np.clip(frames_np, 0, 1) * 255).astype("uint8") \
        if frames_np.dtype.kind == "f" else frames_np.astype("uint8")

    pil_frames = [PILImage.fromarray(f) for f in frames_np]
    if not pil_frames:
        return ""
    fps = getattr(video, "fps", 8)
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _media_payload(media: Any, mime: str) -> str:
    """Build the choice text for a media output.

    Real media objects are base64-encoded into a ``data:<mime>;base64,...``
    URI; placeholder dicts returned by the node system are serialised as
    JSON so the response stays informative even without a real backend.
    """
    b64 = ""
    if mime.startswith("image"):
        b64 = _image_to_b64(media)
    elif mime.startswith("audio"):
        b64 = _audio_to_b64(media)
    elif mime.startswith("video") or mime == "image/gif":
        b64 = _video_to_b64(media)
    if b64:
        return f"data:{mime};base64,{b64}"
    # Placeholder dict or unsupported object -> JSON summary.
    try:
        return json.dumps(media, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(media)


def _decode_b64_image(b64_str: str) -> Any:
    """Decode a base64-encoded image string into a PIL image."""
    from PIL import Image as PILImage

    # Guard against decompression bombs: cap the maximum decoded pixel
    # count and reject oversized base64 payloads before decoding.
    PILImage.MAX_IMAGE_PIXELS = 50_000_000  # 50M pixels limit
    if len(b64_str) > 10 * 1024 * 1024:  # 10MB base64 limit
        raise ValueError("Image too large")

    raw = base64.b64decode(b64_str)
    return PILImage.open(io.BytesIO(raw))


def _decode_b64_audio(b64_str: str) -> Any:
    """Decode a base64-encoded WAV string into a waveform array.

    Returns a plain ``(waveform, sample_rate)`` tuple.  The node system
    operates on plain arrays, so this helper stays free of any
    engine-specific tensor types.
    """
    import numpy as np
    import wave

    raw = base64.b64decode(b64_str)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames_data = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio_np = np.frombuffer(frames_data, dtype=np.int16).astype("float32") / 32768.0
    elif sampwidth == 1:
        audio_np = np.frombuffer(frames_data, dtype=np.uint8).astype("float32") / 128.0 - 1.0
    else:
        audio_np = np.zeros(1024, dtype="float32")

    if n_channels > 1:
        audio_np = audio_np[::n_channels]
    return audio_np, framerate
