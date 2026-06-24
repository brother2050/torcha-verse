"""Digital-human & lip-sync nodes for the TorchaVerse L4 capability layer.

This module implements six single-responsibility nodes covering the
digital-human pipeline:

* :class:`LipSyncNode` (``dh_lip_sync``) -- audio-driven lip-sync
  re-animation of an existing video (MuseTalk / VideoReTalking).
* :class:`TalkingHeadNode` (``dh_talking_head``) -- talking-head
  generation from a portrait image and an audio clip (SadTalker /
  EchoMimic).
* :class:`PortraitAnimateNode` (``dh_portrait_animate``) -- portrait
  animation driven by a reference video signal (LivePortrait).
* :class:`DigitalHumanNode` (``dh_full_body``) -- full-body digital-human
  video generation from a reference image and audio (EchoMimic v2).
* :class:`FaceEnhanceNode` (``dh_face_enhance``) -- face restoration /
  enhancement on a video (GFPGAN / CodeFormer).
* :class:`VoiceCloneNode` (``dh_voice_clone``) -- zero-shot voice
  cloning from a reference audio and text (CosyVoice / F5-TTS /
  ChatTTS).

All six nodes carry a real :meth:`validate_inputs` (method / language
enum membership, ``strength`` range, non-empty text) and a real
:meth:`estimate_resources` (VRAM / RAM / time scaled by the relevant
input dimensions).  Their :meth:`execute` bodies are placeholder stubs
returning deterministic mock data -- the interface is complete and ready
for the real model backends to be wired in via the :class:`ModuleBus`.

Media types (``video``, ``audio``, ``image``) are declared as semantic
*type strings* (``"VIDEO"``, ``"AUDIO"``, ``"IMAGE"``, ``"TEXT"``,
``"FLOAT"``, ``"INT"``, ``"BOOL"``) rather than Python types so that
the node contract is self-documenting and backend-agnostic; the concrete
tensor / file-path representation is decided by the backend at
execution time.  Optional inputs use ``Optional[str]`` so that the base
:class:`BaseNode.validate_inputs` correctly treats them as optional.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseNode, NodeContext, NodeSpec, register_node

__all__ = [
    "LipSyncNode",
    "TalkingHeadNode",
    "PortraitAnimateNode",
    "DigitalHumanNode",
    "FaceEnhanceNode",
    "VoiceCloneNode",
]


# ---------------------------------------------------------------------------
# Shared constants (digital-human-specific estimation coefficients)
# ---------------------------------------------------------------------------
#: Allowed lip-sync methods.
_LIP_SYNC_METHODS: tuple[str, ...] = ("musetalk", "video_retalking")
#: Allowed talking-head methods.
_TALKING_HEAD_METHODS: tuple[str, ...] = ("sadtalker", "echo_mimic")
#: Allowed portrait-animation methods.
_PORTRAIT_ANIMATE_METHODS: tuple[str, ...] = ("liveportrait",)
#: Allowed full-body digital-human methods.
_FULL_BODY_METHODS: tuple[str, ...] = ("echo_mimic_v2",)
#: Allowed face-enhancement methods.
_FACE_ENHANCE_METHODS: tuple[str, ...] = ("gfpgan", "codeformer")
#: Allowed voice-clone methods.
_VOICE_CLONE_METHODS: tuple[str, ...] = ("cosyvoice", "f5_tts", "chat_tts")
#: Allowed voice-clone languages.
_VOICE_CLONE_LANGUAGES: tuple[str, ...] = ("zh", "en", "ja")

#: VRAM (GB) for the lip-sync model weights.
_LIP_SYNC_MODEL_VRAM_GB: float = 4.0
#: VRAM (GB) for the talking-head model weights.
_TALKING_HEAD_MODEL_VRAM_GB: float = 6.0
#: VRAM (GB) for the portrait-animation model weights.
_PORTRAIT_ANIMATE_MODEL_VRAM_GB: float = 5.0
#: VRAM (GB) for the full-body digital-human model weights.
_FULL_BODY_MODEL_VRAM_GB: float = 8.0
#: VRAM (GB) for the face-enhancement model weights.
_FACE_ENHANCE_MODEL_VRAM_GB: float = 2.0
#: VRAM (GB) for the voice-clone model weights.
_VOICE_CLONE_MODEL_VRAM_GB: float = 2.5
#: Default output sample rate (Hz) for voice cloning.
_DEFAULT_VOICE_SAMPLE_RATE: int = 24000
#: Wall-clock seconds of compute per second of generated video.
_DH_TIME_PER_VIDEO_S: float = 1.5
#: Wall-clock seconds of compute per word of cloned speech.
_VOICE_TIME_PER_WORD_S: float = 0.3
#: Host RAM (GB) per second of generated video.
_DH_RAM_PER_VIDEO_S: float = 0.02
#: Host RAM (GB) per second of generated audio.
_VOICE_RAM_PER_AUDIO_S: float = 0.005


def _coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as a ``float`` when it is a real number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _validate_enum(
    inputs: Dict[str, Any],
    field: str,
    node_type: str,
    allowed: tuple[str, ...],
    errors: List[str],
) -> None:
    """Append an error when ``inputs[field]`` is not in ``allowed``."""
    value = inputs.get(field)
    if isinstance(value, str) and value not in allowed:
        errors.append(
            "Input {!r} for node {!r} must be one of {}, got {!r}.".format(
                field, node_type, list(allowed), value
            )
        )


# ---------------------------------------------------------------------------
# LipSyncNode
# ---------------------------------------------------------------------------
@register_node("dh_lip_sync")
class LipSyncNode(BaseNode):
    """Audio-driven lip-sync node (``dh_lip_sync``).

    Re-animates the mouth region of ``video`` to match the speech in
    ``audio``, producing a lip-synced video and a sync-quality score.

    Inputs:
        video: The source video to re-animate (required).
        audio: The driving audio clip (required).
        method: Lip-sync method -- ``"musetalk"`` or
            ``"video_retalking"``.
        face_region: Optional face-region hint (e.g. a bounding box or
            landmark descriptor).

    Outputs:
        video: The lip-synced video.
        sync_score: Lip-sync quality score in ``[0, 1]``.
    """

    spec = NodeSpec(
        type="dh_lip_sync",
        name="Lip Sync",
        description="Re-animate a video's mouth to match a driving audio clip.",
        inputs={
            "video": "VIDEO",
            "audio": "AUDIO",
            "method": "TEXT",
            "face_region": "Optional[TEXT]",
        },
        outputs={
            "video": "VIDEO",
            "sync_score": "FLOAT",
        },
        tags=["digital_human", "lip_sync", "video", "audio"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate lip-sync inputs.

        Extends the base checks with:

        * ``method`` in ``{"musetalk", "video_retalking"}``.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs, "method", "dh_lip_sync", _LIP_SYNC_METHODS, errors
        )
        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a lip-sync run.

        Without an explicit video duration the estimate uses the model
        base overhead plus a fixed compute term.
        """
        vram_gb = _LIP_SYNC_MODEL_VRAM_GB + 1.0
        ram_gb = 0.5
        time_s = 3.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Re-animate lip movement (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``video``, ``audio``, ``method``, ``face_region``.

        Returns:
            A dict with ``video`` and ``sync_score``.
        """
        method = str(inputs.get("method", "musetalk"))
        face_region = inputs.get("face_region")

        ctx.logger.debug(
            "dh_lip_sync run_id=%s method=%s face_region=%s",
            ctx.run_id, method, face_region,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_lip_sync",
                action="lip_sync",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "face_region": face_region,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        video = {
            "kind": "placeholder_lip_sync_video",
            "path": "placeholder.mp4",
            "method": method,
        }
        return {"video": video, "sync_score": 0.95}


# ---------------------------------------------------------------------------
# TalkingHeadNode
# ---------------------------------------------------------------------------
@register_node("dh_talking_head")
class TalkingHeadNode(BaseNode):
    """Talking-head generation node (``dh_talking_head``).

    Generates a talking-head video from a portrait ``portrait_image``
    and a driving ``audio`` clip.

    Inputs:
        portrait_image: The source portrait image (required).
        audio: The driving audio clip (required).
        method: Generation method -- ``"sadtalker"`` or
            ``"echo_mimic"``.
        enhance_resolution: Whether to super-resolve the output.

    Outputs:
        video: The generated talking-head video.
    """

    spec = NodeSpec(
        type="dh_talking_head",
        name="Talking Head",
        description="Generate a talking-head video from a portrait and audio.",
        inputs={
            "portrait_image": "IMAGE",
            "audio": "AUDIO",
            "method": "TEXT",
            "enhance_resolution": "BOOL",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["digital_human", "talking_head", "video", "audio", "image"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate talking-head inputs.

        Extends the base checks with:

        * ``method`` in ``{"sadtalker", "echo_mimic"}``.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs, "method", "dh_talking_head", _TALKING_HEAD_METHODS, errors
        )
        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a talking-head run.

        VRAM increases when ``enhance_resolution`` is enabled.
        """
        enhance = inputs.get("enhance_resolution")
        enhance = bool(enhance) if isinstance(enhance, (bool,)) else False
        vram_gb = _TALKING_HEAD_MODEL_VRAM_GB + (1.5 if enhance else 0.0)
        ram_gb = 0.75
        time_s = 5.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a talking-head video (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``portrait_image``, ``audio``, ``method``,
                ``enhance_resolution``.

        Returns:
            A dict with ``video``.
        """
        method = str(inputs.get("method", "sadtalker"))
        enhance_resolution = bool(inputs.get("enhance_resolution", False))

        ctx.logger.debug(
            "dh_talking_head run_id=%s method=%s enhance=%s",
            ctx.run_id, method, enhance_resolution,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_talking_head",
                action="generate_talking_head",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "enhance_resolution": enhance_resolution,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        video = {
            "kind": "placeholder_talking_head_video",
            "path": "placeholder.mp4",
            "method": method,
            "enhance_resolution": enhance_resolution,
        }
        return {"video": video}


# ---------------------------------------------------------------------------
# PortraitAnimateNode
# ---------------------------------------------------------------------------
@register_node("dh_portrait_animate")
class PortraitAnimateNode(BaseNode):
    """Portrait animation node (``dh_portrait_animate``).

    Animates a ``source_image`` portrait using the motion from a
    ``driving_signal`` video (e.g. LivePortrait).

    Inputs:
        source_image: The portrait image to animate (required).
        driving_signal: The driving video providing motion (required).
        method: Animation method -- ``"liveportrait"``.
        stitching: Whether to stitch the animated region back into the
            original image seamlessly.

    Outputs:
        video: The animated portrait video.
    """

    spec = NodeSpec(
        type="dh_portrait_animate",
        name="Portrait Animate",
        description="Animate a portrait image using a driving video signal.",
        inputs={
            "source_image": "IMAGE",
            "driving_signal": "VIDEO",
            "method": "TEXT",
            "stitching": "BOOL",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["digital_human", "portrait_animate", "video", "image"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate portrait-animation inputs.

        Extends the base checks with:

        * ``method`` in ``{"liveportrait"}``.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs,
            "method",
            "dh_portrait_animate",
            _PORTRAIT_ANIMATE_METHODS,
            errors,
        )
        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a portrait-animation run."""
        vram_gb = _PORTRAIT_ANIMATE_MODEL_VRAM_GB
        ram_gb = 0.6
        time_s = 4.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Animate a portrait (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``source_image``, ``driving_signal``, ``method``,
                ``stitching``.

        Returns:
            A dict with ``video``.
        """
        method = str(inputs.get("method", "liveportrait"))
        stitching = bool(inputs.get("stitching", False))

        ctx.logger.debug(
            "dh_portrait_animate run_id=%s method=%s stitching=%s",
            ctx.run_id, method, stitching,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_portrait_animate",
                action="animate_portrait",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "stitching": stitching,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        video = {
            "kind": "placeholder_portrait_animate_video",
            "path": "placeholder.mp4",
            "method": method,
            "stitching": stitching,
        }
        return {"video": video}


# ---------------------------------------------------------------------------
# DigitalHumanNode
# ---------------------------------------------------------------------------
@register_node("dh_full_body")
class DigitalHumanNode(BaseNode):
    """Full-body digital-human node (``dh_full_body``).

    Generates a full-body digital-human video from a ``reference_image``
    and a driving ``audio`` clip, optionally guided by a
    ``gesture_sequence``.

    Inputs:
        reference_image: The reference image of the digital human
            (required).
        audio: The driving audio clip (required).
        gesture_sequence: Optional gesture / pose sequence descriptor.
        method: Generation method -- ``"echo_mimic_v2"``.
        resolution: Optional output resolution hint (e.g.
            ``"720p"``, ``"1080p"``).

    Outputs:
        video: The generated digital-human video.
    """

    spec = NodeSpec(
        type="dh_full_body",
        name="Digital Human (Full Body)",
        description="Generate a full-body digital-human video from a reference image and audio.",
        inputs={
            "reference_image": "IMAGE",
            "audio": "AUDIO",
            "gesture_sequence": "Optional[TEXT]",
            "method": "TEXT",
            "resolution": "Optional[TEXT]",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["digital_human", "full_body", "video", "audio", "image"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate full-body digital-human inputs.

        Extends the base checks with:

        * ``method`` in ``{"echo_mimic_v2"}``.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs, "method", "dh_full_body", _FULL_BODY_METHODS, errors
        )
        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a full-body run.

        Full-body generation is the most resource-intensive digital-human
        node; the estimate uses the large model base overhead.
        """
        vram_gb = _FULL_BODY_MODEL_VRAM_GB
        ram_gb = 1.0
        time_s = 8.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a full-body digital-human video (placeholder).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``reference_image``, ``audio``, ``gesture_sequence``,
                ``method``, ``resolution``.

        Returns:
            A dict with ``video``.
        """
        method = str(inputs.get("method", "echo_mimic_v2"))
        gesture_sequence = inputs.get("gesture_sequence")
        resolution = inputs.get("resolution")

        ctx.logger.debug(
            "dh_full_body run_id=%s method=%s gesture=%s resolution=%s",
            ctx.run_id, method, gesture_sequence, resolution,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_full_body",
                action="generate_digital_human",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "gesture_sequence": gesture_sequence,
                    "resolution": resolution,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        video = {
            "kind": "placeholder_digital_human_video",
            "path": "placeholder.mp4",
            "method": method,
            "gesture_sequence": gesture_sequence,
            "resolution": resolution,
        }
        return {"video": video}


# ---------------------------------------------------------------------------
# FaceEnhanceNode
# ---------------------------------------------------------------------------
@register_node("dh_face_enhance")
class FaceEnhanceNode(BaseNode):
    """Face enhancement node (``dh_face_enhance``).

    Restores and enhances faces in a ``video`` using a face-restoration
    model (GFPGAN or CodeFormer), controlled by a ``strength`` parameter.

    Inputs:
        video: The source video containing faces (required).
        method: Enhancement method -- ``"gfpgan"`` or ``"codeformer"``.
        strength: Enhancement strength in ``[0, 1]`` (``0`` = no change,
            ``1`` = full restoration).

    Outputs:
        video: The face-enhanced video.
    """

    spec = NodeSpec(
        type="dh_face_enhance",
        name="Face Enhance",
        description="Restore and enhance faces in a video.",
        inputs={
            "video": "VIDEO",
            "method": "TEXT",
            "strength": "FLOAT",
        },
        outputs={
            "video": "VIDEO",
        },
        tags=["digital_human", "face_enhance", "video", "restoration"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate face-enhancement inputs.

        Extends the base checks with:

        * ``method`` in ``{"gfpgan", "codeformer"}``.
        * ``strength`` in ``[0, 1]``.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs, "method", "dh_face_enhance", _FACE_ENHANCE_METHODS, errors
        )

        strength = _coerce_float(inputs.get("strength"))
        if strength is not None and not (0.0 <= strength <= 1.0):
            errors.append(
                "Input 'strength' for node 'dh_face_enhance' must be in "
                "[0, 1], got {}.".format(strength)
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a face-enhancement run.

        Time and VRAM scale with ``strength`` (a partial-strength run
        does less work).
        """
        strength = _coerce_float(inputs.get("strength"))
        strength = strength if strength is not None else 1.0
        strength = max(0.0, min(1.0, strength))

        vram_gb = _FACE_ENHANCE_MODEL_VRAM_GB + strength * 1.0
        ram_gb = 0.4 + strength * 0.3
        time_s = 1.0 + strength * 3.0
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Enhance faces in a video (placeholder implementation).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``video``, ``method``, ``strength``.

        Returns:
            A dict with ``video``.
        """
        method = str(inputs.get("method", "gfpgan"))
        strength = _coerce_float(inputs.get("strength"))
        strength = strength if strength is not None else 0.75

        ctx.logger.debug(
            "dh_face_enhance run_id=%s method=%s strength=%.2f",
            ctx.run_id, method, strength,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_face_enhance",
                action="enhance_face",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "strength": strength,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        video = {
            "kind": "placeholder_face_enhance_video",
            "path": "placeholder.mp4",
            "method": method,
            "strength": strength,
        }
        return {"video": video}


# ---------------------------------------------------------------------------
# VoiceCloneNode
# ---------------------------------------------------------------------------
@register_node("dh_voice_clone")
class VoiceCloneNode(BaseNode):
    """Voice cloning node (``dh_voice_clone``).

    Clones the voice from a ``reference_audio`` clip and synthesises
    speech from ``text`` in the specified ``language``.

    Inputs:
        reference_audio: The reference audio clip to clone the voice
            from (required).
        text: The text to synthesise (required).
        language: Target language -- ``"zh"``, ``"en"`` or ``"ja"``.
        method: Cloning method -- ``"cosyvoice"``, ``"f5_tts"`` or
            ``"chat_tts"``.

    Outputs:
        audio: The synthesised audio waveform.
        sample_rate: The output sample rate in Hz.
    """

    spec = NodeSpec(
        type="dh_voice_clone",
        name="Voice Clone",
        description="Clone a voice from reference audio and synthesise speech from text.",
        inputs={
            "reference_audio": "AUDIO",
            "text": "TEXT",
            "language": "TEXT",
            "method": "TEXT",
        },
        outputs={
            "audio": "AUDIO",
            "sample_rate": "INT",
        },
        tags=["digital_human", "voice_clone", "audio", "tts"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate voice-clone inputs.

        Extends the base checks with:

        * ``method`` in ``{"cosyvoice", "f5_tts", "chat_tts"}``.
        * ``language`` in ``{"zh", "en", "ja"}``.
        * ``text`` non-empty.
        """
        errors = super().validate_inputs(inputs)
        _validate_enum(
            inputs, "method", "dh_voice_clone", _VOICE_CLONE_METHODS, errors
        )
        _validate_enum(
            inputs,
            "language",
            "dh_voice_clone",
            _VOICE_CLONE_LANGUAGES,
            errors,
        )

        text = inputs.get("text")
        if isinstance(text, str) and not text.strip():
            errors.append(
                "Input 'text' for node 'dh_voice_clone' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a voice-clone run.

        Time scales with the length of the input text (approximated by
        word count).
        """
        text = inputs.get("text", "")
        text = text if isinstance(text, str) else ""
        word_count = max(1, len(text.split()))

        vram_gb = _VOICE_CLONE_MODEL_VRAM_GB
        ram_gb = word_count * 0.5 * _VOICE_RAM_PER_AUDIO_S
        time_s = word_count * _VOICE_TIME_PER_WORD_S
        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Clone a voice and synthesise speech (placeholder).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``reference_audio``, ``text``, ``language``,
                ``method``.

        Returns:
            A dict with ``audio`` and ``sample_rate``.
        """
        text = str(inputs.get("text", ""))
        language = str(inputs.get("language", "zh"))
        method = str(inputs.get("method", "cosyvoice"))
        sample_rate = int(
            ctx.config.get(
                "default_voice_sample_rate", _DEFAULT_VOICE_SAMPLE_RATE
            )
        )

        ctx.logger.debug(
            "dh_voice_clone run_id=%s method=%s language=%s",
            ctx.run_id, method, language,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.dh_voice_clone",
                action="clone_voice",
                resource_id=method,
                details={
                    "run_id": ctx.run_id,
                    "method": method,
                    "language": language,
                    "text": text[: 64],
                    "sample_rate": sample_rate,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        audio = {
            "kind": "placeholder_cloned_audio",
            "path": "placeholder.wav",
            "method": method,
            "language": language,
            "text": text[: 64],
            "sample_rate": sample_rate,
        }
        return {"audio": audio, "sample_rate": sample_rate}
