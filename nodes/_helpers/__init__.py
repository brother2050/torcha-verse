"""Shared helper functions for node modules (v0.6.x).

The v0.4.x ``nodes/_helpers.py`` was a single 710-line file
mixing four distinct responsibilities: numeric / ref coercion,
no-model echo backends, the default model backend registry
(``register_default_*`` / ``call_*_backend``) and RAG document
normalisation.

In v0.6.x we split that file into focused sub-modules:

* :mod:`._coerce`   -- :func:`coerce_dim`, :func:`coerce_int`,
  :func:`coerce_float`, :func:`ref_id`.
* :mod:`._echo`     -- ``_*_echo_factory`` (text / image / video
  / audio / multimodal).
* :mod:`._backends` -- default backend registry, ``register_*``,
  ``call_*_backend``, ``reset_default_backends``.
* :mod:`._rag`      -- RAG index name pattern, RAG defaults,
  :func:`_normalise_rag_documents`.

This ``__init__`` re-exports the public surface so existing
callers (``from nodes._helpers import register_default_image_backend``
and friends) keep working unchanged.
"""

from __future__ import annotations

from ._backends import (
    call_audio_backend,
    call_consistency_score_backend,
    call_depth_backend,
    call_diffusion_loop_backend,
    call_diffusion_scheduler_backend,
    call_digital_human_backend,
    call_face_enhance_backend,
    call_frame_interpolation_backend,
    call_full_body_backend,
    call_image_backend,
    call_inpaint_backend,
    call_lipsync_backend,
    call_motion_module_backend,
    call_multimodal_backend,
    call_music_backend,
    call_portrait_anim_backend,
    call_super_resolution_backend,
    call_talking_head_backend,
    call_text_backend,
    call_tts_backend,
    call_video_backend,
    call_video_stitch_backend,
    register_default_audio_backend,
    register_default_image_backend,
    register_default_multimodal_backend,
    register_default_text_backend,
    register_default_video_backend,
    reset_default_backends,
)
from ._coerce import (
    _MEGAPIXEL_PIXELS,
    coerce_dim,
    coerce_float,
    coerce_int,
    ref_id,
)
from ._latent import (
    LatentStats,
    LatentValidationError,
    LatentValidator,
    quick_validate,
    validate_range,
    validate_shape,
)
from ._rag import (
    _RAG_DEFAULT_BACKEND,
    _RAG_DEFAULT_CHUNK_OVERLAP,
    _RAG_DEFAULT_CHUNK_SIZE,
    _RAG_DEFAULT_TOP_K,
    _RAG_INDEX_NAME_PATTERN,
    _normalise_rag_documents,
)

__all__ = [
    # Coercion / refs
    "_MEGAPIXEL_PIXELS",
    "coerce_dim",
    "coerce_int",
    "coerce_float",
    "ref_id",
    # Backend registry
    "register_default_text_backend",
    "register_default_image_backend",
    "register_default_video_backend",
    "register_default_audio_backend",
    "register_default_multimodal_backend",
    "call_text_backend",
    "call_image_backend",
    "call_video_backend",
    "call_audio_backend",
    "call_multimodal_backend",
    "call_digital_human_backend",
    "call_lipsync_backend",
    "call_talking_head_backend",
    "call_portrait_anim_backend",
    "call_full_body_backend",
    "call_face_enhance_backend",
    "call_tts_backend",
    "call_depth_backend",
    "call_consistency_score_backend",
    "call_frame_interpolation_backend",
    "call_motion_module_backend",
    "call_diffusion_scheduler_backend",
    "call_diffusion_loop_backend",
    "call_super_resolution_backend",
    "call_inpaint_backend",
    "call_music_backend",
    "call_video_stitch_backend",
    "reset_default_backends",
    # RAG
    "_RAG_INDEX_NAME_PATTERN",
    "_RAG_DEFAULT_TOP_K",
    "_RAG_DEFAULT_CHUNK_SIZE",
    "_RAG_DEFAULT_CHUNK_OVERLAP",
    "_RAG_DEFAULT_BACKEND",
    "_normalise_rag_documents",
    # Latent validation (v0.8.5)
    "LatentStats",
    "LatentValidationError",
    "LatentValidator",
    "quick_validate",
    "validate_range",
    "validate_shape",
]
