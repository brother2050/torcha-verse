"""Service layer that bridges the TorchaVerse serving API to the
node / pipeline system (v0.6.x).

This sub-package hosts the :class:`PipelineService` together with
the request/response helpers used by the FastAPI routers in
:mod:`serving.app._routers` and the Click commands in
:mod:`serving.cli`.

The original ``service.py`` (848 lines) was split into focused
sub-modules:

* :mod:`._service` -- :class:`PipelineService` constructor, the
  executor-bridge (``_make_executor`` / ``_run``) and the
  property accessors.  The capability methods are attached at
  import time via the ``attach_*_methods`` helpers below.
* :mod:`._service_text` -- :meth:`text_completion` /
  :meth:`text_chat`.
* :mod:`._service_media` -- :meth:`image_txt2img` /
  :meth:`image_img2img` / :meth:`audio_tts` /
  :meth:`video_txt2vid`.
* :mod:`._service_llm` -- :meth:`multimodal_understand` /
  :meth:`rag_query` / :meth:`agent_run` and the
  :meth:`list_models` introspection helper.
* :mod:`._ids` -- :func:`_generate_id` / :func:`_estimate_tokens`
  / :func:`_messages_to_prompt`.
* :mod:`._response` -- :func:`_make_response` /
  :func:`_error_response`.
* :mod:`._media` -- :func:`_image_to_b64` /
  :func:`_audio_to_b64` / :func:`_video_to_b64` /
  :func:`_media_payload` / :func:`_decode_b64_image` /
  :func:`_decode_b64_audio`.

The "attach at import time" pattern is the v0.4.x "duck-punching"
idiom for splitting a class across modules without breaking
``monkeypatch.setattr(service, "text_chat", ...)`` or
``PipelineService.text_chat = my_override`` -- tests that
override individual capability methods keep working.

Public surface (preserved from v0.4.x / v0.5.x):

* :class:`PipelineService`
* :func:`_generate_id` / :func:`_estimate_tokens` /
  :func:`_messages_to_prompt` / :func:`_make_response` /
  :func:`_error_response` / :func:`_image_to_b64` /
  :func:`_audio_to_b64` / :func:`_video_to_b64` /
  :func:`_media_payload` / :func:`_decode_b64_image` /
  :func:`_decode_b64_audio`

The string import path ``"serving.service"`` (used by the CLI,
the FastAPI factory, the Web UI and the test suite) keeps
working -- it now resolves to this sub-package.
"""

from __future__ import annotations

# Re-export the public API at the sub-package level so that
# ``from serving.service import PipelineService`` and
# ``from serving.service import _error_response`` and friends
# all work the same as they did in v0.4.x / v0.5.x.
from ._ids import _estimate_tokens, _generate_id, _messages_to_prompt
from ._media import (
    _audio_to_b64,
    _decode_b64_audio,
    _decode_b64_image,
    _image_to_b64,
    _media_payload,
    _video_to_b64,
)
from ._response import _error_response, _make_response
from ._service import PipelineService

__all__ = [
    "PipelineService",
    # Id / token / prompt helpers
    "_generate_id",
    "_estimate_tokens",
    "_messages_to_prompt",
    # Response builders
    "_make_response",
    "_error_response",
    # Media (de)serialisation
    "_image_to_b64",
    "_audio_to_b64",
    "_video_to_b64",
    "_media_payload",
    "_decode_b64_image",
    "_decode_b64_audio",
]
