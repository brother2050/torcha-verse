"""Multimodal understanding router (v0.6.x).

One endpoint:

* ``POST /v1/multimodal/understand`` -- answer a question
  about a (text | image | image+audio) payload.

Backed by two L4 nodes selected on the request shape:

* ``image_understand`` when the request carries a base64 image
  (an optional audio attachment is forwarded alongside it as
  an "audio" modality key).
* ``text_chat`` for the text-only path (the framework
  :class:`LocalTorchTextProvider` handles the byte-level
  language model).

The selected node is executed through
:meth:`PipelineService._run` so the request flows through
the same security gates (input sanitisation, prompt-injection
detection, output filter) as the other generation endpoints.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

from serving.models import MultimodalRequest
from serving.service import (
    PipelineService,
    _decode_b64_audio,
    _decode_b64_image,
    _error_response,
    _estimate_tokens,
    _make_response,
)

__all__ = ["build_router"]


def build_router(service: PipelineService) -> APIRouter:
    """Build the multimodal router bound to ``service``."""
    router = APIRouter()

    @router.post("/v1/multimodal/understand")
    async def multimodal_understand(request: MultimodalRequest) -> Any:
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "multimodal_understand"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                if request.text:
                    request.text = service._sanitizer.sanitize_text(request.text)
                if request.question:
                    request.question = service._sanitizer.sanitize_text(
                        request.question
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts on
            # the textual payload.
            text_for_injection_check = (request.text or "") + " " + (request.question or "")
            text_for_injection_check = text_for_injection_check.strip()
            if text_for_injection_check:
                injection_result = service._sanitizer.detect_prompt_injection(
                    text_for_injection_check
                )
                if injection_result.is_injected:
                    return _error_response(
                        "Prompt injection detected", error_type="injection", code=400
                    )

            has_image = bool(request.image)
            has_audio = bool(request.audio)
            question = (request.question or request.text or "").strip()
            if not question:
                question = "Describe the input in detail."

            if has_image or has_audio:
                # Multimodal path: image (and optional audio) +
                # question are forwarded to the image_understand L4
                # node, which dispatches to the multimodal provider.
                try:
                    if has_image:
                        image_obj = _decode_b64_image(request.image)
                    else:
                        image_obj = None
                    if has_audio:
                        try:
                            audio_obj = _decode_b64_audio(request.audio)
                        except Exception:  # noqa: BLE001
                            # Fall back to raw bytes if the audio is
                            # not a WAV -- the multimodal provider
                            # handles str inputs transparently.
                            audio_obj = request.audio
                except ValueError as exc:
                    return _error_response(
                        "Input rejected: " + str(exc), code=400
                    )

                node_inputs: Dict[str, Any] = {
                    "image": image_obj,
                    "question": question,
                    "max_new_tokens": int(request.max_tokens or 128),
                }
                if has_audio:
                    # image_understand only consumes `image`; we
                    # forward the audio as metadata so the multimodal
                    # provider can decide whether to consume it.
                    node_inputs["_audio"] = audio_obj
                result = service._run(
                    "multimodal_understand",
                    "image_understand",
                    "img",
                    node_inputs,
                    config={"default_multimodal_model": request.model},
                )
            else:
                # Text-only path: forward to the text_chat L4 node.
                result = service._run(
                    "multimodal_understand",
                    "text_chat",
                    "chat",
                    {
                        "prompt": question,
                        "model": request.model,
                        "max_tokens": int(request.max_tokens or 256),
                    },
                    config={"default_text_model": request.model},
                )

            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            text = str(result.get("text", ""))

            # Security Gate 3: filter the text response.
            try:
                filter_result = service._filter.filter_text(text)
                if not filter_result.passed:
                    return _error_response(
                        "Output filtered: " + filter_result.action,
                        error_type="output_filtered",
                        code=403,
                    )
            except Exception as filter_exc:  # noqa: BLE001
                service._logger.warning(
                    "Output filter failed, allowing response: %s", filter_exc
                )

            response = _make_response(
                model=request.model,
                text=text,
                object_type="multimodal.understanding",
                prompt_tokens=_estimate_tokens(
                    (request.text or "") + (request.question or "")
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Multimodal understanding failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    return router
