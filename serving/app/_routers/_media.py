"""Image / audio / video routers (v0.6.x).

Three endpoints:

* ``POST /v1/images/generate`` -- text-to-image generation.
* ``POST /v1/audio/synthesize`` -- text-to-speech.
* ``POST /v1/videos/generate`` -- text-to-video generation.

The shape mirrors the text/chat routers: input sanitisation, an
optional output filter for the (rare) text payloads, and a
:class:`UnifiedResponse` envelope around the media payload.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from serving.models import AudioRequest, ImageRequest, UnifiedResponse, VideoRequest
from serving.models import Choice, Usage
from serving.service import (
    PipelineService,
    _error_response,
    _estimate_tokens,
    _generate_id,
    _media_payload,
)

__all__ = ["build_router"]


def build_router(service: PipelineService) -> APIRouter:
    """Build the media router bound to ``service``."""
    router = APIRouter()

    @router.post("/v1/images/generate")
    async def images_generate(request: ImageRequest) -> Any:
        """Generate an image from a text prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "images_generate"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                if request.negative_prompt:
                    request.negative_prompt = service._sanitizer.sanitize_text(
                        request.negative_prompt
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.image_txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            image = result.get("image", result)

            # Security Gate 3: filter the generated image output.
            try:
                filter_result = service._filter.filter_image(image)
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

            payload = _media_payload(image, "image/png")
            response = UnifiedResponse(
                id=_generate_id(),
                object="image",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.prompt) + 1,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Image generation failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    @router.post("/v1/audio/synthesize")
    async def audio_synthesize(request: AudioRequest) -> Any:
        """Synthesize speech from text."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "audio_synthesize"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.text = service._sanitizer.sanitize_text(request.text)
                request.model = service._sanitizer.sanitize_text(request.model)
                request.emotion = service._sanitizer.sanitize_text(request.emotion)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.audio_tts(
                text=request.text,
                voice=request.speaker_id,
                speed=request.speed,
                emotion=request.emotion,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            # Security Gate 3: filter the output text content rather than
            # the input.  The audio node currently returns pure media data
            # (no text transcript), so there is nothing to filter.  If a
            # text transcript/caption is added to the output in the future,
            # it should be passed through service._filter.filter_text()
            # before release.
            output_text = str(result.get("text", ""))
            if output_text:
                try:
                    filter_result = service._filter.filter_text(output_text)
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

            audio = result.get("audio", result)
            payload = _media_payload(audio, "audio/wav")
            response = UnifiedResponse(
                id=_generate_id(),
                object="audio",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.text),
                    completion_tokens=1,
                    total_tokens=_estimate_tokens(request.text) + 1,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Audio synthesis failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    @router.post("/v1/videos/generate")
    async def videos_generate(request: VideoRequest) -> Any:
        """Generate a video from a text prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "videos_generate"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                if request.negative_prompt:
                    request.negative_prompt = service._sanitizer.sanitize_text(
                        request.negative_prompt
                    )
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.video_txt2vid(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                model=request.model,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            video = result.get("video", result)

            # Security Gate 3: filter any text description that may accompany
            # the video.  Currently the video node returns pure media data
            # (no text caption), so there is nothing to filter.  If a text
            # description is added to the output in the future, it should
            # be passed through service._filter.filter_text() before
            # release.
            output_text = str(result.get("text", ""))
            if output_text:
                try:
                    filter_result = service._filter.filter_text(output_text)
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

            payload = _media_payload(video, "image/gif")
            response = UnifiedResponse(
                id=_generate_id(),
                object="video",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(index=0, text=payload, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=_estimate_tokens(request.prompt),
                    completion_tokens=request.num_frames,
                    total_tokens=_estimate_tokens(request.prompt) + request.num_frames,
                ),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Video generation failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    return router
