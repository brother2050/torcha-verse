"""FastAPI application factory for the TorchaVerse inference API.

This module exposes :func:`create_app` -- the public entry point that
builds and configures the :class:`FastAPI` application together with all
route handlers (text, image, audio, video, multimodal, RAG, agent) and
the Server-Sent Events streaming generators.

It was extracted from the original monolithic ``api_server.py``.  The
request/response models live in :mod:`serving.models`, the
:class:`PipelineService` and helpers live in :mod:`serving.service`, and
the :class:`MetricsCollector` lives in :mod:`serving.metrics`.

Public surface:

* :func:`create_app` -- build a configured :class:`FastAPI` instance.
* :func:`get_app` -- return the singleton app (lazily created).
* :func:`main` -- CLI entry point launching the Uvicorn ASGI server.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterator, Optional

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

from serving.metrics import MetricsCollector  # noqa: F401 -- re-exported surface
from serving.models import (
    AgentRequest,
    AudioRequest,
    ChatRequest,
    Choice,
    ImageRequest,
    MultimodalRequest,
    RAGRequest,
    TextCompletionRequest,
    UnifiedResponse,
    Usage,
    VideoRequest,
)
from serving.service import (
    PipelineService,
    _error_response,
    _estimate_tokens,
    _generate_id,
    _make_response,
    _media_payload,
    _messages_to_prompt,
)

try:  # FastAPI / Pydantic are declared in requirements.txt but guarded
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastAPI and Pydantic are required for the serving API. "
        "Install them with: pip install fastapi uvicorn pydantic"
    ) from _exc

__all__ = ["create_app", "get_app", "main"]


# ===========================================================================
# Application factory
# ===========================================================================
def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured :class:`FastAPI` instance with all routes
        registered.
    """
    app = FastAPI(
        title="TorchaVerse Inference API",
        description=(
            "Unified inference API for text, image, audio, video, "
            "multimodal, RAG, and agent capabilities."
        ),
        version="0.3.1",
    )

    # CORS middleware.  Origins are read from the TORCHA_CORS_ORIGINS
    # environment variable (comma-separated).  The default ``"*"`` is
    # permissive and intended for development only -- in production,
    # configure specific origins (e.g. ``https://app.example.com``).
    # ``allow_credentials`` is intentionally omitted: it is incompatible
    # with the wildcard ``allow_origins=["*"]`` and would be silently
    # dropped (or rejected) by the browser.
    cors_origins = os.environ.get("TORCHA_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = PipelineService()

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        service._logger.error("Unhandled exception: %s", exc, exc_info=True)
        # Never leak the raw exception text to the client in production;
        # return a generic message instead.
        return _error_response(
            "Internal Server Error", error_type="internal_error", code=500
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint."""
        device_info = DeviceManager().get_device_info()
        return {
            "status": "healthy",
            "version": "0.3.1",
            "device": device_info.get("device", "cpu"),
            "uptime": time.time() - service.metrics._start_time,
            "node_types": len(service.list_models()),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics() -> str:
        """Prometheus-format metrics endpoint."""
        return service.metrics.render()

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List all registered node types."""
        models = service.list_models()
        return {
            "object": "list",
            "data": models,
        }

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------
    @app.post("/v1/text/completions")
    async def text_completions(request: TextCompletionRequest) -> Any:
        """Generate text from a prompt."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "text_completions"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.prompt = service._sanitizer.sanitize_text(request.prompt)
                request.model = service._sanitizer.sanitize_text(request.model)
                if request.stop:
                    if isinstance(request.stop, str):
                        request.stop = service._sanitizer.sanitize_text(request.stop)
                    elif isinstance(request.stop, list):
                        request.stop = [
                            service._sanitizer.sanitize_text(s) for s in request.stop
                        ]
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(
                request.prompt
            )
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _text_completion_stream(service, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.text_completion(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            text = result.get("text", "")

            # Security Gate 3: filter the generated text output.
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

            prompt_tokens = _estimate_tokens(request.prompt)
            response = _make_response(
                model=request.model,
                text=text,
                object_type="text_completion",
                prompt_tokens=prompt_tokens,
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Text completion failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    def _text_completion_stream(
        svc: PipelineService,
        request: TextCompletionRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming text completion.

        The node system returns a complete generation in one shot, so the
        full text is emitted as a single chunk followed by the terminal
        ``[DONE]`` marker -- preserving the SSE contract.
        """
        start = time.time()
        try:
            result = svc.text_completion(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            text = result.get("text", "")

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "text": text, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            # Final frame.
            done_data = {
                "id": _generate_id(),
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "text": "", "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    @app.post("/v1/text/chat")
    async def text_chat(request: ChatRequest) -> Any:
        """Run a multi-turn chat conversation."""
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "text_chat"
        try:
            # Security Gate 1: sanitise every message's text content.
            try:
                for msg in request.messages:
                    msg.content = service._sanitizer.sanitize_text(msg.content)
                request.model = service._sanitizer.sanitize_text(request.model)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            prompt = _messages_to_prompt(request.messages)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(prompt)
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _chat_stream(service, prompt, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.text_chat(
                prompt=prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            text = result.get("text", "")

            # Security Gate 3: filter the generated text output.
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

            prompt_tokens = sum(_estimate_tokens(m.content) for m in request.messages)
            response = _make_response(
                model=request.model,
                text=text,
                object_type="chat.completion",
                prompt_tokens=prompt_tokens,
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Chat failed: %s", exc)
            return _error_response("Internal error", error_type="engine_error", code=500)

    def _chat_stream(
        svc: PipelineService,
        prompt: str,
        request: ChatRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming chat."""
        start = time.time()
        try:
            result = svc.text_chat(
                prompt=prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            text = result.get("text", "")

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            done_data = {
                "id": _generate_id(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    @app.post("/v1/images/generate")
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

    # ------------------------------------------------------------------
    # Audio synthesis
    # ------------------------------------------------------------------
    @app.post("/v1/audio/synthesize")
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

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------
    @app.post("/v1/videos/generate")
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
            # description is added to the output in the future, it should be
            # passed through service._filter.filter_text() before release.
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

    # ------------------------------------------------------------------
    # Multimodal understanding
    # ------------------------------------------------------------------
    @app.post("/v1/multimodal/understand")
    async def multimodal_understand(request: MultimodalRequest) -> Any:
        """Understand multi-modal input and optionally answer a question.

        Multimodal understanding is not yet backed by a node; a
        ``not_implemented`` error is returned while the REST contract is
        preserved.
        """
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

            result = service.multimodal_understand()
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

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------
    @app.post("/v1/rag/query")
    async def rag_query(request: RAGRequest) -> Any:
        """Answer a question using retrieval-augmented generation.

        RAG query is not yet backed by a node; a ``not_implemented``
        error is returned while the REST contract is preserved.
        """
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "rag_query"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.question = service._sanitizer.sanitize_text(request.question)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            result = service.rag_query()
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
                model="rag",
                text=text,
                object_type="rag.answer",
                prompt_tokens=_estimate_tokens(request.question),
            )
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("RAG query failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    # ------------------------------------------------------------------
    # Agent run
    # ------------------------------------------------------------------
    @app.post("/v1/agent/run")
    async def agent_run(request: AgentRequest) -> Any:
        """Execute an agent on a task.

        Agent execution is not yet backed by a node; a
        ``not_implemented`` error is returned while the REST contract is
        preserved.
        """
        # Rate limiting: reject early when the token bucket is empty.
        if not service._rate_limiter.try_acquire():
            return _error_response(
                "Rate limit exceeded", error_type="rate_limit", code=429
            )
        start = time.time()
        endpoint = "agent_run"
        try:
            # Security Gate 1: sanitise user-supplied text input.
            try:
                request.task = service._sanitizer.sanitize_text(request.task)
                request.agent_type = service._sanitizer.sanitize_text(request.agent_type)
                if request.flow:
                    request.flow = service._sanitizer.sanitize_text(request.flow)
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts.
            injection_result = service._sanitizer.detect_prompt_injection(
                request.task
            )
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            if request.stream:
                return StreamingResponse(
                    _agent_stream(service, request, endpoint),
                    media_type="text/event-stream",
                )

            result = service.agent_run()
            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            output_text = str(result.get("output", ""))

            # Security Gate 3: filter the final text response.
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

            response_dict = {
                "id": _generate_id(),
                "object": "agent.result",
                "created": int(time.time()),
                "model": "agent",
                "choices": [
                    {"index": 0, "text": output_text, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": _estimate_tokens(request.task),
                    "completion_tokens": 0,
                    "total_tokens": _estimate_tokens(request.task),
                },
                "steps": [],
                "metadata": {},
            }
            service.metrics.record_request(endpoint, time.time() - start)
            return response_dict

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Agent run failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    def _agent_stream(
        svc: PipelineService,
        request: AgentRequest,
        endpoint: str,
    ) -> Iterator[str]:
        """Yield SSE frames for streaming agent execution."""
        start = time.time()
        try:
            result = svc.agent_run()
            if "error" in result:
                raise RuntimeError(result["error"])
            output_text = str(result.get("output", ""))

            # Security Gate 3: filter the streamed text before yielding.
            try:
                filter_result = svc._filter.filter_text(output_text)
                if not filter_result.passed:
                    yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                    return
            except Exception:
                pass  # filter errors should not block the stream

            data = {
                "id": _generate_id(),
                "object": "agent.step",
                "created": int(time.time()),
                "model": "agent",
                "step": {"output": output_text},
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            svc.metrics.record_request(endpoint, time.time() - start)
        except Exception as exc:
            svc.metrics.record_request(endpoint, time.time() - start, error=True)
            error_data = {"error": {"message": str(exc), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    return app


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """Entry point for the TorchaVerse API server.

    Parses ``--host`` and ``--port`` arguments and launches the
    Uvicorn ASGI server.
    """
    parser = argparse.ArgumentParser(
        description="TorchaVerse Inference API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    args = parser.parse_args()

    logger = get_logger("api_server")
    logger.info("Starting TorchaVerse API on %s:%d", args.host, args.port)

    try:
        import uvicorn

        uvicorn.run(
            "serving.api_server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install it with: pip install uvicorn"
        )
        raise


# Lazy app creation to avoid import side-effects (e.g. binding ports,
# loading config at import time).  Use ``get_app()`` to obtain the
# singleton, or reference ``serving.api_server:create_app`` with
# ``factory=True`` in uvicorn.
app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Return the singleton :class:`FastAPI` app, creating it on first call."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    main()
