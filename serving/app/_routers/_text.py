"""Text / chat routers (v0.6.x).

Two endpoints:

* ``POST /v1/text/completions`` -- single-prompt text generation
  with optional SSE streaming.
* ``POST /v1/text/chat`` -- multi-turn chat with optional SSE
  streaming.

Both endpoints follow the same contract:

* Security Gate 1 (input sanitisation) is applied first; ``ValueError``
  from the sanitizer is converted to a 400 with the reason.
* Security Gate 1b (prompt-injection detection) is applied next;
  a positive result is a 400.
* Streaming requests go through :class:`StreamingResponse`; the
  per-frame yield functions live in this module too.
* Security Gate 3 (output filter) is applied to the final text;
  on a hit the response is a 403 ``output_filtered`` error.
* All other exceptions are caught and returned as a 500
  ``engine_error``; the metrics collector records the call.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from serving.models import ChatRequest, TextCompletionRequest
from serving.service import (
    PipelineService,
    _error_response,
    _estimate_tokens,
    _generate_id,
    _make_response,
    _messages_to_prompt,
)

__all__ = ["build_router", "_text_completion_stream", "_chat_stream"]


def build_router(service: PipelineService) -> APIRouter:
    """Build the text / chat router bound to ``service``.

    The router is created *without* a prefix so the existing
    ``/v1/text/completions`` and ``/v1/text/chat`` paths are
    preserved.
    """
    router = APIRouter()

    @router.post("/v1/text/completions")
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

    @router.post("/v1/text/chat")
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

    return router


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
        except Exception as exc:
            svc._logger.debug("filter (SSE chunk) failed; passing through: %s", exc)

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
        except Exception as exc:
            svc._logger.debug("filter (SSE chunk) failed; passing through: %s", exc)

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
