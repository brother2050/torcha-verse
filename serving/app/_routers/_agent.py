"""Agent run router (v0.6.x).

One endpoint:

* ``POST /v1/agent/run`` -- execute an agent on a task with
  optional SSE streaming.

Backed by the ``agent_run`` L4 node -- a thin wrapper over the
default :class:`infrastructure.agent.AgentBus` (ReAct loop with
tool-calling).  The node returns the final answer, the per-step
transcript (``thought / action / observation``) and the number
of iterations; the response envelope mirrors that shape so
callers can introspect the agent's reasoning without a second
roundtrip.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from serving.models import AgentRequest
from serving.service import (
    PipelineService,
    _error_response,
    _estimate_tokens,
    _generate_id,
)

__all__ = ["build_router", "_agent_stream"]


def build_router(service: PipelineService) -> APIRouter:
    """Build the agent router bound to ``service``."""
    router = APIRouter()

    @router.post("/v1/agent/run")
    async def agent_run(request: AgentRequest) -> Any:
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

            result = service._run(
                "agent_run",
                "agent_run",
                "agent",
                {
                    "query": request.task,
                    "max_steps": int(request.max_steps),
                    "temperature": float(request.temperature),
                },
            )
            if "error" in result:
                raise RuntimeError(
                    f"{result['error']} [{result.get('error_type', 'engine_error')}]"
                )

            output_text = str(result.get("final_answer", ""))
            ok = bool(result.get("ok", False))
            steps = result.get("steps", [])
            iterations = int(result.get("iterations", 0))

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
                "steps": steps,
                "iterations": iterations,
                "ok": ok,
                "metadata": {
                    "agent_type": request.agent_type,
                    "max_steps": int(request.max_steps),
                },
            }
            service.metrics.record_request(endpoint, time.time() - start)
            return response_dict

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("Agent run failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    return router


def _agent_stream(
    svc: PipelineService,
    request: AgentRequest,
    endpoint: str,
) -> Iterator[str]:
    """Yield SSE frames for streaming agent execution.

    The agent is a one-shot tool-calling loop (the LLM emits
    ``final_answer`` once it has decided), so the streamed output
    is the final answer wrapped in an ``agent.step`` frame,
    followed by the terminal ``[DONE]`` marker -- preserving the
    SSE contract.
    """
    start = time.time()
    try:
        result = svc._run(
            "agent_run",
            "agent_run",
            "agent",
            {
                "query": request.task,
                "max_steps": int(request.max_steps),
                "temperature": float(request.temperature),
            },
        )
        if "error" in result:
            raise RuntimeError(result["error"])
        output_text = str(result.get("final_answer", ""))

        # Security Gate 3: filter the streamed text before yielding.
        try:
            filter_result = svc._filter.filter_text(output_text)
            if not filter_result.passed:
                yield f"data: {json.dumps({'error': 'Output filtered'})}\n\n"
                return
        except Exception as exc:
            svc._logger.debug("filter (SSE chunk) failed; passing through: %s", exc)

        data = {
            "id": _generate_id(),
            "object": "agent.step",
            "created": int(time.time()),
            "model": "agent",
            "step": {
                "output": output_text,
                "iterations": int(result.get("iterations", 0)),
                "ok": bool(result.get("ok", False)),
            },
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
        svc.metrics.record_request(endpoint, time.time() - start)
    except Exception as exc:
        svc.metrics.record_request(endpoint, time.time() - start, error=True)
        error_data = {"error": {"message": str(exc), "type": "stream_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
