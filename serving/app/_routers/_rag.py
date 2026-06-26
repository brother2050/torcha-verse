"""RAG query router (v0.6.x).

One endpoint:

* ``POST /v1/rag/query`` -- answer a question using
  retrieval-augmented generation.

Backed by two L4 nodes:

* ``rag_query`` performs the embedding + top-k retrieval from
  the named index and returns the hits + assembled context
  block.
* ``text_chat`` synthesises the final answer (when
  ``request.synthesize`` is True -- the default) by feeding
  the context to the LLM provider.

The two nodes are executed through
:meth:`PipelineService._run` so the request flows through
the same security gates as the other generation endpoints.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from serving.models import RAGRequest
from serving.service import (
    PipelineService,
    _error_response,
    _estimate_tokens,
    _make_response,
)

__all__ = ["build_router"]


def build_router(service: PipelineService) -> APIRouter:
    """Build the RAG router bound to ``service``."""
    router = APIRouter()

    @router.post("/v1/rag/query")
    async def rag_query(request: RAGRequest) -> Any:
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
                request.index_name = service._sanitizer.sanitize_text(
                    request.index_name
                )
            except ValueError as exc:
                return _error_response("Input rejected: " + str(exc), code=400)

            # Security Gate 1b: detect prompt-injection attempts on
            # the question.
            injection_result = service._sanitizer.detect_prompt_injection(
                request.question
            )
            if injection_result.is_injected:
                return _error_response(
                    "Prompt injection detected", error_type="injection", code=400
                )

            # Step 1: run the rag_query L4 node to retrieve top-k
            # chunks from the named index.
            retrieval = service._run(
                "rag_query_retrieve",
                "rag_query",
                "retrieval",
                {
                    "index_name": request.index_name,
                    "query": request.question,
                    "top_k": int(request.top_k),
                },
            )
            if "error" in retrieval:
                raise RuntimeError(
                    f"{retrieval['error']} [{retrieval.get('error_type', 'engine_error')}]"
                )

            hits = retrieval.get("hits", [])
            context = retrieval.get("context", "")

            if not request.synthesize:
                # Caller asked for raw retrieval output only.
                response = _make_response(
                    model="rag",
                    text=context or "(no context retrieved)",
                    object_type="rag.retrieval",
                    prompt_tokens=_estimate_tokens(request.question),
                )
                # Surface the raw hits list alongside the response so
                # downstream consumers don't need a second roundtrip.
                response["hits"] = hits
                response["index_name"] = request.index_name
                service.metrics.record_request(endpoint, time.time() - start)
                return response

            # Step 2: synthesise the final answer via the LLM.
            if context:
                user_prompt = (
                    "Use the following context to answer the question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {request.question}\n\nAnswer:"
                )
            else:
                user_prompt = (
                    f"Question: {request.question}\n\nAnswer:"
                )
            answer_result = service._run(
                "rag_query_synthesise",
                "text_chat",
                "answer",
                {
                    "prompt": user_prompt,
                    "model": "default",
                    "max_tokens": int(request.max_tokens or 256),
                },
            )
            if "error" in answer_result:
                raise RuntimeError(
                    f"{answer_result['error']} [{answer_result.get('error_type', 'engine_error')}]"
                )

            text = str(answer_result.get("text", ""))

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
            response["hits"] = hits
            response["index_name"] = request.index_name
            response["context"] = context
            service.metrics.record_request(endpoint, time.time() - start)
            return response

        except Exception as exc:
            service.metrics.record_request(endpoint, time.time() - start, error=True)
            service._logger.error("RAG query failed: %s", exc)
            error_type = "not_implemented" if "not_implemented" in str(exc).lower() else "engine_error"
            code = 501 if error_type == "not_implemented" else 500
            return _error_response("Internal error", error_type=error_type, code=code)

    return router
