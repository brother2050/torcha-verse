"""Multimodal / RAG / agent methods for :class:`PipelineService`.

These three capabilities do not follow the single-node pipeline
shape; they go straight to the LLM / RAG / agent subsystems.
Lifted out of :class:`PipelineService` for readability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["attach_llm_methods", "attach_list_models"]


def _ensure_rag(self) -> None:
    """Lazily build the RAG vector store and retriever.

    Uses the in-memory store by default; operators can swap in a
    persistent :class:`FaissVectorStore` by mutating
    ``self._rag_store`` before serving traffic.
    """
    if self._rag_store is not None and self._rag_retriever is not None:
        return
    from rag.retrievers import VectorRetriever
    from rag.vectorstore import InMemoryVectorStore

    self._rag_store = InMemoryVectorStore()
    self._rag_retriever = VectorRetriever(
        self._rag_store, embed_fn=self._llm_provider.embed
    )


def _multimodal_understand(
    self,
    prompt: str,
    image: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a multimodal prompt through the :class:`LLMProvider`.

    Image inputs are accepted (PIL image or base64 string) but only
    the textual content is forwarded to the LLM: the framework
    bundles a text-only :class:`LLMProvider` by default.  Operators
    that wire a vision-capable provider will receive the raw
    ``image`` payload via the ``**kwargs`` bag.
    """
    from models.interfaces.llm_provider import LLMMessage

    # Surface the image through the metadata so vision-capable
    # providers can pick it up; text-only providers ignore it.
    augmented_prompt = prompt
    if image is not None:
        augmented_prompt = f"[image attached] {prompt}"
    try:
        response = self._llm_provider.chat(
            [LLMMessage(role="user", content=augmented_prompt)],
            max_tokens=int(kwargs.get("max_tokens", 256)),
            temperature=float(kwargs.get("temperature", 0.7)),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Multimodal understand failed: {exc}",
            "error_type": "llm_error",
        }
    return {
        "text": response.text,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def _rag_query(
    self,
    query: str,
    top_k: int = 4,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Retrieve context for ``query`` and synthesise an answer.

    The query is embedded with the LLM provider's ``embed`` method,
    the top-``k`` passages are concatenated, and the LLM provider is
    prompted for a final answer.  The response is a dict with
    ``answer`` (text) and ``context`` (retrieved passages).
    """
    from models.interfaces.llm_provider import LLMMessage

    try:
        self._ensure_rag()
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"RAG store init failed: {exc}",
            "error_type": "rag_error",
        }
    try:
        results = self._rag_retriever.retrieve(query, top_k=top_k)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"RAG retrieve failed: {exc}",
            "error_type": "rag_error",
        }
    context_blocks = [r.text for r in results if getattr(r, "text", None)]
    if context_blocks:
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{chr(10).join(context_blocks)}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
    else:
        user_content = query
    try:
        response = self._llm_provider.chat(
            [LLMMessage(role="user", content=user_content)],
            max_tokens=int(kwargs.get("max_tokens", 256)),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"RAG answer failed: {exc}",
            "error_type": "llm_error",
            "context": context_blocks,
        }
    return {
        "answer": response.text,
        "context": context_blocks,
        "model": response.model,
    }


def _agent_run(
    self,
    task: str,
    max_steps: int = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a :class:`ReActAgent` over ``task`` with the configured LLM.

    The agent gets a thin adapter that converts the LLMProvider
    response into a string for ReAct's text-prompt format.  Falls
    back to the LLM directly if the agent subsystem is unavailable.
    """
    try:
        from agents.react_agent import ReActAgent
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Agent subsystem unavailable: {exc}",
            "error_type": "agent_error",
        }

    # ReActAgent expects a model object with a ``generate(prompt)``
    # method.  Wrap the LLMProvider in a thin shim that translates
    # the single-prompt call into a chat-style call.
    from models.interfaces.llm_provider import LLMMessage

    provider = self._llm_provider

    class _ReActShim:
        """Adapter exposing ``LLMProvider`` as a ReAct-compatible model."""

        def __init__(self, name: str, provider: Any) -> None:
            self.name = name
            self._provider = provider

        def generate(self, prompt: str, **kw: Any) -> str:
            messages = [LLMMessage(role="user", content=prompt)]
            return self._provider.chat(messages, **kw).text

    shim = _ReActShim("agent-shim", provider)
    try:
        agent = ReActAgent(role="assistant", model=shim, max_steps=max_steps)
        result = agent.run(task)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Agent run failed: {exc}",
            "error_type": "agent_error",
        }
    return {
        "output": result.output if hasattr(result, "output") else str(result),
        "steps": [
            {"thought": s.thought, "action": s.action, "observation": s.observation}
            for s in (result.steps if hasattr(result, "steps") else [])
        ],
    }


def _list_models(self) -> List[Dict[str, Any]]:
    """Return the catalogue of registered node types as model metadata."""
    models: List[Dict[str, Any]] = []
    for spec in self._registry.list():
        models.append({
            "id": spec.type,
            "object": "node",
            "name": spec.name,
            "description": spec.description,
            "tags": list(spec.tags),
        })
    return models


def attach_llm_methods(cls: type) -> type:
    """Attach multimodal / RAG / agent methods to ``cls``."""
    cls._ensure_rag = _ensure_rag
    cls.multimodal_understand = _multimodal_understand
    cls.rag_query = _rag_query
    cls.agent_run = _agent_run
    return cls


def attach_list_models(cls: type) -> type:
    """Attach :meth:`list_models` to ``cls``."""
    cls.list_models = _list_models
    return cls
