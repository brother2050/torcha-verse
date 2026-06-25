"""L4 nodes for the RAG (Retrieval-Augmented Generation) stack.

Six nodes are exposed, all registered through the
:func:`register_node` decorator so the runtime can resolve them
by name:

* :class:`RAGIngestNode` (``rag_ingest``) -- chunk + embed +
  upsert a batch of documents into a named
  :class:`infrastructure.rag.RAGIndex`.
* :class:`RAGQueryNode` (``rag_query``) -- embed a query and
  return the top-k matching chunks with their provenance and
  the assembled context block.
* :class:`RAGDeleteNode` (``rag_delete``) -- drop a single
  document or an entire index.
* :class:`RAGListIndexesNode` (``rag_list_indexes``) -- return
  the names of all indexes known to the
  :class:`infrastructure.rag.RAGIndexStore`.
* :class:`RAGGetIndexNode` (``rag_get_index``) -- return
  metadata (size / dim / doc count) for a single index.
* :class:`RAGSearchTextNode` (``rag_search_text``) -- full-text
  keyword search over the indexed ``text_preview`` metadata;
  useful when the user knows the document id but not the
  chunk-level position.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseNode, NodeContext, NodeSpec, register_node
from ._helpers import (
    _RAG_INDEX_NAME_PATTERN,
    _normalise_rag_documents,
    _RAG_DEFAULT_BACKEND,
    _RAG_DEFAULT_TOP_K,
    _RAG_DEFAULT_CHUNK_SIZE,
    _RAG_DEFAULT_CHUNK_OVERLAP,
)

__all__ = [
    "RAGIngestNode",
    "RAGQueryNode",
    "RAGDeleteNode",
    "RAGListIndexesNode",
    "RAGGetIndexNode",
    "RAGSearchTextNode",
]


@register_node("rag_ingest")
class RAGIngestNode(BaseNode):
    """Ingest a batch of documents into a named RAG index.

    Inputs:
        ``index_name`` (str, required): The index to ingest into.
        ``documents`` (list[dict], required): Each entry must have
            ``"doc_id"`` and ``"text"``; ``"metadata"`` is
            optional.  ``doc_id`` may be omitted and one will be
            generated.
        ``chunk_size`` (int, optional): Characters per chunk.
        ``chunk_overlap`` (int, optional): Overlap between chunks.

    Returns:
        A dict with ``"index_name"``, ``"documents"`` and
        ``"vectors"`` (number of chunks embedded + inserted).
    """

    spec: NodeSpec = NodeSpec(
        type="rag_ingest",
        name="RAG Ingest",
        description="Chunk + embed + upsert a batch of documents into a named RAG index.",
        inputs={
            "index_name": "TEXT",
            "documents": "Optional[JSON]",
            "chunk_size": "Optional[INT]",
            "chunk_overlap": "Optional[INT]",
            "backend": "Optional[TEXT]",
        },
        outputs={
            "index_name": "TEXT",
            "documents": "INT",
            "vectors": "INT",
        },
        tags=["rag", "ingest"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import (
            RAGDocument,
            RAGIngestor,
            TextChunker,
            default_rag_index_store,
        )

        index_name = str(inputs.get("index_name", "")).strip()
        if not index_name:
            raise ValueError("rag_ingest requires a non-empty `index_name`.")
        if not _RAG_INDEX_NAME_PATTERN.match(index_name):
            raise ValueError(
                f"index_name must match {_RAG_INDEX_NAME_PATTERN.pattern!r}; "
                f"got {index_name!r}."
            )

        documents = _normalise_rag_documents(inputs.get("documents"))
        if not documents:
            return {"index_name": index_name, "documents": 0, "vectors": 0}

        chunk_size = int(inputs.get("chunk_size", _RAG_DEFAULT_CHUNK_SIZE) or _RAG_DEFAULT_CHUNK_SIZE)
        chunk_overlap = int(inputs.get("chunk_overlap", _RAG_DEFAULT_CHUNK_OVERLAP) or _RAG_DEFAULT_CHUNK_OVERLAP)
        backend = str(inputs.get("backend", _RAG_DEFAULT_BACKEND) or _RAG_DEFAULT_BACKEND)
        progress_sink: List[int] = []

        def _progress(done: int, total: int) -> None:
            progress_sink.append(done)
            if ctx.logger is not None:
                ctx.logger.debug("rag_ingest progress %d / %d", done, total)

        store = default_rag_index_store()
        idx = store.try_get(index_name)
        if idx is None:
            # Derive dim from the embedder's first call -- we use a
            # 1-sample smoke embed to figure it out without making
            # this a public parameter.
            from models.providers import fetch_and_load_text
            probe_vec = fetch_and_load_text().embed_batch(["__probe__"])[0]
            idx = store.create(index_name, dim=len(probe_vec), backend=backend)

        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ingestor = RAGIngestor(index=idx, chunker=chunker)
        if ctx.audit is not None:
            ctx.audit.log(
                "RAG_INGEST",
                actor="node.rag_ingest",
                action="ingest",
                resource_id=index_name,
                details={"documents": len(documents), "chunk_size": chunk_size},
                severity="info",
            )
        vectors = ingestor.ingest(documents, progress=_progress)
        return {
            "index_name": index_name,
            "documents": len(documents),
            "vectors": vectors,
        }


@register_node("rag_query")
class RAGQueryNode(BaseNode):
    """Embed a free-form query and return top-k matches + context block.

    Inputs:
        ``index_name`` (str, required): Index to query.
        ``query`` (str, required): Natural-language query.
        ``top_k`` (int, optional, default 5).
        ``threshold`` (float, optional): Minimum cosine score.

    Returns:
        A dict with ``"hits"`` (list of
        ``{doc_id, chunk_id, score, metadata}``) and
        ``"context"`` (assembled evidence block).
    """

    spec: NodeSpec = NodeSpec(
        type="rag_query",
        name="RAG Query",
        description="Embed a query and return the top-k matching chunks with provenance + context block.",
        inputs={
            "index_name": "TEXT",
            "query": "TEXT",
            "top_k": "Optional[INT]",
            "threshold": "Optional[FLOAT]",
        },
        outputs={
            "index_name": "TEXT",
            "query": "TEXT",
            "hits": "JSON",
            "context": "TEXT",
        },
        tags=["rag", "query"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import RAGRetriever, default_rag_index_store

        index_name = str(inputs.get("index_name", "")).strip()
        query = str(inputs.get("query", ""))
        if not index_name:
            raise ValueError("rag_query requires a non-empty `index_name`.")
        if not query.strip():
            raise ValueError("rag_query requires a non-empty `query`.")

        store = default_rag_index_store()
        idx = store.get(index_name)
        top_k = int(inputs.get("top_k", _RAG_DEFAULT_TOP_K) or _RAG_DEFAULT_TOP_K)
        threshold_raw = inputs.get("threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None

        retriever = RAGRetriever(idx, default_top_k=top_k, default_threshold=threshold)
        if ctx.audit is not None:
            ctx.audit.log(
                "RAG_QUERY",
                actor="node.rag_query",
                action="query",
                resource_id=index_name,
                details={"top_k": top_k, "threshold": threshold},
                severity="info",
            )
        hits, context = retriever.retrieve_with_context(
            query, top_k=top_k, threshold=threshold
        )
        return {
            "index_name": index_name,
            "query": query,
            "hits": [
                {
                    "doc_id": h.doc_id,
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "metadata": h.metadata,
                }
                for h in hits
            ],
            "context": context,
        }


@register_node("rag_delete")
class RAGDeleteNode(BaseNode):
    """Delete either a single document or an entire index.

    Inputs:
        ``index_name`` (str, required).
        ``doc_id`` (str, optional): When supplied, only the
            document is removed and the index is preserved.
            When omitted (and ``drop_index`` is True), the
            entire index is dropped.

    Returns:
        A dict with ``"removed_chunks"`` or ``"dropped"`` flag.
    """

    spec: NodeSpec = NodeSpec(
        type="rag_delete",
        name="RAG Delete",
        description="Remove a single document or drop an entire RAG index.",
        inputs={
            "index_name": "TEXT",
            "doc_id": "Optional[TEXT]",
            "drop_index": "Optional[BOOL]",
        },
        outputs={
            "index_name": "TEXT",
            "doc_id": "TEXT",
            "dropped": "BOOL",
            "removed_chunks": "INT",
        },
        tags=["rag", "delete"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import default_rag_index_store

        index_name = str(inputs.get("index_name", "")).strip()
        if not index_name:
            raise ValueError("rag_delete requires a non-empty `index_name`.")
        doc_id = inputs.get("doc_id")
        drop_index = bool(inputs.get("drop_index", False))
        store = default_rag_index_store()
        if ctx.audit is not None:
            ctx.audit.log(
                "RAG_DELETE",
                actor="node.rag_delete",
                action="delete",
                resource_id=index_name,
                details={"doc_id": doc_id, "drop_index": drop_index},
                severity="warning",
            )
        if drop_index and doc_id is None:
            dropped = store.remove(index_name)
            return {"index_name": index_name, "dropped": dropped, "removed_chunks": 0}
        if doc_id is None:
            raise ValueError(
                "rag_delete: provide either `doc_id` or `drop_index=True`."
            )
        idx = store.get(index_name)
        removed = idx.delete_document(str(doc_id))
        return {
            "index_name": index_name,
            "doc_id": str(doc_id),
            "dropped": False,
            "removed_chunks": removed,
        }


@register_node("rag_list_indexes")
class RAGListIndexesNode(BaseNode):
    """Return the names of all RAG indexes in the default store.

    Inputs: none.

    Returns:
        A dict with ``"indexes"`` (sorted list of names).
    """

    spec: NodeSpec = NodeSpec(
        type="rag_list_indexes",
        name="RAG List Indexes",
        description="List all RAG indexes known to the process-wide index store.",
        inputs={},
        outputs={"indexes": "JSON"},
        tags=["rag", "list"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import default_rag_index_store

        return {"indexes": default_rag_index_store().list()}


@register_node("rag_get_index")
class RAGGetIndexNode(BaseNode):
    """Return metadata about a single RAG index.

    Inputs: ``index_name`` (str, required).

    Returns:
        A dict with ``"index_name"``, ``"size"`` (vector count),
        ``"documents"``, and ``"dim"``.
    """

    spec: NodeSpec = NodeSpec(
        type="rag_get_index",
        name="RAG Get Index",
        description="Return metadata (size, documents, dim) for a single RAG index.",
        inputs={"index_name": "TEXT"},
        outputs={
            "index_name": "TEXT",
            "size": "INT",
            "documents": "INT",
            "dim": "INT",
        },
        tags=["rag", "metadata"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import default_rag_index_store

        index_name = str(inputs.get("index_name", "")).strip()
        if not index_name:
            raise ValueError("rag_get_index requires a non-empty `index_name`.")
        idx = default_rag_index_store().get(index_name)
        return {
            "index_name": index_name,
            "size": idx.size(),
            "documents": idx.document_count(),
            "dim": idx.dim,
        }


@register_node("rag_search_text")
class RAGSearchTextNode(BaseNode):
    """Full-text keyword search over the chunk-level ``text_preview``.

    The RAG stack stores a short preview of every chunk in the
    ``metadata["text_preview"]`` field.  This node scans that
    field for case-insensitive substring matches and returns
    the matching chunks in order of cosine score, optionally
    filtered by ``doc_id``.

    Inputs:
        ``index_name`` (str, required).
        ``text`` (str, required): The substring to look for.
        ``doc_id`` (str, optional): Restrict to one document.
        ``top_k`` (int, optional, default 5).

    Returns:
        A dict with ``"matches"``.
    """

    spec: NodeSpec = NodeSpec(
        type="rag_search_text",
        name="RAG Search Text",
        description="Case-insensitive keyword search over chunk-level text_preview metadata.",
        inputs={
            "index_name": "TEXT",
            "text": "TEXT",
            "doc_id": "Optional[TEXT]",
            "top_k": "Optional[INT]",
        },
        outputs={
            "index_name": "TEXT",
            "text": "TEXT",
            "doc_id": "TEXT",
            "matches": "JSON",
        },
        tags=["rag", "search"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        from infrastructure.rag import default_rag_index_store

        index_name = str(inputs.get("index_name", "")).strip()
        text = str(inputs.get("text", ""))
        if not index_name:
            raise ValueError("rag_search_text requires a non-empty `index_name`.")
        if not text.strip():
            raise ValueError("rag_search_text requires a non-empty `text`.")
        idx = default_rag_index_store().get(index_name)
        doc_id = inputs.get("doc_id")
        top_k = int(inputs.get("top_k", _RAG_DEFAULT_TOP_K) or _RAG_DEFAULT_TOP_K)
        needle = text.lower()
        matches: List[Dict[str, Any]] = []
        for d in idx.list_documents():
            if doc_id is not None and d != str(doc_id):
                continue
            meta = idx.get_document_metadata(d) or {}
            preview = (meta.get("text_preview") or "").lower()
            if needle in preview:
                matches.append(
                    {
                        "doc_id": d,
                        "chunk_id": meta.get("chunk_id", ""),
                        "score": 1.0,
                        "metadata": meta,
                    }
                )
            if len(matches) >= top_k:
                break
        return {
            "index_name": index_name,
            "text": text,
            "doc_id": doc_id,
            "matches": matches,
        }
