"""Retrieval-Augmented Generation (RAG) building blocks for v0.4.x.

This module is the framework-side implementation of the RAG
stack used by the ``rag_*`` L4 nodes and the
``/serving/v1/rag/*`` HTTP endpoints.

The public surface is intentionally small:

* :class:`TextChunker` -- a deterministic, sliding-window text
  splitter that produces chunks of at most
  ``chunk_size`` characters (with a ``chunk_overlap`` tail of
  the previous chunk to preserve context).  No external
  dependency, no tokeniser assumption.
* :class:`RAGIngestor` -- takes a list of documents
  (``doc_id -> text``), chunks them, embeds them with a
  :class:`models.interfaces.llm_provider.LLMProvider`, and
  inserts the resulting :class:`VectorIndex` records into a
  :class:`infrastructure.vector_store.VectorStoreProtocol`.
* :class:`RAGRetriever` -- embeds a free-form query, runs top-k
  on the vector store, and returns the matching chunks with
  their provenance.
* :class:`RAGIndex` -- the per-tenant / per-purpose container
  bundling a single :class:`VectorStoreProtocol` and a
  metadata index.  Use :class:`RAGIndexStore` to keep many
  indexes side-by-side (e.g. one per tenant, one per project).

The RAG stack is **single-process** by design; it is the
v0.4.x fit for the project's "single-system" roadmap.  A
distributed replacement is left for v1.0.0+.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger
from infrastructure.vector_store import (
    InMemoryVectorStore,
    SearchHit,
    VectorIndex,
    VectorStoreProtocol,
    make_vector_store,
)

__all__ = [
    "TextChunker",
    "RAGDocument",
    "RAGIngestor",
    "RAGRetriever",
    "RAGIndex",
    "RAGIndexStore",
    "default_rag_index_store",
]


_logger = get_logger("infrastructure.rag")


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
@dataclass
class TextChunker:
    """Sliding-window character-level text chunker.

    Attributes:
        chunk_size: Target maximum chunk size in characters
            (default 512).  Must be > 0.
        chunk_overlap: Number of characters of overlap between
            consecutive chunks (default 64).  Must be in
            ``[0, chunk_size)``.
        separator: Optional hard separator (e.g. ``"\\n\\n"``).
            When supplied, splits attempt to break on the
            separator before applying the sliding window.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    separator: Optional[str] = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap} "
                f"with chunk_size={self.chunk_size}."
            )

    def split(self, text: str) -> List[str]:
        """Return a list of non-empty chunk strings covering ``text``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text:
            return []
        # Pre-split on the hard separator first, then chunk each
        # resulting segment with the sliding window.
        if self.separator:
            segments: List[str] = []
            for part in text.split(self.separator):
                part = part.strip()
                if part:
                    segments.append(part)
        else:
            segments = [text]

        chunks: List[str] = []
        for segment in segments:
            if len(segment) <= self.chunk_size:
                chunks.append(segment)
                continue
            start = 0
            n = len(segment)
            while start < n:
                end = min(start + self.chunk_size, n)
                piece = segment[start:end]
                if piece.strip():
                    chunks.append(piece)
                if end == n:
                    break
                start = end - self.chunk_overlap
        return chunks


# ---------------------------------------------------------------------------
# Ingestor / Retriever
# ---------------------------------------------------------------------------
@dataclass
class RAGDocument:
    """A document destined for the RAG stack.

    Attributes:
        doc_id: Stable id; if empty, a uuid4 hex is generated.
        text: The full document text (will be chunked).
        metadata: Optional free-form JSON-serialisable metadata
            propagated to every chunk of the document.
    """

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}.")


Embedder = Callable[[Sequence[str]], List[List[float]]]


class RAGIngestor:
    """Chunk + embed + insert documents into a :class:`RAGIndex`."""

    def __init__(
        self,
        index: "RAGIndex",
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self._index: "RAGIndex" = index
        self._chunker: TextChunker = chunker or TextChunker()
        if embedder is None:
            from models.providers import fetch_and_load_text  # local import

            provider = fetch_and_load_text()
            self._embedder: Embedder = lambda texts: provider.embed_batch(list(texts))  # noqa: E731
        else:
            self._embedder = embedder
        self._logger = get_logger("infrastructure.rag.ingestor")

    def ingest(
        self,
        documents: Sequence[RAGDocument],
        *,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Insert all chunks of ``documents`` into the index.

        Returns the number of vectors written.  ``progress`` is
        an optional callback ``(done, total)`` invoked once per
        batch -- useful for surfacing progress in the serving
        layer's ``POST /v1/rag/ingest`` endpoint.
        """
        if not documents:
            return 0
        total = 0
        batch_texts: List[str] = []
        batch_chunks: List[VectorIndex] = []
        # Batch up to 32 chunks per embed call -- the LocalTorch
        # provider handles batches of 32 in a single forward
        # pass which is materially faster than per-chunk calls.
        batch_threshold = 32

        def _flush() -> None:
            nonlocal total
            if not batch_chunks:
                return
            vectors = self._embedder(batch_texts)
            if len(vectors) != len(batch_chunks):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for "
                    f"{len(batch_chunks)} chunks"
                )
            for chunk, vec in zip(batch_chunks, vectors):
                chunk.vector = list(vec)
            self._index.add(batch_chunks)
            total += len(batch_chunks)
            if progress is not None:
                progress(total, total + len(batch_texts))  # rough, see below
            batch_texts.clear()
            batch_chunks.clear()

        for doc in documents:
            for chunk_idx, chunk_text in enumerate(self._chunker.split(doc.text)):
                if not chunk_text.strip():
                    continue
                chunk_id = f"chunk-{chunk_idx:04d}"
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.doc_id
                meta["chunk_id"] = chunk_id
                meta["text_preview"] = chunk_text[:80]
                batch_chunks.append(
                    VectorIndex(
                        doc_id=doc.doc_id,
                        chunk_id=chunk_id,
                        vector=[0.0] * self._index.dim,  # placeholder, replaced by embedder
                        metadata=meta,
                    )
                )
                batch_texts.append(chunk_text)
                if len(batch_chunks) >= batch_threshold:
                    _flush()
        _flush()
        if progress is not None:
            progress(total, total)
        self._logger.info(
            "RAG ingest: %d documents -> %d vectors", len(documents), total
        )
        return total


class RAGRetriever:
    """Embed + top-k search against a :class:`RAGIndex`."""

    def __init__(
        self,
        index: "RAGIndex",
        embedder: Optional[Embedder] = None,
        *,
        default_top_k: int = 5,
        default_threshold: Optional[float] = None,
    ) -> None:
        self._index: "RAGIndex" = index
        if embedder is None:
            from models.providers import fetch_and_load_text  # local import

            provider = fetch_and_load_text()
            self._embedder: Embedder = lambda texts: provider.embed_batch(list(texts))  # noqa: E731
        else:
            self._embedder = embedder
        if default_top_k <= 0:
            raise ValueError(f"default_top_k must be > 0, got {default_top_k}.")
        self._default_top_k: int = int(default_top_k)
        self._default_threshold: Optional[float] = default_threshold
        self._logger = get_logger("infrastructure.rag.retriever")

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[SearchHit]:
        """Embed ``query`` and return the top-k matching chunks."""
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else self._default_top_k
        thr = threshold if threshold is not None else self._default_threshold
        vectors = self._embedder([query])
        if not vectors:
            return []
        return self._index.search(vectors[0], top_k=k, threshold=thr)

    def retrieve_with_context(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[SearchHit], str]:
        """Convenience: return ``(hits, assembled_context)`` for the query.

        The assembled context concatenates each hit's
        ``metadata["text_preview"]`` (or the chunk id when no
        preview is stored), separated by ``"\\n\\n---\\n\\n"``,
        so the LLM can be prompted with the retrieved evidence.
        """
        hits = self.retrieve(query, top_k=top_k, threshold=threshold)
        blocks: List[str] = []
        for hit in hits:
            text = hit.metadata.get("text_preview") or f"[{hit.doc_id}:{hit.chunk_id}]"
            blocks.append(
                f"doc={hit.doc_id} chunk={hit.chunk_id} score={hit.score:.3f}\n{text}"
            )
        return hits, "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Index container
# ---------------------------------------------------------------------------
class RAGIndex:
    """A single RAG index: a vector store + a metadata map."""

    def __init__(
        self,
        name: str,
        *,
        dim: int,
        backend: str = "auto",
        store: Optional[VectorStoreProtocol] = None,
    ) -> None:
        if not name:
            raise ValueError("name must be a non-empty string.")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        self._name: str = name
        self._dim: int = int(dim)
        self._store: VectorStoreProtocol = store or make_vector_store(dim=dim, backend=backend)
        if self._store.dim != dim:
            raise ValueError(
                f"store.dim {self._store.dim} does not match RAGIndex dim {dim}."
            )
        self._lock: threading.RLock = threading.RLock()
        self._doc_count: int = 0
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}
        self._logger = get_logger(f"infrastructure.rag.index.{name}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def store(self) -> VectorStoreProtocol:
        """The underlying :class:`VectorStoreProtocol` (read-only access)."""
        return self._store

    def size(self) -> int:
        return self._store.size()

    def document_count(self) -> int:
        with self._lock:
            return self._doc_count

    def add(self, chunks: Sequence[VectorIndex]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        new_docs: set = set()
        for c in chunks:
            new_docs.add(c.doc_id)
            self._doc_metadata.setdefault(c.doc_id, dict(c.metadata))
        self._store.add(chunks)
        with self._lock:
            self._doc_count += len(new_docs)

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchHit]:
        return self._store.search(query, top_k=top_k, threshold=threshold)

    def delete_document(self, doc_id: str) -> int:
        n = self._store.delete(doc_id)
        with self._lock:
            self._doc_metadata.pop(doc_id, None)
            if n:
                self._doc_count = max(0, self._doc_count - 1)
        return n

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._doc_metadata.get(doc_id, {}))

    def list_documents(self) -> List[str]:
        with self._lock:
            return sorted(self._doc_metadata.keys())

    def clear(self) -> None:
        self._store.clear()
        with self._lock:
            self._doc_count = 0
            self._doc_metadata.clear()


# ---------------------------------------------------------------------------
# Multi-index store
# ---------------------------------------------------------------------------
class RAGIndexStore:
    """A process-wide index of :class:`RAGIndex` objects."""

    def __init__(self) -> None:
        self._indexes: Dict[str, RAGIndex] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("infrastructure.rag.index_store")

    def create(
        self,
        name: str,
        *,
        dim: int,
        backend: str = "auto",
    ) -> RAGIndex:
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"RAG index {name!r} already exists.")
            idx = RAGIndex(name=name, dim=dim, backend=backend)
            self._indexes[name] = idx
            return idx

    def get(self, name: str) -> RAGIndex:
        with self._lock:
            idx = self._indexes.get(name)
            if idx is None:
                raise KeyError(f"no RAG index named {name!r}")
            return idx

    def try_get(self, name: str) -> Optional[RAGIndex]:
        with self._lock:
            return self._indexes.get(name)

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._indexes.pop(name, None) is not None

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._indexes.keys())

    def clear(self) -> None:
        with self._lock:
            for idx in self._indexes.values():
                idx.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.list()

    def __len__(self) -> int:
        with self._lock:
            return len(self._indexes)


#: Process-wide default :class:`RAGIndexStore`; tests are free
#: to construct their own.
_default_store: RAGIndexStore = RAGIndexStore()


def default_rag_index_store() -> RAGIndexStore:
    """Return the process-wide default :class:`RAGIndexStore`."""
    return _default_store
