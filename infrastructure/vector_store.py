"""Vector-store backends for the v0.4.x RAG stack.

This module ships two implementations of a tiny ``VectorStoreProtocol``:

* :class:`InMemoryVectorStore` -- a pure-Python / NumPy store
  with cosine / inner-product / L2 search.  Ships in the
  framework by default; no extra dependency.
* :class:`FaissVectorStore` -- an optional swap-in that uses
  :mod:`faiss` (CPU or GPU) for sub-millisecond top-k queries on
  millions of vectors.  Falls back to :class:`InMemoryVectorStore`
  when :mod:`faiss` is not installed.

The :class:`BruteForceIVF` / :class:`IndexFlat` algorithms are
intentionally avoided in v0.4.x -- the in-process deployment
target of a single system is "10k-1M vectors, batched top-k,
sub-100ms".  The NumPy / FAISS path covers both ends without
introducing a new dependency.

The :class:`VectorIndex` dataclass bundles a vector, its source
``doc_id`` and ``chunk_id``, and arbitrary ``metadata`` so the
RAG layer can recover document-level provenance after a search.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from infrastructure.logger import get_logger

__all__ = [
    "VectorIndex",
    "VectorStoreProtocol",
    "InMemoryVectorStore",
    "FaissVectorStore",
    "make_vector_store",
]


_logger = get_logger("infrastructure.vector_store")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class VectorIndex:
    """A single indexed vector with its source provenance.

    Attributes:
        doc_id: Stable id of the source document (e.g. ``"wiki/llm"``).
        chunk_id: Stable id of the chunk within the document
            (e.g. ``"chunk-3"``); the pair ``(doc_id, chunk_id)``
            uniquely identifies a chunk in the RAG stack.
        vector: L2-normalised float list of length ``dim``.
        metadata: Free-form JSON-serialisable metadata (e.g.
            ``{"title": "...", "offset": 42, "text_preview": "..."}``).
    """

    doc_id: str
    chunk_id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id or not isinstance(self.doc_id, str):
            raise ValueError(f"doc_id must be a non-empty string, got {self.doc_id!r}.")
        if not self.chunk_id or not isinstance(self.chunk_id, str):
            raise ValueError(
                f"chunk_id must be a non-empty string, got {self.chunk_id!r}."
            )
        if not self.vector:
            raise ValueError("vector must be a non-empty float list.")
        for v in self.vector:
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"vector entries must be numeric, got {type(v).__name__}."
                )


@dataclass
class SearchHit:
    """A single search result returned by the vector store.

    Attributes:
        doc_id: The source document id of the matched chunk.
        chunk_id: The matched chunk id within ``doc_id``.
        score: Similarity score (cosine by default, range ``[-1, 1]``).
        metadata: The metadata stored alongside the vector.
    """

    doc_id: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
class VectorStoreProtocol(Protocol):
    """The interface every vector store backend implements."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality of the store."""
        ...

    def add(self, items: Sequence[VectorIndex]) -> None:
        """Append vectors to the store."""
        ...

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchHit]:
        """Return the top-k most similar chunks for ``query``."""
        ...

    def delete(self, doc_id: str) -> int:
        """Remove every chunk belonging to ``doc_id``. Returns count removed."""
        ...

    def clear(self) -> None:
        """Remove all vectors from the store."""
        ...

    def size(self) -> int:
        """Number of vectors currently indexed."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _l2_normalise(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return ``vec / (||vec|| + eps)`` row-wise for 2-D inputs."""
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(float(norm), eps)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norms, eps)


# ---------------------------------------------------------------------------
# In-memory NumPy store
# ---------------------------------------------------------------------------
class InMemoryVectorStore:
    """A pure-NumPy implementation of :class:`VectorStoreProtocol`.

    Vectors are kept in a single ``(N, D)`` :class:`numpy.ndarray`
    that is **appended** to as new chunks arrive.  The store
    re-builds the L2-normalised copy on demand, so cosine search
    is O(N) and dominated by the BLAS dot product.

    Suitable for indices up to ~1M vectors at dim 256; for
    larger indices, install :mod:`faiss` and switch to
    :class:`FaissVectorStore`.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        self._dim: int = int(dim)
        self._lock: threading.RLock = threading.RLock()
        # ``_chunks`` holds provenance in insertion order; ``_vectors``
        # is the (N, D) row-major NumPy array of float32 vectors.
        self._chunks: List[VectorIndex] = []
        self._vectors: Optional[np.ndarray] = None
        self._dirty: bool = True
        self._logger = get_logger("infrastructure.vector_store.in_memory")

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_dim(self, vec: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[-1] != self._dim:
            raise ValueError(
                f"vector dimension {arr.shape[-1]} does not match store dim {self._dim}."
            )
        return arr

    def _normalised_matrix(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, self._dim), dtype=np.float32)
        if self._dirty or self._vectors is None or self._vectors.shape[0] != len(self._chunks):
            self._vectors = np.stack(
                [_l2_normalise(np.asarray(c.vector, dtype=np.float32)) for c in self._chunks],
                axis=0,
            )
            self._dirty = False
        return self._vectors

    def add(self, items: Sequence[VectorIndex]) -> None:
        items = list(items)
        if not items:
            return
        with self._lock:
            for item in items:
                if len(item.vector) != self._dim:
                    raise ValueError(
                        f"vector dim {len(item.vector)} != store dim {self._dim}"
                    )
            self._chunks.extend(items)
            self._dirty = True

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchHit]:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        q = self._ensure_dim(query)
        with self._lock:
            if not self._chunks:
                return []
            mat = self._normalised_matrix()
            qn = _l2_normalise(q)
            scores = mat @ qn
            k = min(top_k, len(self._chunks))
            # argpartition is O(N) and avoids a full sort.
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            hits: List[SearchHit] = []
            for i in idx:
                score = float(scores[i])
                if threshold is not None and score < threshold:
                    continue
                chunk = self._chunks[int(i)]
                hits.append(
                    SearchHit(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        score=score,
                        metadata=dict(chunk.metadata),
                    )
                )
            return hits

    def delete(self, doc_id: str) -> int:
        with self._lock:
            kept: List[VectorIndex] = []
            removed = 0
            for c in self._chunks:
                if c.doc_id == doc_id:
                    removed += 1
                else:
                    kept.append(c)
            if removed:
                self._chunks = kept
                self._dirty = True
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._vectors = None
            self._dirty = True

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)


# ---------------------------------------------------------------------------
# Optional FAISS backend
# ---------------------------------------------------------------------------
class FaissVectorStore:
    """An optional swap-in that uses :mod:`faiss` for top-k queries.

    Falls back to :class:`InMemoryVectorStore` (with a clear log
    message) when :mod:`faiss` is not installed.  This lets
    callers always construct a :class:`FaissVectorStore` and get
    a working store regardless of the dependency state.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        self._dim: int = int(dim)
        self._lock: threading.RLock = threading.RLock()
        self._faiss = _try_import_faiss()
        if self._faiss is not None:
            self._index: Any = self._faiss.IndexFlatIP(dim)
            self._fallback: Optional[InMemoryVectorStore] = None
            self._backend_name: str = "faiss"
        else:
            self._index = None
            self._fallback = InMemoryVectorStore(dim=dim)
            self._backend_name = "in_memory"
            _logger.info(
                "faiss is not installed; FaissVectorStore is using the in-memory "
                "backend. Install faiss-cpu / faiss-gpu for sub-ms top-k on large "
                "indices."
            )
        # We still keep chunk metadata in a list (faiss only stores
        # vectors).
        self._chunks: List[VectorIndex] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> str:
        """``"faiss"`` when faiss is available, else ``"in_memory"``."""
        return self._backend_name

    def add(self, items: Sequence[VectorIndex]) -> None:
        items = list(items)
        if not items:
            return
        with self._lock:
            for item in items:
                if len(item.vector) != self._dim:
                    raise ValueError(
                        f"vector dim {len(item.vector)} != store dim {self._dim}"
                    )
            if self._index is not None:
                arr = np.stack(
                    [_l2_normalise(np.asarray(c.vector, dtype=np.float32)) for c in items],
                    axis=0,
                )
                self._index.add(arr)
            else:
                assert self._fallback is not None
                self._fallback.add(items)
            self._chunks.extend(items)

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchHit]:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        q = np.asarray(query, dtype=np.float32)
        if q.shape[-1] != self._dim:
            raise ValueError(
                f"query dim {q.shape[-1]} != store dim {self._dim}"
            )
        with self._lock:
            if not self._chunks:
                return []
            if self._index is not None:
                qn = _l2_normalise(q).reshape(1, -1)
                scores, idx = self._index.search(qn, min(top_k, len(self._chunks)))
                hits: List[SearchHit] = []
                for s, i in zip(scores[0].tolist(), idx[0].tolist()):
                    if i < 0 or i >= len(self._chunks):
                        continue
                    if threshold is not None and s < threshold:
                        continue
                    chunk = self._chunks[i]
                    hits.append(
                        SearchHit(
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.chunk_id,
                            score=float(s),
                            metadata=dict(chunk.metadata),
                        )
                    )
                return hits
            assert self._fallback is not None
            return self._fallback.search(query, top_k=top_k, threshold=threshold)

    def delete(self, doc_id: str) -> int:
        with self._lock:
            kept: List[VectorIndex] = []
            removed = 0
            for c in self._chunks:
                if c.doc_id == doc_id:
                    removed += 1
                else:
                    kept.append(c)
            if removed:
                # Re-build the index from scratch -- faiss does not
                # support arbitrary deletion in IndexFlatIP.  For
                # larger indices, callers should use the IVF /
                # HNSW variants.
                if self._index is not None:
                    self._index = self._faiss.IndexFlatIP(self._dim)
                    if kept:
                        arr = np.stack(
                            [_l2_normalise(np.asarray(c.vector, dtype=np.float32))
                             for c in kept],
                            axis=0,
                        )
                        self._index.add(arr)
                self._chunks = kept
            return removed

    def clear(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index = self._faiss.IndexFlatIP(self._dim)
            else:
                assert self._fallback is not None
                self._fallback.clear()
            self._chunks.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
def _try_import_faiss() -> Any:
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_vector_store(
    *,
    dim: int,
    backend: str = "auto",
) -> VectorStoreProtocol:
    """Build a :class:`VectorStoreProtocol` from a backend name.

    Args:
        dim: Embedding dimensionality.
        backend: One of ``"auto"`` (default; uses faiss if
            installed, else in-memory), ``"in_memory"`` or
            ``"faiss"``.

    Returns:
        A vector store instance with a uniform protocol.
    """
    backend = backend.lower().strip()
    if backend == "auto":
        if _try_import_faiss() is not None:
            return FaissVectorStore(dim=dim)
        return InMemoryVectorStore(dim=dim)
    if backend == "in_memory":
        return InMemoryVectorStore(dim=dim)
    if backend == "faiss":
        return FaissVectorStore(dim=dim)
    raise ValueError(
        f"unknown vector store backend: {backend!r} (expected 'auto', "
        f"'in_memory' or 'faiss')."
    )
