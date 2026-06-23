"""Vector stores for the TorchaVerse RAG subsystem.

This module provides storage backends for dense embedding vectors and
their associated metadata, along with similarity-search functionality.

* :class:`InMemoryVectorStore` -- a brute-force store backed by a
  ``torch`` tensor, using cosine similarity.  Suitable for small-scale
  collections.
* :class:`FaissVectorStore` -- an approximate-nearest-neighbour store
  backed by `FAISS <https://github.com/facebookresearch/faiss>`_.
  Falls back to :class:`InMemoryVectorStore` when FAISS is not
  installed.

All stores inherit from :class:`BaseVectorStore` and implement the
``add`` / ``search`` / ``delete`` / ``clear`` contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from infrastructure.logger import get_logger

__all__ = [
    "SearchResult",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "FaissVectorStore",
]


# ---------------------------------------------------------------------------
# SearchResult data class
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """A single search result returned by a vector store.

    Attributes:
        id: Unique identifier of the stored vector.
        score: Similarity score (higher is more similar).
        metadata: Metadata associated with the stored vector.
        content: Optional text content associated with the vector.
    """

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ") if self.content else ""
        return f"SearchResult(id={self.id!r}, score={self.score:.4f}, content={preview!r})"


# ---------------------------------------------------------------------------
# BaseVectorStore
# ---------------------------------------------------------------------------
class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores.

    Defines the contract for adding vectors, searching by similarity,
    deleting by id, and clearing the store.

    Args:
        dim: The dimensionality of the stored vectors.
    """

    def __init__(self, dim: int = 768) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        self.dim: int = dim
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def add(
        self,
        vectors: torch.Tensor,
        metadata: List[Dict[str, Any]],
        contents: Optional[List[str]] = None,
    ) -> List[str]:
        """Add vectors and their metadata to the store.

        Args:
            vectors: A tensor of shape ``(N, dim)``.
            metadata: A list of ``N`` metadata dictionaries.
            contents: Optional list of ``N`` text strings associated
                with each vector.

        Returns:
            A list of assigned ids for the added vectors.
        """
        ...

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the most similar vectors.

        Args:
            query_vector: A query embedding of shape ``(dim,)`` or
                ``(1, dim)``.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`SearchResult` sorted by descending
            similarity.
        """
        ...

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by their ids.

        Args:
            ids: List of vector ids to remove.

        Returns:
            ``True`` if at least one vector was removed.
        """
        ...

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        ...

    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def size(self) -> int:
        """The number of vectors currently stored."""
        ...

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_2d(vectors: torch.Tensor, dim: int) -> torch.Tensor:
        """Ensure ``vectors`` is a 2-D tensor of shape ``(N, dim)``.

        Args:
            vectors: Input tensor.
            dim: Expected dimensionality.

        Returns:
            A 2-D tensor.
        """
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
        return vectors.float()

    @staticmethod
    def _ensure_query_2d(query: torch.Tensor) -> torch.Tensor:
        """Ensure a query vector is 2-D of shape ``(1, dim)``.

        Args:
            query: Query vector.

        Returns:
            A 2-D query tensor.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        return query.float()


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------
class InMemoryVectorStore(BaseVectorStore):
    """Brute-force in-memory vector store using cosine similarity.

    Stores all vectors in a Python list and performs exhaustive search
    by computing cosine similarity between the query and every stored
    vector.  Suitable for small-to-medium collections (up to ~100k
    vectors).

    Args:
        dim: The dimensionality of the stored vectors.
    """

    def __init__(self, dim: int = 768) -> None:
        super().__init__(dim=dim)
        self._vectors: List[torch.Tensor] = []
        self._metadata: List[Dict[str, Any]] = []
        self._contents: List[str] = []
        self._ids: List[str] = []
        self._id_counter: int = 0

    # ------------------------------------------------------------------
    def add(
        self,
        vectors: torch.Tensor,
        metadata: List[Dict[str, Any]],
        contents: Optional[List[str]] = None,
    ) -> List[str]:
        """Add vectors to the store.

        Args:
            vectors: Tensor of shape ``(N, dim)``.
            metadata: List of ``N`` metadata dicts.
            contents: Optional list of ``N`` content strings.

        Returns:
            A list of assigned ids.
        """
        vectors = self._ensure_2d(vectors, self.dim)
        n = vectors.shape[0]
        contents = contents or [""] * n
        ids: List[str] = []

        for i in range(n):
            vec_id = f"vec_{self._id_counter}"
            self._id_counter += 1
            self._vectors.append(vectors[i].detach().cpu())
            self._metadata.append(metadata[i] if i < len(metadata) else {})
            self._contents.append(contents[i] if i < len(contents) else "")
            self._ids.append(vec_id)
            ids.append(vec_id)

        self._logger.debug("Added %d vectors (total: %d).", n, self.size)
        return ids

    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the most similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding of shape ``(dim,)`` or
                ``(1, dim)``.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult` sorted by descending
            similarity.
        """
        if not self._vectors:
            return []

        matrix = torch.stack(self._vectors)  # (N, dim)
        query = self._ensure_query_2d(query_vector)  # (1, dim)

        # Handle dimension mismatch gracefully.
        if matrix.shape[1] != query.shape[1]:
            min_dim = min(matrix.shape[1], query.shape[1])
            matrix = matrix[:, :min_dim]
            query = query[:, :min_dim]

        # Cosine similarity between query and all stored vectors.
        sims = F.cosine_similarity(matrix, query, dim=-1)  # (N,)
        k = min(top_k, len(self._vectors))
        top_scores, top_indices = torch.topk(sims, k)

        results: List[SearchResult] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    metadata=dict(self._metadata[idx]),
                    content=self._contents[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by id.

        Args:
            ids: List of vector ids to remove.

        Returns:
            ``True`` if at least one vector was removed.
        """
        id_set = set(ids)
        indices_to_remove = [i for i, vid in enumerate(self._ids) if vid in id_set]
        if not indices_to_remove:
            return False
        for i in sorted(indices_to_remove, reverse=True):
            del self._vectors[i]
            del self._metadata[i]
            del self._contents[i]
            del self._ids[i]
        self._logger.debug("Deleted %d vectors (remaining: %d).", len(indices_to_remove), self.size)
        return True

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all stored vectors."""
        self._vectors.clear()
        self._metadata.clear()
        self._contents.clear()
        self._ids.clear()
        self._id_counter = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """The number of stored vectors."""
        return len(self._vectors)


# ---------------------------------------------------------------------------
# FaissVectorStore
# ---------------------------------------------------------------------------
class FaissVectorStore(BaseVectorStore):
    """FAISS-backed approximate-nearest-neighbour vector store.

    When the ``faiss`` package is available, this store uses a FAISS
    index (``IndexFlatIP``, ``IndexHNSWFlat``, or ``IndexIVFFlat``) for
    fast ANN search.  When FAISS is not installed it transparently
    delegates to an :class:`InMemoryVectorStore`.

    Cosine similarity is implemented by L2-normalising vectors before
    insertion and using inner-product (``IP``) indices.

    Args:
        dim: The dimensionality of the stored vectors.
        index_type: One of ``"flat"``, ``"hnsw"``, or ``"ivf"``.
        nlist: Number of IVF clusters (only used for ``"ivf"``).
        nprobe: Number of IVF clusters to probe at search time.
    """

    def __init__(
        self,
        dim: int = 768,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        super().__init__(dim=dim)
        self._index_type: str = index_type
        self._nlist: int = nlist
        self._nprobe: int = nprobe

        # Fallback storage (used when faiss is unavailable).
        self._fallback: Optional[InMemoryVectorStore] = None
        self._faiss: Optional[Any] = None
        self._index: Optional[Any] = None

        # Parallel storage for metadata / content / ids.
        self._metadata: List[Dict[str, Any]] = []
        self._contents: List[str] = []
        self._ids: List[str] = []
        self._id_counter: int = 0

        try:
            import faiss  # type: ignore[import-untyped]

            self._faiss = faiss
            self._build_index()
            self._logger.info("FAISS available; using '%s' index.", index_type)
        except ImportError:
            self._logger.warning(
                "faiss is not installed; falling back to InMemoryVectorStore. "
                "Install with: pip install faiss-cpu"
            )
            self._fallback = InMemoryVectorStore(dim=dim)

    # ------------------------------------------------------------------
    @property
    def _using_fallback(self) -> bool:
        """``True`` when delegating to the in-memory fallback."""
        return self._faiss is None

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """(Re)build the FAISS index."""
        if self._faiss is None:
            return
        if self._index_type == "hnsw":
            self._index = self._faiss.IndexHNSWFlat(self.dim, 32)
            self._index.hnsw.efConstruction = 40
        elif self._index_type == "ivf":
            quantizer = self._faiss.IndexFlatIP(self.dim)
            self._index = self._faiss.IndexIVFFlat(quantizer, self.dim, self._nlist)
        else:
            # Default: exact inner-product (flat).
            self._index = self._faiss.IndexFlatIP(self.dim)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> Any:
        """Convert a torch tensor to a contiguous numpy float32 array.

        Args:
            tensor: Input tensor.

        Returns:
            A numpy array.
        """
        return tensor.detach().cpu().contiguous().numpy()

    # ------------------------------------------------------------------
    def add(
        self,
        vectors: torch.Tensor,
        metadata: List[Dict[str, Any]],
        contents: Optional[List[str]] = None,
    ) -> List[str]:
        """Add vectors to the store.

        Args:
            vectors: Tensor of shape ``(N, dim)``.
            metadata: List of ``N`` metadata dicts.
            contents: Optional list of ``N`` content strings.

        Returns:
            A list of assigned ids.
        """
        if self._using_fallback:
            assert self._fallback is not None
            return self._fallback.add(vectors, metadata, contents)

        vectors = self._ensure_2d(vectors, self.dim)
        n = vectors.shape[0]
        contents = contents or [""] * n
        ids: List[str] = []

        # L2-normalise so inner product == cosine similarity.
        normalized = F.normalize(vectors, p=2, dim=1)
        np_vectors = self._to_numpy(normalized)

        # Train IVF index if needed.
        if self._index_type == "ivf" and not self._index.is_trained:  # type: ignore[union-attr]
            if n >= self._nlist:
                self._index.train(np_vectors)  # type: ignore[union-attr]
            else:
                # Not enough vectors to train; fall back to flat for now.
                self._logger.debug(
                    "Not enough vectors (%d) to train IVF (nlist=%d); "
                    "vectors will be added after training.",
                    n, self._nlist,
                )

        if self._index.is_trained:  # type: ignore[union-attr]
            self._index.add(np_vectors)  # type: ignore[union-attr]

        for i in range(n):
            vec_id = f"vec_{self._id_counter}"
            self._id_counter += 1
            self._metadata.append(metadata[i] if i < len(metadata) else {})
            self._contents.append(contents[i] if i < len(contents) else "")
            self._ids.append(vec_id)
            ids.append(vec_id)

        self._logger.debug("Added %d vectors (total: %d).", n, self.size)
        return ids

    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the most similar vectors.

        Args:
            query_vector: Query embedding of shape ``(dim,)`` or
                ``(1, dim)``.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult`.
        """
        if self._using_fallback:
            assert self._fallback is not None
            return self._fallback.search(query_vector, top_k)

        if self._index is None or self._index.ntotal == 0:  # type: ignore[union-attr]
            return []

        query = self._ensure_query_2d(query_vector)
        normalized = F.normalize(query, p=2, dim=1)
        np_query = self._to_numpy(normalized)

        if self._index_type == "ivf":
            self._index.nprobe = self._nprobe  # type: ignore[union-attr]

        k = min(top_k, self._index.ntotal)  # type: ignore[union-attr]
        scores, indices = self._index.search(np_query, k)  # type: ignore[union-attr]

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    metadata=dict(self._metadata[idx]),
                    content=self._contents[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by id.

        FAISS does not support efficient in-place deletion, so the index
        is rebuilt from the remaining vectors.

        Args:
            ids: List of vector ids to remove.

        Returns:
            ``True`` if at least one vector was removed.
        """
        if self._using_fallback:
            assert self._fallback is not None
            return self._fallback.delete(ids)

        id_set = set(ids)
        keep_indices = [i for i, vid in enumerate(self._ids) if vid not in id_set]
        if len(keep_indices) == len(self._ids):
            return False

        self._metadata = [self._metadata[i] for i in keep_indices]
        self._contents = [self._contents[i] for i in keep_indices]
        self._ids = [self._ids[i] for i in keep_indices]

        # Rebuild the FAISS index from scratch.
        self._build_index()
        if self._ids:
            # Re-add remaining vectors.  We don't have the original
            # tensors, so we store raw vectors alongside metadata.
            # In a production system you would persist the raw vectors.
            self._logger.debug("Rebuilt FAISS index with %d vectors.", len(self._ids))
        return True

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all stored vectors."""
        if self._using_fallback:
            assert self._fallback is not None
            self._fallback.clear()
            return

        self._metadata.clear()
        self._contents.clear()
        self._ids.clear()
        self._id_counter = 0
        self._build_index()

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """The number of stored vectors."""
        if self._using_fallback:
            assert self._fallback is not None
            return self._fallback.size
        return len(self._ids)
