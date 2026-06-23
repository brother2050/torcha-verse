"""Retrievers for the TorchaVerse RAG subsystem.

This module provides the retrieval layer that sits between the vector
store and the generation engine.  It includes:

* :class:`VectorRetriever` -- pure dense (embedding-based) retrieval.
* :class:`HybridRetriever` -- combines dense retrieval with sparse
  BM25 keyword matching.
* :class:`QueryRewriter` -- query expansion via HyDE (hypothetical
  document embeddings) and multi-query generation.
* :class:`Reranker` -- a simplified cross-encoder reranker that
  re-scores candidates by cosine similarity.
* :class:`ContextAssembler` -- assembles and de-duplicates retrieved
  context into a single prompt string.

All retrievers inherit from :class:`BaseRetriever` and implement the
``retrieve(query, top_k) -> List[SearchResult]`` contract.
"""

from __future__ import annotations

import abc
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from infrastructure.logger import get_logger
from rag.vectorstore.vector_store import BaseVectorStore, SearchResult

__all__ = [
    "BaseRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "QueryRewriter",
    "Reranker",
    "ContextAssembler",
]

#: Type alias for a callable that converts text into a dense embedding.
EmbedFn = Callable[[str], torch.Tensor]


# ---------------------------------------------------------------------------
# BaseRetriever
# ---------------------------------------------------------------------------
class BaseRetriever(abc.ABC):
    """Abstract base class for all retrievers.

    Args:
        vector_store: The backing vector store.
        embed_fn: A callable that converts a text string into a
            ``torch.Tensor`` embedding.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self.vector_store: BaseVectorStore = vector_store
        self._embed_fn: Optional[EmbedFn] = embed_fn
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve the top-k results for ``query``.

        Args:
            query: The query text.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult` sorted by relevance.
        """
        ...

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> torch.Tensor:
        """Embed ``text`` using the configured embed function.

        Args:
            text: Text to embed.

        Returns:
            A 1-D embedding tensor.

        Raises:
            RuntimeError: If no embed function is configured.
        """
        if self._embed_fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires an embed_fn to embed text."
            )
        return self._embed_fn(text)


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------
class VectorRetriever(BaseRetriever):
    """Dense (embedding-based) retriever.

    Embeds the query and performs a similarity search against the
    backing vector store.

    Args:
        vector_store: The vector store to search.
        embed_fn: Callable that converts text to an embedding.
    """

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve the top-k results via dense similarity search.

        Args:
            query: The query text.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult`.
        """
        query_vector = self._embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        self._logger.debug(
            "VectorRetriever returned %d results for '%s...'.",
            len(results), query[:50],
        )
        return results


# ---------------------------------------------------------------------------
# BM25 (internal helper)
# ---------------------------------------------------------------------------
class _BM25:
    """A simple in-memory Okapi BM25 implementation.

    Args:
        documents: List of document texts.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.
    """

    _TOKEN_RE = re.compile(r"\w+")

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1: float = k1
        self.b: float = b
        self.docs: List[List[str]] = [self._tokenize(d) for d in documents]
        self.doc_len: List[int] = [len(d) for d in self.docs]
        self.avg_len: float = (
            sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        )
        self.N: int = len(self.docs)

        # Document frequency for each term.
        self.doc_freq: Dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    # ------------------------------------------------------------------
    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Lowercase and tokenize ``text`` into word tokens."""
        return cls._TOKEN_RE.findall(text.lower())

    # ------------------------------------------------------------------
    def score(self, query: str, doc_idx: int) -> float:
        """Compute the BM25 score of ``query`` against document ``doc_idx``.

        Args:
            query: The query text.
            doc_idx: The document index.

        Returns:
            The BM25 relevance score.
        """
        query_terms = self._tokenize(query)
        doc = self.docs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        score = 0.0

        for term in query_terms:
            if term not in self.doc_freq:
                continue
            # Inverse document frequency.
            idf = math.log(
                (self.N - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5) + 1.0
            )
            # Term frequency in this document.
            tf = doc.count(term)
            if tf == 0:
                continue
            # BM25 term score.
            denom = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_len if self.avg_len > 0 else 0)
            )
            score += idf * (tf * (self.k1 + 1)) / denom if denom > 0 else 0.0

        return score

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return the top-k ``(doc_idx, score)`` pairs for ``query``.

        Args:
            query: The query text.
            top_k: Maximum number of results.

        Returns:
            A list of ``(document_index, bm25_score)`` tuples sorted by
            descending score.
        """
        scored = [(i, self.score(query, i)) for i in range(self.N)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense (vector) and sparse (BM25) search.

    Scores are fused using a weighted combination.  Both score lists are
    min-max normalised before blending so they are on a comparable scale.

    Args:
        vector_store: The vector store for dense search.
        embed_fn: Callable that converts text to an embedding.
        documents: The corpus of document texts for BM25 indexing.
        alpha: Weight for the dense score in ``[0, 1]``.  The sparse
            score receives ``1 - alpha``.
        k1: BM25 term-frequency saturation.
        b: BM25 length normalisation.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embed_fn: Optional[EmbedFn] = None,
        documents: Optional[List[str]] = None,
        alpha: float = 0.5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        super().__init__(vector_store=vector_store, embed_fn=embed_fn)
        self.alpha: float = alpha
        self._bm25: Optional[_BM25] = None
        self._documents: List[str] = documents or []
        if self._documents:
            self._bm25 = _BM25(self._documents, k1=k1, b=b)

    # ------------------------------------------------------------------
    def set_documents(self, documents: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        """Set or update the BM25 corpus.

        Args:
            documents: List of document texts.
            k1: BM25 term-frequency saturation.
            b: BM25 length normalisation.
        """
        self._documents = documents
        self._bm25 = _BM25(documents, k1=k1, b=b)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve results using a hybrid of dense and sparse search.

        Args:
            query: The query text.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult` sorted by fused score.
        """
        # Dense retrieval.
        dense_results = self.vector_store.search(self._embed(query), top_k=top_k * 2)

        # Sparse (BM25) retrieval.
        bm25_scores: Dict[str, float] = {}
        if self._bm25 is not None and self._documents:
            for doc_idx, score in self._bm25.search(query, top_k=top_k * 2):
                if doc_idx < len(self._documents):
                    bm25_scores[self._documents[doc_idx]] = score

        if not bm25_scores:
            return dense_results[:top_k]

        # Normalise dense scores to [0, 1].
        dense_scores = {r.id: r.score for r in dense_results}
        dense_norm = self._normalise(dense_scores)
        bm25_norm = self._normalise(bm25_scores)

        # Build a lookup from content to SearchResult for BM25 matches.
        content_to_id: Dict[str, str] = {}
        for r in dense_results:
            if r.content:
                content_to_id[r.content] = r.id

        # Fuse scores.
        fused: Dict[str, float] = {}
        all_ids = set(dense_norm.keys())

        for content, score in bm25_norm.items():
            rid = content_to_id.get(content, content)
            all_ids.add(rid)
            fused[rid] = self.alpha * dense_norm.get(rid, 0.0) + (1 - self.alpha) * score

        for rid, score in dense_norm.items():
            fused[rid] = fused.get(rid, 0.0) + self.alpha * score

        # Sort by fused score.
        ranked_ids = sorted(fused.keys(), key=lambda rid: -fused[rid])[:top_k]

        results: List[SearchResult] = []
        for rid in ranked_ids:
            # Find the corresponding SearchResult.
            match = next((r for r in dense_results if r.id == rid), None)
            if match is not None:
                results.append(
                    SearchResult(
                        id=match.id,
                        score=fused[rid],
                        metadata=match.metadata,
                        content=match.content,
                    )
                )
            else:
                # BM25-only result (no dense match).
                results.append(
                    SearchResult(
                        id=rid,
                        score=fused[rid],
                        metadata={"source": "bm25"},
                        content=rid if rid in self._documents else "",
                    )
                )

        self._logger.debug(
            "HybridRetriever returned %d results for '%s...'.",
            len(results), query[:50],
        )
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalise a score dictionary to ``[0, 1]``.

        Args:
            scores: Mapping of id to raw score.

        Returns:
            Mapping of id to normalised score.
        """
        if not scores:
            return {}
        values = list(scores.values())
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------
class QueryRewriter:
    """Query rewriting and expansion utilities.

    Supports two strategies:

    * **HyDE** (Hypothetical Document Embeddings) -- generate a
      hypothetical answer document for the query, then retrieve using
      that document's embedding.
    * **Multi-query expansion** -- generate multiple reformulations of
      the query and merge their retrieval results.

    Args:
        model: An object with a ``generate(prompt, max_tokens) -> str``
            method (e.g. a :class:`~torcha_verse.engines.TextEngine`).
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model: Optional[Any] = model
        self._logger = get_logger("QueryRewriter")

    # ------------------------------------------------------------------
    def hyde(self, query: str, max_tokens: int = 256) -> str:
        """Generate a hypothetical document for ``query`` (HyDE).

        The hypothetical document is more likely to be semantically
        similar to the real answer, improving retrieval quality.

        Args:
            query: The original query.
            max_tokens: Maximum tokens for the generated document.

        Returns:
            The hypothetical document text.
        """
        if self.model is None:
            self._logger.warning("No model available for HyDE; returning original query.")
            return query

        prompt = (
            f"Please write a passage that answers the following question. "
            f"Write it as if it were an excerpt from a document.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        doc = self.model.generate(prompt, max_tokens=max_tokens)
        self._logger.debug("HyDE generated document (%d chars).", len(doc))
        return doc

    # ------------------------------------------------------------------
    def multi_query(
        self,
        query: str,
        n: int = 3,
        max_tokens: int = 128,
    ) -> List[str]:
        """Expand ``query`` into ``n`` reformulations.

        Args:
            query: The original query.
            n: Number of reformulations to generate.
            max_tokens: Maximum tokens per reformulation.

        Returns:
            A list of ``n`` reformulated queries (including the original).
        """
        if self.model is None:
            self._logger.warning("No model available for multi-query; returning original.")
            return [query]

        prompt = (
            f"Rewrite the following query in {n} different ways to improve "
            f"search retrieval. Put each rewrite on a separate line.\n\n"
            f"Query: {query}\n\nRewrites:"
        )
        response = self.model.generate(prompt, max_tokens=max_tokens)

        # Parse line-by-line.
        rewrites = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]
        rewrites = [r for r in rewrites if r]

        if not rewrites:
            return [query]

        # Always include the original query.
        return [query] + rewrites[:n]

    # ------------------------------------------------------------------
    def expand_and_retrieve(
        self,
        query: str,
        retriever: BaseRetriever,
        top_k: int = 5,
        n: int = 3,
    ) -> List[SearchResult]:
        """Expand the query and merge retrieval results.

        Args:
            query: The original query.
            retriever: The retriever to use.
            top_k: Final number of results.
            n: Number of query expansions.

        Returns:
            De-duplicated, merged :class:`SearchResult` list.
        """
        queries = self.multi_query(query, n=n)
        seen: Dict[str, SearchResult] = {}

        for q in queries:
            for result in retriever.retrieve(q, top_k=top_k):
                if result.id not in seen or result.score > seen[result.id].score:
                    seen[result.id] = result

        merged = sorted(seen.values(), key=lambda r: -r.score)[:top_k]
        self._logger.debug(
            "Multi-query (%d expansions) merged into %d results.", len(queries), len(merged)
        )
        return merged


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
class Reranker:
    """Simplified cross-encoder reranker.

    Re-scores candidate results by computing the cosine similarity
    between the query embedding and each candidate's embedding.  In a
    full implementation this would use a dedicated cross-encoder model;
    here we use the same embedding function as a lightweight proxy.

    Args:
        embed_fn: Callable that converts text to an embedding.
    """

    def __init__(self, embed_fn: Optional[EmbedFn] = None) -> None:
        self._embed_fn: Optional[EmbedFn] = embed_fn
        self._logger = get_logger("Reranker")

    # ------------------------------------------------------------------
    def set_embed_fn(self, embed_fn: EmbedFn) -> None:
        """Set the embedding function.

        Args:
            embed_fn: Callable that converts text to an embedding.
        """
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank ``results`` by cross-encoding similarity.

        Args:
            query: The query text.
            results: Candidate search results.
            top_k: If given, return only the top-k reranked results.

        Returns:
            Reranked :class:`SearchResult` list.
        """
        if not results:
            return []
        if self._embed_fn is None:
            self._logger.warning("No embed_fn; returning results in original order.")
            return results[:top_k] if top_k else results

        query_vec = self._embed_fn(query)
        query_vec = F.normalize(
            query_vec.unsqueeze(0) if query_vec.dim() == 1 else query_vec, p=2, dim=1
        )

        scored: List[Tuple[float, SearchResult]] = []
        for result in results:
            content = result.content or result.metadata.get("text", "")
            if not content:
                scored.append((result.score, result))
                continue
            doc_vec = self._embed_fn(content)
            doc_vec = F.normalize(
                doc_vec.unsqueeze(0) if doc_vec.dim() == 1 else doc_vec, p=2, dim=1
            )
            # Cosine similarity (vectors are normalised, so dot product).
            sim = float(torch.mm(query_vec, doc_vec.t()).item())
            scored.append((sim, result))

        scored.sort(key=lambda x: -x[0])

        reranked = [
            SearchResult(
                id=r.id,
                score=score,
                metadata=r.metadata,
                content=r.content,
            )
            for score, r in scored
        ]

        if top_k is not None:
            reranked = reranked[:top_k]

        self._logger.debug("Reranked %d results.", len(reranked))
        return reranked


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------
class ContextAssembler:
    """Assemble retrieved context into a single prompt string.

    Handles de-duplication (by content hash), truncation, and
    formatting of retrieved chunks.

    Args:
        separator: String used to join context chunks.
        max_chars: Optional maximum total character length.  Chunks are
            added in order until the limit is reached.
        include_scores: Whether to include relevance scores in the
            formatted context.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        max_chars: Optional[int] = None,
        include_scores: bool = False,
    ) -> None:
        self.separator: str = separator
        self.max_chars: Optional[int] = max_chars
        self.include_scores: bool = include_scores
        self._logger = get_logger("ContextAssembler")

    # ------------------------------------------------------------------
    def assemble(self, results: List[SearchResult]) -> str:
        """Assemble ``results`` into a context string.

        Args:
            results: Retrieved search results.

        Returns:
            A formatted context string with de-duplicated chunks.
        """
        seen: set = set()
        parts: List[str] = []
        total_len = 0

        for result in results:
            content = result.content or result.metadata.get("text", "")
            if not content:
                continue

            # De-duplicate by a content hash.
            key = hash(content.strip())
            if key in seen:
                continue
            seen.add(key)

            # Format the chunk.
            if self.include_scores:
                chunk = f"[score: {result.score:.3f}] {content}"
            else:
                chunk = content

            # Respect the character limit.
            if self.max_chars is not None:
                remaining = self.max_chars - total_len
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                total_len += len(chunk) + len(self.separator)

            parts.append(chunk)

        context = self.separator.join(parts)
        self._logger.debug(
            "Assembled context (%d chars from %d results).", len(context), len(results)
        )
        return context

    # ------------------------------------------------------------------
    def assemble_with_citations(
        self, results: List[SearchResult]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Assemble context with numbered citation markers.

        Args:
            results: Retrieved search results.

        Returns:
            A tuple ``(context_string, citation_list)`` where each
            citation is a dict with ``id``, ``source``, ``score``, and
            ``content``.
        """
        seen: set = set()
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        total_len = 0
        citation_num = 0

        for result in results:
            content = result.content or result.metadata.get("text", "")
            if not content:
                continue

            key = hash(content.strip())
            if key in seen:
                continue
            seen.add(key)

            citation_num += 1
            chunk = f"[{citation_num}] {content}"

            if self.max_chars is not None:
                remaining = self.max_chars - total_len
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                total_len += len(chunk) + len(self.separator)

            parts.append(chunk)
            citations.append(
                {
                    "id": citation_num,
                    "result_id": result.id,
                    "source": result.metadata.get("source", "unknown"),
                    "score": result.score,
                    "content": content[:200],
                }
            )

        context = self.separator.join(parts)
        return context, citations
