"""Retrieval-Augmented Generation (RAG) engine for TorchaVerse.

This module provides :class:`RAGEngine`, the capability-layer entry point
for retrieval-augmented generation.  It composes an indexing pipeline
(document loading + chunking + embedding + storage), a retrieval pipeline
(query embedding + similarity search + optional reranking), and a
generation pipeline (context-augmented text generation with citation
extraction).

Because the ``rag/`` sub-packages (``loaders``, ``chunkers``,
``vectorstore``, ``retrievers``) are currently empty stubs, the
supporting classes are implemented directly in this module.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger
from .text_engine import TextEngine

__all__ = [
    "Document",
    "Chunk",
    "DocumentLoader",
    "TextChunker",
    "VectorStore",
    "Retriever",
    "CitationExtractor",
    "Answer",
    "Sources",
    "RAGEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """A loaded document.

    Attributes:
        content: The full text content.
        metadata: Optional metadata (source path, title, etc.).
        doc_id: Unique document identifier.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A text chunk extracted from a document.

    Attributes:
        text: The chunk text.
        embedding: Dense embedding vector.
        doc_id: Parent document id.
        chunk_id: Unique chunk identifier.
        metadata: Optional metadata.
        score: Retrieval score (populated during search).
    """

    text: str
    embedding: Optional[torch.Tensor] = None
    doc_id: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class Sources:
    """Retrieved source chunks for a RAG answer.

    Attributes:
        chunks: List of source :class:`Chunk` objects.
    """

    chunks: List[Chunk] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialise to a list of dictionaries."""
        return [
            {
                "text": c.text[:200],
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]


@dataclass
class Answer:
    """A RAG answer with citations.

    Attributes:
        text: The generated answer text.
        sources: The source chunks used.
        confidence: Confidence score in ``[0, 1]``.
    """

    text: str = ""
    sources: Sources = field(default_factory=Sources)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------
class DocumentLoader:
    """Load documents from files or directories.

    Supports plain text (``.txt``), Markdown (``.md``), and raw strings.

    Args:
        encoding: File encoding.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".txt", ".md", ".text", "")

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding: str = encoding
        self._logger = get_logger("DocumentLoader")

    def load(self, source: Union[str, os.PathLike, List[str]]) -> List[Document]:
        """Load documents from a path, list of paths, or raw text.

        Args:
            source: A file path, directory path, list of paths, or
                raw text string.

        Returns:
            A list of :class:`Document` objects.
        """
        if isinstance(source, list):
            return self._load_from_list(source)
        if isinstance(source, str) and os.path.isdir(source):
            return self._load_from_directory(source)
        if isinstance(source, str) and os.path.isfile(source):
            return self._load_from_file(source)
        # Treat as raw text.
        return [Document(content=source, metadata={"source": "raw_text"})]

    def _load_from_list(self, paths: List[str]) -> List[Document]:
        """Load documents from a list of file paths or strings."""
        docs: List[Document] = []
        for p in paths:
            if os.path.isfile(p):
                docs.extend(self._load_from_file(p))
            else:
                docs.append(Document(content=p, metadata={"source": "raw_text"}))
        return docs

    def _load_from_directory(self, dir_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        docs: List[Document] = []
        for root, _dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                if any(fname.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    fpath = os.path.join(root, fname)
                    docs.extend(self._load_from_file(fpath))
        self._logger.info("Loaded %d documents from '%s'.", len(docs), dir_path)
        return docs

    def _load_from_file(self, file_path: str) -> List[Document]:
        """Load a single file."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            self._logger.warning("Unsupported file type '%s'; skipping.", file_path)
            return []

        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as exc:
            self._logger.error("Failed to read '%s': %s", file_path, exc)
            return []

        return [
            Document(
                content=content,
                metadata={"source": file_path, "filename": os.path.basename(file_path)},
            )
        ]


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------
class TextChunker:
    """Split documents into overlapping text chunks.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        separator: Default separator for splitting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separator: str = "\n\n",
    ) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.separator: str = separator

    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into overlapping chunks.

        Args:
            document: The document to chunk.

        Returns:
            A list of :class:`Chunk` objects.
        """
        text = document.content
        if not text.strip():
            return []

        # Split by separator first, then by chunk_size.
        segments = text.split(self.separator)
        chunks: List[Chunk] = []
        current: str = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # If adding this segment exceeds chunk_size, flush current.
            if current and len(current) + len(seg) + 2 > self.chunk_size:
                chunks.append(self._make_chunk(current, document))
                # Start new chunk with overlap.
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap:] + self.separator + seg
                else:
                    current = seg
            else:
                current = f"{current}{self.separator}{seg}" if current else seg

            # Split very long segments.
            while len(current) > self.chunk_size:
                chunk_text = current[: self.chunk_size]
                chunks.append(self._make_chunk(chunk_text, document))
                current = current[self.chunk_size - self.chunk_overlap:]

        if current.strip():
            chunks.append(self._make_chunk(current, document))

        return chunks

    def _make_chunk(self, text: str, document: Document) -> Chunk:
        """Create a :class:`Chunk` from text and document metadata."""
        return Chunk(
            text=text.strip(),
            doc_id=document.doc_id,
            metadata={**document.metadata, "chunk_index": 0},
        )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------
class VectorStore:
    """In-memory vector store for chunk embeddings.

    Stores chunk embeddings and supports cosine-similarity search.

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim: int = dim
        self._chunks: List[Chunk] = []
        self._embeddings: List[torch.Tensor] = []
        self._logger = get_logger("VectorStore")

    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks with pre-computed embeddings.

        Args:
            chunks: Chunks with ``.embedding`` set.
        """
        for chunk in chunks:
            if chunk.embedding is None:
                self._logger.warning(
                    "Chunk %s has no embedding; skipping.", chunk.chunk_id
                )
                continue
            self._chunks.append(chunk)
            self._embeddings.append(chunk.embedding)

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Chunk]:
        """Search for the most similar chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.

        Returns:
            A list of :class:`Chunk` objects sorted by similarity.
        """
        if not self._embeddings:
            return []

        # Stack all embeddings.
        matrix = torch.stack(self._embeddings)  # (N, dim)
        query = query_embedding.unsqueeze(0)  # (1, dim)

        # Ensure compatible dimensions.
        if matrix.shape[1] != query.shape[1]:
            min_dim = min(matrix.shape[1], query.shape[1])
            matrix = matrix[:, :min_dim]
            query = query[:, :min_dim]

        # Cosine similarity.
        sims = F.cosine_similarity(matrix, query, dim=-1)  # (N,)
        top_k = min(top_k, len(self._chunks))

        top_scores, top_indices = torch.topk(sims, top_k)

        results: List[Chunk] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            chunk = self._chunks[idx]
            chunk.score = score
            results.append(chunk)

        return results

    @property
    def size(self) -> int:
        """Number of stored chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all stored chunks."""
        self._chunks.clear()
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """Retrieve relevant chunks for a query.

    Composes a :class:`TextEngine` (for query embedding) and a
    :class:`VectorStore` (for similarity search).

    Args:
        vector_store: The vector store to search.
        text_engine: The text engine for query embedding.
        rerank: Whether to apply a simple reranking step.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        text_engine: TextEngine,
        rerank: bool = False,
    ) -> None:
        self.vector_store: VectorStore = vector_store
        self.text_engine: TextEngine = text_engine
        self.rerank: bool = rerank
        self._logger = get_logger("Retriever")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: The query text.
            top_k: Number of chunks to retrieve.

        Returns:
            A list of :class:`Chunk` objects.
        """
        # Embed the query.
        query_embedding = self.text_engine.embed(query)

        # Search the vector store.
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Optional reranking: boost chunks that share keywords with the query.
        if self.rerank and results:
            results = self._rerank(query, results)

        self._logger.debug(
            "Retrieved %d chunks for query '%s...'.", len(results), query[:50]
        )
        return results

    def _rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """Apply a simple keyword-overlap reranking.

        Args:
            query: The query text.
            chunks: Initial retrieval results.

        Returns:
            Reranked chunks.
        """
        query_words = set(query.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            # Blend original score with keyword overlap.
            chunk.score = 0.7 * chunk.score + 0.3 * (overlap / max(len(query_words), 1))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks


# ---------------------------------------------------------------------------
# CitationExtractor
# ---------------------------------------------------------------------------
class CitationExtractor:
    """Extract and format citations from retrieved chunks.

    Inserts inline citation markers (``[1]``, ``[2]``, ...) into the
    generated answer and appends a sources list.
    """

    # Patterns that indicate a factual claim worth citing.
    CLAIM_PATTERNS: List[str] = [
        r"according to",
        r"based on",
        r"the (?:document|text|source) (?:states|mentions|says)",
        r"(?:figure|table|section) \d+",
    ]

    def extract(
        self,
        answer: str,
        sources: List[Chunk],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Add citation markers to the answer.

        Args:
            answer: The generated answer text.
            sources: The source chunks used.

        Returns:
            A tuple ``(cited_answer, citation_list)``.
        """
        if not sources:
            return answer, []

        citations: List[Dict[str, Any]] = []
        cited_answer = answer

        for i, chunk in enumerate(sources, 1):
            # Find sentences in the answer that overlap with the chunk.
            answer_sentences = re.split(r"(?<=[.!?])\s+", cited_answer)
            chunk_words = set(chunk.text.lower().split())

            for j, sentence in enumerate(answer_sentences):
                sent_words = set(sentence.lower().split())
                overlap = len(sent_words & chunk_words)
                if overlap > 3 and f"[{i}]" not in sentence:
                    # Insert citation marker at the end of the sentence.
                    answer_sentences[j] = f"{sentence.rstrip()} [{i}]"
                    break

            cited_answer = " ".join(answer_sentences)

            citations.append(
                {
                    "id": i,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:200],
                    "score": chunk.score,
                    "source": chunk.metadata.get("source", "unknown"),
                }
            )

        # Append sources list.
        if citations:
            cited_answer += "\n\n--- Sources ---\n"
            for c in citations:
                cited_answer += f"[{c['id']}] {c['source']} (score: {c['score']:.3f})\n"

        return cited_answer, citations


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------
class RAGEngine:
    """Retrieval-Augmented Generation engine.

    Composes an indexing pipeline (loader + chunker + embedder + store),
    a retrieval pipeline (retriever + optional reranker), and a
    generation pipeline (text engine + citation extractor).

    Args:
        text_engine: The text engine for embedding and generation.
        config: Optional configuration dictionary.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        device: Optional device override.
    """

    def __init__(
        self,
        text_engine: Optional[TextEngine] = None,
        config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger("RAGEngine")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        # Text engine for embedding and generation.
        self.text_engine: TextEngine = text_engine or TextEngine(
            "default", device=self._device
        )

        # Pipeline components.
        rag_cfg = self._cfg_manager.get("rag", {})
        self.loader: DocumentLoader = DocumentLoader(
            encoding=rag_cfg.get("encoding", "utf-8")
        )
        self.chunker: TextChunker = TextChunker(
            chunk_size=rag_cfg.get("chunk_size", chunk_size),
            chunk_overlap=rag_cfg.get("chunk_overlap", chunk_overlap),
            separator=rag_cfg.get("separator", "\n\n"),
        )
        embed_dim = rag_cfg.get("embedding_dim", 768)
        self.vector_store: VectorStore = VectorStore(dim=embed_dim)
        self.retriever: Retriever = Retriever(
            vector_store=self.vector_store,
            text_engine=self.text_engine,
            rerank=rag_cfg.get("rerank", False),
        )
        self.citation_extractor: CitationExtractor = CitationExtractor()

        # Conversation history for multi-turn RAG.
        self._history: List[Dict[str, str]] = []

        self._logger.info("RAGEngine initialised.")

    # ------------------------------------------------------------------
    # Indexing pipeline
    # ------------------------------------------------------------------
    def ingest(self, documents_or_paths: Union[str, List[str], Document, List[Document]]) -> None:
        """Ingest documents into the vector store.

        Loads, chunks, embeds, and stores documents.

        Args:
            documents_or_paths: File paths, directories, raw text,
                or :class:`Document` objects.
        """
        # Step 1: Load documents.
        if isinstance(documents_or_paths, Document):
            documents = [documents_or_paths]
        elif isinstance(documents_or_paths, list) and all(
            isinstance(d, Document) for d in documents_or_paths
        ):
            documents = documents_or_paths  # type: ignore[assignment]
        else:
            documents = self.loader.load(documents_or_paths)  # type: ignore[arg-type]

        if not documents:
            self._logger.warning("No documents to ingest.")
            return

        # Step 2: Chunk documents.
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        self._logger.info("Chunked %d documents into %d chunks.", len(documents), len(all_chunks))

        # Step 3: Embed chunks.
        for chunk in all_chunks:
            chunk.embedding = self.text_engine.embed(chunk.text)

        # Step 4: Store.
        self.vector_store.add(all_chunks)

        self._logger.info("Ingested %d chunks (total: %d).", len(all_chunks), self.vector_store.size)

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> Tuple[Answer, Sources]:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The question to answer.
            top_k: Number of chunks to retrieve.
            rerank: Whether to apply reranking.

        Returns:
            A tuple ``(answer, sources)``.
        """
        # Step 1: Retrieve relevant chunks.
        self.retriever.rerank = rerank
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            # No context available; generate without retrieval.
            answer_text = self.text_engine.generate(
                f"Question: {question}\nAnswer:", max_tokens=256
            )
            return Answer(text=answer_text, sources=Sources(), confidence=0.0), Sources()

        # Step 2: Build context-augmented prompt.
        context = "\n\n".join(
            f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        prompt = (
            f"Use the following sources to answer the question.\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # Step 3: Generate answer.
        answer_text = self.text_engine.generate(prompt, max_tokens=256, temperature=0.3)

        # Step 4: Extract citations.
        cited_text, citations = self.citation_extractor.extract(answer_text, chunks)

        # Step 5: Compute confidence.
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

        answer = Answer(
            text=cited_text,
            sources=Sources(chunks=chunks),
            confidence=avg_score,
        )

        return answer, Sources(chunks=chunks)

    # ------------------------------------------------------------------
    # Multi-turn chat
    # ------------------------------------------------------------------
    def chat(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
    ) -> Tuple[Answer, Sources]:
        """Multi-turn RAG chat.

        Maintains conversation history and incorporates it into the
        retrieval and generation prompts.

        Args:
            question: The user's question.
            conversation_history: Optional list of ``{"role": ..., "content": ...}``
                dicts.  When ``None`` the internal history is used.
            top_k: Number of chunks to retrieve.

        Returns:
            A tuple ``(answer, sources)``.
        """
        history = conversation_history or self._history

        # Build a context-aware query from history.
        if history:
            recent = history[-3:]  # Last 3 turns.
            context_query = " ".join(m["content"] for m in recent) + " " + question
        else:
            context_query = question

        # Retrieve.
        chunks = self.retriever.retrieve(context_query, top_k=top_k)

        # Build prompt with history and retrieved context.
        context_text = "\n\n".join(
            f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        history_text = "\n".join(
            f"[{m['role'].upper()}] {m['content']}" for m in history
        )

        prompt = (
            f"Sources:\n{context_text}\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"[USER] {question}\n"
            f"[ASSISTANT]"
        ) if history else (
            f"Sources:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

        # Generate.
        answer_text = self.text_engine.generate(prompt, max_tokens=256, temperature=0.3)

        # Extract citations.
        cited_text, _citations = self.citation_extractor.extract(answer_text, chunks)

        # Compute confidence.
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

        answer = Answer(
            text=cited_text,
            sources=Sources(chunks=chunks),
            confidence=avg_score,
        )

        # Update history.
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer_text})

        return answer, Sources(chunks=chunks)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.clear()
        self._logger.info("Index cleared.")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def index_size(self) -> int:
        """Number of chunks in the index."""
        return self.vector_store.size

    def __repr__(self) -> str:
        return f"RAGEngine(index_size={self.index_size}, device={self._device})"
