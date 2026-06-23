"""Retrieval-Augmented Generation (RAG) subsystem for TorchaVerse.

This package provides the building blocks for a RAG pipeline:

* :mod:`rag.loaders` -- document loaders for text, PDF, HTML, Markdown.
* :mod:`rag.chunkers` -- text chunking strategies.
* :mod:`rag.vectorstore` -- vector storage with similarity search.
* :mod:`rag.retrievers` -- retrieval, reranking, and context assembly.
"""

from __future__ import annotations

from .chunkers import (
    BaseChunker,
    Chunk,
    FixedLengthChunker,
    RecursiveChunker,
    SemanticChunker,
)
from .loaders import (
    BaseDocumentLoader,
    Document,
    DocumentLoaderFactory,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextFileLoader,
)
from .retrievers import (
    BaseRetriever,
    ContextAssembler,
    HybridRetriever,
    QueryRewriter,
    Reranker,
    VectorRetriever,
)
from .vectorstore import (
    BaseVectorStore,
    FaissVectorStore,
    InMemoryVectorStore,
    SearchResult,
)

__all__ = [
    # loaders
    "BaseDocumentLoader",
    "Document",
    "DocumentLoaderFactory",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextFileLoader",
    # chunkers
    "BaseChunker",
    "Chunk",
    "FixedLengthChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # vectorstore
    "BaseVectorStore",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "SearchResult",
    # retrievers
    "BaseRetriever",
    "ContextAssembler",
    "HybridRetriever",
    "QueryRewriter",
    "Reranker",
    "VectorRetriever",
]
