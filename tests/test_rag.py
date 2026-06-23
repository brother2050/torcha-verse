"""Tests for the RAG subsystem."""

from __future__ import annotations

import os
import tempfile
import pytest
import torch

from rag.loaders.document_loader import (
    DocumentLoaderFactory, TextFileLoader, MarkdownLoader, Document,
)
from rag.chunkers.text_chunker import (
    FixedLengthChunker, SemanticChunker, RecursiveChunker, Chunk,
)
from rag.vectorstore.vector_store import InMemoryVectorStore, SearchResult
from rag.retrievers.retriever import (
    VectorRetriever, ContextAssembler, Reranker,
)


class TestDocumentLoader:
    """Test document loaders."""

    def test_text_loader(self, tmp_path):
        """TextFileLoader loads .txt files."""
        path = tmp_path / "test.txt"
        path.write_text("Hello World\nThis is a test.")
        loader = TextFileLoader()
        docs = loader.load(str(path))
        assert len(docs) == 1
        assert "Hello World" in docs[0].content

    def test_markdown_loader(self, tmp_path):
        """MarkdownLoader loads .md files and strips syntax."""
        path = tmp_path / "test.md"
        path.write_text("# Title\n\n**Bold** text and [link](http://x.com).")
        loader = MarkdownLoader()
        docs = loader.load(str(path))
        assert len(docs) == 1
        assert "Title" in docs[0].content

    def test_factory(self, tmp_path):
        """DocumentLoaderFactory selects correct loader by extension."""
        path = tmp_path / "test.txt"
        path.write_text("content")
        loader = DocumentLoaderFactory.create_loader(str(path))
        assert loader is not None


class TestTextChunker:
    """Test text chunkers."""

    def test_fixed_length(self):
        """FixedLengthChunker splits by character count."""
        chunker = FixedLengthChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk("A" * 50)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 20

    def test_semantic(self):
        """SemanticChunker splits by sentences."""
        chunker = SemanticChunker(chunk_size=30, overlap=5)
        text = "First sentence here. Second sentence. Third one is longer than expected."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_recursive(self):
        """RecursiveChunker splits hierarchically."""
        chunker = RecursiveChunker(chunk_size=25, overlap=5)
        text = "Para one.\n\nPara two.\n\nPara three with more text."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestVectorStore:
    """Test vector store."""

    def test_add_and_search(self):
        """InMemoryVectorStore add and search."""
        store = InMemoryVectorStore(dim=8)
        vectors = torch.stack([torch.randn(8) for _ in range(5)])
        metadata = [{"content": f"doc {i}"} for i in range(5)]
        store.add(vectors, metadata)
        assert store.size == 5

        query = vectors[0]
        results = store.search(query, top_k=3)
        assert len(results) == 3
        assert results[0].score >= results[1].score  # sorted by score

    def test_clear(self):
        """clear() removes all entries."""
        store = InMemoryVectorStore(dim=4)
        vec = torch.randn(1, 4)
        store.add(vec, [{"content": "x"}])
        store.clear()
        assert store.size == 0


class TestRetriever:
    """Test retriever."""

    def test_vector_retriever(self):
        """VectorRetriever retrieves relevant chunks."""
        store = InMemoryVectorStore(dim=8)
        vectors = torch.stack([torch.randn(8) for _ in range(3)])
        metadata = [{"content": f"doc {i}"} for i in range(3)]
        store.add(vectors, metadata)

        retriever = VectorRetriever(
            vector_store=store,
            embed_fn=lambda x: torch.randn(8),
        )
        results = retriever.retrieve("query", top_k=2)
        assert len(results) == 2

    def test_context_assembler(self):
        """ContextAssembler assembles context from results."""
        results = [
            SearchResult(id=0, score=0.9, metadata={}, content="First chunk."),
            SearchResult(id=1, score=0.8, metadata={}, content="Second chunk."),
        ]
        assembler = ContextAssembler(max_chars=100)
        context = assembler.assemble(results)
        assert "First chunk" in context
        assert "Second chunk" in context

    def test_reranker(self):
        """Reranker reorders results with embed_fn."""
        reranker = Reranker(embed_fn=lambda x: torch.randn(8))
        results = [
            SearchResult(id=0, score=0.5, metadata={}, content="apple"),
            SearchResult(id=1, score=0.9, metadata={}, content="banana"),
        ]
        reranked = reranker.rerank("banana", results)
        assert len(reranked) == 2
