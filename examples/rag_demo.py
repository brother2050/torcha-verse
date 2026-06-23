"""RAG (Retrieval-Augmented Generation) demo.

Demonstrates document ingestion, chunking, vector storage, and retrieval.

Run with::

    python examples/rag_demo.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rag.loaders.document_loader import DocumentLoaderFactory, Document
from rag.chunkers.text_chunker import RecursiveChunker
from rag.vectorstore.vector_store import InMemoryVectorStore
from rag.retrievers.retriever import VectorRetriever, ContextAssembler


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — RAG Demo")
    print("=" * 60)

    # --- 1. Create sample documents ---
    print("\n[1] Creating sample documents...")
    documents = [
        Document(
            content="TorchaVerse is a pure PyTorch framework for generative AI. "
                    "It supports text, image, audio, and video generation without "
                    "relying on high-level wrappers like Diffusers or Transformers.",
            metadata={"source": "overview.txt", "page": 1},
        ),
        Document(
            content="The framework uses a four-layer architecture: Infrastructure, "
                    "Core, Capability, and Application layers. Each layer is "
                    "independently composable based on project requirements.",
            metadata={"source": "architecture.txt", "page": 1},
        ),
        Document(
            content="The Core Layer contains ModelRegistry, TokenizerHub, "
                    "KVCacheManager, DiffusionScheduler, and MemoryManager. "
                    "These components are shared across all modalities.",
            metadata={"source": "core.txt", "page": 1},
        ),
        Document(
            content="For inference acceleration, TorchaVerse implements PagedAttention "
                    "and Continuous Batching to maximize GPU utilization, similar "
                    "to the vLLM approach.",
            metadata={"source": "inference.txt", "page": 1},
        ),
    ]
    print(f"    Created {len(documents)} documents")

    # --- 2. Chunk documents ---
    print("\n[2] Chunking documents...")
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc.content)
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        all_chunks.extend(chunks)
    print(f"    Created {len(all_chunks)} chunks")

    # --- 3. Create embeddings and store ---
    print("\n[3] Storing embeddings in vector store...")
    vector_store = InMemoryVectorStore(dim=64)
    for i, chunk in enumerate(all_chunks):
        # Use random vectors as dummy embeddings.
        embedding = torch.randn(1, 64)
        vector_store.add(embedding, [{"content": chunk.text, **chunk.metadata}])
    print(f"    Stored {vector_store.size} vectors")

    # --- 4. Retrieve relevant chunks ---
    print("\n[4] Retrieving relevant chunks...")
    retriever = VectorRetriever(vector_store=vector_store, embed_fn=lambda x: torch.randn(1, 64))
    query = "What architecture does TorchaVerse use?"
    results = retriever.retrieve(query, top_k=3)
    print(f"    Query: {query}")
    print(f"    Retrieved {len(results)} results:")
    for i, result in enumerate(results):
        print(f"      [{i+1}] Score: {result.score:.4f}")
        print(f"          Content: {result.content[:80]}...")

    # --- 5. Assemble context ---
    print("\n[5] Assembling context...")
    assembler = ContextAssembler(max_chars=500)
    context = assembler.assemble(results)
    print(f"    Context length: {len(context)} chars")
    print(f"    Context preview:\n    {context[:200]}...")

    # --- 6. Simulate RAG answer ---
    print("\n[6] Simulated RAG answer:")
    print("    Based on retrieved context:")
    print("    TorchaVerse uses a four-layer architecture: Infrastructure, Core,")
    print("    Capability, and Application layers. The Core Layer contains")
    print("    ModelRegistry, TokenizerHub, KVCacheManager, and other shared")
    print("    components for all modalities.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
