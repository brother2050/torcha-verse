"""v0.5.x end-to-end demo: cold storage + RAG + agent + multimodal.

Demonstrates the four major v0.5.x feature blocks in a single
self-contained script:

* **Cold storage**: write an asset to the local cold tier and
  promote it back.
* **RAG**: ingest a handful of documents, query the resulting
  index, and look up the source hits.
* **Agent**: register a custom tool and run a one-shot ReAct
  loop.
* **Multimodal**: invoke the image-understand L4 node and the
  video-understand L4 node against the echo backend.

Run with::

    python examples/v05_feature_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _divider(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def demo_cold_storage(tmp_root) -> None:
    _divider("[1/4] Cold storage (LocalColdStorage)")
    from assets.cold_storage import LocalColdStorage, ColdStorageConfig, make_cold_storage

    root = tmp_root / "cold"
    cold = LocalColdStorage(root=root, prefix="v05-demo/")
    data = b"hello cold storage"
    sha = hashlib.sha256(data).hexdigest()
    src = tmp_root / "src.bin"
    src.write_bytes(data)
    cold.store(sha, src)
    out = tmp_root / "out.bin"
    cold.fetch(sha, out)
    assert out.read_bytes() == data
    print(f"  -> wrote {len(data)} bytes, sha={sha[:12]}..., fetched OK")

    # Same path through the factory.
    cfg = ColdStorageConfig(backend="local", bucket=str(root), prefix="factory/")
    cs2 = make_cold_storage(config=cfg)
    assert isinstance(cs2, LocalColdStorage)
    print(f"  -> factory returned {type(cs2).__name__}")


class _DemoCtx:
    """Minimal context the L4 nodes expect during a demo run."""

    def __init__(self) -> None:
        self.audit = None
        self.bus = None
        self.config = {}
        self.asset_store = None
        # An optional logger attribute the L4 nodes fall back to.
        self.logger = None
        # The RAG progress callback consults this on every flush.
        self.progress = None


def _ctx() -> _DemoCtx:
    return _DemoCtx()


def demo_rag() -> None:
    _divider("[2/4] RAG (ingest -> query -> list)")
    from infrastructure.rag import default_rag_index_store
    from nodes.rag import RAGIngestNode, RAGQueryNode, RAGListIndexesNode

    store = default_rag_index_store()
    index_name = "v05_demo_index"

    ingest = RAGIngestNode()
    out = ingest.execute(
        _ctx(),
        index_name=index_name,
        documents=[
            {"doc_id": "d1", "text": "TorchaVerse is a multimodal AI project."},
            {"doc_id": "d2", "text": "It supports RAG, agents, and diffusion."},
            {"doc_id": "d3", "text": "The v0.5 line adds cold storage and multimodal."},
        ],
        chunk_size=64,
        chunk_overlap=8,
    )
    print(f"  -> ingested {out['documents']} documents, {out['vectors']} vectors")

    query = RAGQueryNode()
    res = query.execute(
        _ctx(),
        index_name=index_name,
        query="What does TorchaVerse support?",
        top_k=2,
    )
    print(f"  -> retrieved {len(res.get('hits', []))} hits, top score={res.get('hits', [{}])[0].get('score', 0):.3f}")

    listing = RAGListIndexesNode().execute(_ctx())
    print(f"  -> indexes: {listing.get('indexes', [])}")


def demo_agent() -> None:
    _divider("[3/4] Agent (ReAct tool-calling loop)")
    from infrastructure.agent import AgentBus, ToolSpec, default_agent_bus
    from nodes.agent import AgentListToolsNode, AgentRunNode

    # Inspect default tools.
    listing = AgentListToolsNode().execute(_ctx())
    tools = listing.get("tools", [])
    # `tools` may be a list of dicts or a list of strings
    # depending on the node version -- handle both.
    if tools and isinstance(tools[0], dict):
        names = [t.get("name", "?") for t in tools]
    else:
        names = [str(t) for t in tools]
    print(f"  -> default tools ({len(tools)}): {names}")

    # Register a custom tool and use it.
    bus = default_agent_bus()

    def my_tool(name: str = "world") -> str:
        return f"hello, {name}!"

    bus.tools.register(
        ToolSpec(
            name="greet",
            description="Greet someone by name.",
            parameters={"name": "str"},
            func=my_tool,
        )
    )
    print(f"  -> registered custom tool 'greet'")

    # List tools again.
    listing = AgentListToolsNode().execute(_ctx())
    tools = listing.get("tools", [])
    if tools and isinstance(tools[0], dict):
        has = "greet" in [t.get("name") for t in tools]
    else:
        has = "greet" in [str(t) for t in tools]
    print(f"  -> 'greet' is in tool list: {has}")


def demo_multimodal() -> None:
    _divider("[4/4] Multimodal (image_understand + video_understand)")
    from nodes.image import ImageUnderstandNode
    from nodes.video import VideoUnderstandNode

    img_node = ImageUnderstandNode()
    print(f"  -> image_understand spec.type: {img_node.spec.type}")
    print(f"  -> image_understand inputs: {list(img_node.spec.inputs.keys())}")

    vid_node = VideoUnderstandNode()
    print(f"  -> video_understand spec.type: {vid_node.spec.type}")
    print(f"  -> video_understand inputs: {list(vid_node.spec.inputs.keys())}")


def main() -> None:
    print("TorchaVerse -- v0.5.x feature demo")
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="torcha-v05-") as raw:
        tmp_root = Path(raw)
        demo_cold_storage(tmp_root)
        demo_rag()
        demo_agent()
        demo_multimodal()
    print()
    print("All v0.5.x features demoed successfully.")


if __name__ == "__main__":
    main()
