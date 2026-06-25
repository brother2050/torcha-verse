"""Text generation via the L4 node layer.

Demonstrates the v0.3 architecture: a Pipeline with a single
``text_chat`` node, executed through the L4 BaseNode abstraction and
``PipelineBuilder``.  No real model is registered, so the node falls back
to the echo backend in :mod:`nodes._helpers`.

Run with::

    python examples/basic_text_gen.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — L4 Text Generation (text_chat)")
    print("=" * 60)

    pipeline = (
        PipelineBuilder("text_chat_demo")
        .node(
            "text_chat",
            id="chat",
            prompt="用一句话介绍 TorchaVerse 框架",
            max_tokens=64,
            temperature=0.7,
        )
        .build()
    )

    results = pipeline.run(NodeContext())
    out = results["chat"]
    print(f"\n[output keys] {sorted(out.keys())}")
    print(f"[text]        {out.get('text', '')[:120]}")
    if "usage" in out:
        print(f"[usage]       {out['usage']}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
