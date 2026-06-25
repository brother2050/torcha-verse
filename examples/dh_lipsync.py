"""Digital-human pipeline via the L4 node layer.

Demonstrates ``dh_lip_sync``: re-animate a video's mouth to match a
driving audio clip.  Without a registered model the node falls back to
the echo backend in :mod:`nodes._helpers`.

Run with::

    python examples/dh_lipsync.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — L4 Digital Human (dh_lip_sync)")
    print("=" * 60)

    pipeline = (
        PipelineBuilder("lipsync_demo")
        .node(
            "dh_lip_sync",
            id="lipsync",
            video={"kind": "source_video", "duration_s": 3.0, "fps": 24},
            audio={"kind": "source_audio", "duration_s": 3.0, "sample_rate": 22050},
            method="musetalk",
        )
        .build()
    )

    out = pipeline.run(NodeContext())["lipsync"]
    print(f"\n[output keys] {sorted(out.keys())}")
    print(f"[video kind]  {out.get('video', {}).get('kind', '?')}")
    print(f"[method]      {out.get('video', {}).get('method', '?')}")
    print(f"[path]        {out.get('video', {}).get('path', '?')}")
    print(f"[sync_score]  {out.get('sync_score', '?')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
