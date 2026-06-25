"""Video generation via the L4 node layer.

Demonstrates ``video_txt2vid``.  Without a registered video diffusion
model the node falls back to the echo backend in :mod:`nodes._helpers`.

Run with::

    python examples/video_gen.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — L4 Video Generation (video_txt2vid)")
    print("=" * 60)

    pipeline = (
        PipelineBuilder("txt2vid_demo")
        .node(
            "video_txt2vid",
            id="vid",
            prompt="A robot walks through a neon-lit city",
            num_frames=16,
            fps=8,
            width=256,
            height=256,
            steps=6,
            guidance_scale=7.0,
            seed=11,
        )
        .build()
    )

    out = pipeline.run(NodeContext())["vid"]
    print(f"\n[output keys] {sorted(out.keys())}")
    print(f"[video kind]  {out.get('video', {}).get('kind', '?')}")
    print(f"[num_frames]  {out.get('video', {}).get('num_frames', '?')}")
    print(f"[fps]         {out.get('video', {}).get('fps', '?')}")
    print(f"[seed]        {out.get('seed')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
