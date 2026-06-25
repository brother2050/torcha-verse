"""Video generation via the L4 node layer (v0.4.x P0 real-model path).

Demonstrates ``video_txt2vid`` **with the project-owned
:class:`LocalTorchVideoProvider`** installed as the fallback
backend -- so the demo actually runs a real VideoDiT + VideoVAE
forward pass instead of the echo stub.

Run with::

    python examples/video_gen.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._helpers import register_default_video_backend
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — v0.4.x P0 Video Generation (LocalTorch)")
    print("=" * 60)

    # Install the project-owned real-model backend so the
    # node exercises an actual VideoDiT + VideoVAE forward
    # pass (no echo).
    register_default_video_backend()

    pipeline = (
        PipelineBuilder("txt2vid_demo")
        .node(
            "video_txt2vid",
            id="vid",
            prompt="A robot walks through a neon-lit city",
            num_frames=4,
            fps=8,
            width=64,
            height=64,
            steps=2,
            guidance_scale=7.0,
            seed=11,
        )
        .build()
    )

    t0 = time.time()
    out = pipeline.run(NodeContext())["vid"]
    elapsed = time.time() - t0
    print(f"\n[output keys] {sorted(out.keys())}")
    print(f"[video kind]  {out.get('video', {}).get('kind', '?')}")
    print(f"[num_frames]  {out.get('video', {}).get('num_frames', '?')}")
    print(f"[fps]         {out.get('video', {}).get('fps', '?')}")
    print(f"[seed]        {out.get('seed')}")
    print(f"[elapsed]     {elapsed:.2f}s")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
