"""Image generation via the L4 node layer.

Demonstrates ``image_txt2img`` (single node) and the
``image_txt2img -> image_upscale`` chained pipeline.  Without a
registered model the nodes use the echo backend in :mod:`nodes._helpers`.

Run with::

    python examples/image_gen.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — L4 Image Generation (image_txt2img + upscale)")
    print("=" * 60)

    # --- 1. single node ---
    print("\n[1] image_txt2img (512x512)...")
    p1 = (
        PipelineBuilder("txt2img_demo")
        .node(
            "image_txt2img",
            id="gen",
            prompt="a cat playing piano, cyberpunk style",
            width=512,
            height=512,
            steps=8,
            guidance_scale=7.0,
            seed=42,
        )
        .build()
    )
    out1 = p1.run(NodeContext())["gen"]
    print(f"    output keys: {sorted(out1.keys())}")
    print(f"    seed:        {out1.get('seed')}")
    print(f"    image kind:  {out1.get('image', {}).get('kind', '?')}")

    # --- 2. chained pipeline ---
    print("\n[2] image_txt2img -> image_upscale (2x)...")
    p2 = (
        PipelineBuilder("txt2img_upscale_demo")
        .node(
            "image_txt2img",
            id="gen",
            prompt="雪山日落",
            width=512,
            height=512,
            steps=6,
            guidance_scale=7.5,
            seed=7,
        )
        .node("image_upscale", id="up", scale=2)
        .connect("gen", "up", output_key="image", input_key="image")
        .build()
    )
    out2 = p2.run(NodeContext())
    print(f"    gen output keys: {sorted(out2['gen'].keys())}")
    print(f"    up  output keys: {sorted(out2['up'].keys())}")
    print(f"    up  scale:       {out2['up'].get('image', {}).get('scale')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
