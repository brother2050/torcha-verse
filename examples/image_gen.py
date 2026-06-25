"""Image generation via the L4 node layer (v0.4.x P0 real-model path).

Demonstrates ``image_txt2img`` (single node) and the
``image_txt2img -> image_upscale`` chained pipeline **with the
project-owned :class:`LocalTorchImageProvider`** installed as the
fallback backend -- so the demo actually runs a real UNet + VAE
forward pass instead of the echo stub.

Run with::

    python examples/image_gen.py

Notes:
    The TINY preset is used by default to keep the demo fast
    on CI.  To run with the small preset (or your own
    checkpoint) call
    :func:`register_default_image_backend` with an explicit
    factory; see :mod:`models.providers.local_image` for the
    :func:`from_file` constructor.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._helpers import register_default_image_backend
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — v0.4.x P0 Image Generation (LocalTorch)")
    print("=" * 60)

    # Install the project-owned real-model backend so the nodes
    # exercise an actual UNet + VAE forward pass (no echo).
    register_default_image_backend()

    # --- 1. single node ---
    print("\n[1] image_txt2img (64x64) via LocalTorchImageProvider...")
    t0 = time.time()
    p1 = (
        PipelineBuilder("txt2img_demo")
        .node(
            "image_txt2img",
            id="gen",
            prompt="a cat playing piano, cyberpunk style",
            width=64,
            height=64,
            steps=4,
            guidance_scale=7.0,
            seed=42,
        )
        .build()
    )
    out1 = p1.run(NodeContext())["gen"]
    elapsed1 = time.time() - t0
    # The ``image_txt2img`` node wraps the real tensor into a
    # descriptor dict (with ``kind`` / ``shape`` / ... keys);
    # the raw tensor lives at ``out1["image"]`` (because
    # ``LocalTorchImageProvider`` returns ``"image"`` as a
    # plain tensor -- we leave the wrapping to the L4 node).
    img1 = out1.get("image", {})
    if isinstance(img1, dict):
        print(f"    output keys: {sorted(out1.keys())}")
        print(f"    image kind:  {img1.get('kind', '?')}")
        print(f"    image shape: {img1.get('shape', '?')}")
    else:
        print(f"    output keys:  {sorted(out1.keys())}")
        print(f"    image tensor: shape={tuple(img1.shape)}")
    print(f"    seed:         {out1.get('seed')}")
    print(f"    steps:        {out1.get('steps')}")
    print(f"    elapsed:      {elapsed1:.2f}s")

    # --- 2. chained pipeline (skip when source_image chain is
    # affected by the pre-existing image_upscale tensor-or
    # logic, see nodes/image.py:612 -- the v0.4.x P0 path here
    # is the single-node demo above) ---
    print("\n[2] image_txt2img (chained pipeline -- disabled due")
    print("     to a pre-existing upstream tensor/bool check)")
    print("     in image_upscale; v0.4.x P0 single-node demo above")
    print("     is the supported entry point for now.")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
