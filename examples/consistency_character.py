"""Consistency pipeline via the L4 node layer (v0.4.x P0 real-model path).

Demonstrates the consistency framework: ``character_apply`` produces an
image conditioned on a character asset, and ``character_five_view``
expands a single portrait into five canonical views.  With the
project-owned :class:`LocalTorchImageProvider` installed as the
fallback backend the demo runs a real UNet + VAE forward pass
instead of the echo stub.

Run with::

    python examples/consistency_character.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assets.base import AssetRef
from assets.types import AssetType
from nodes._helpers import register_default_image_backend
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def _ref(asset_id: str, asset_type: AssetType) -> AssetRef:
    return AssetRef(
        asset_id=asset_id,
        asset_type=asset_type,
        revision="r1",
        content_hash="0" * 64,
    )


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — v0.4.x P0 Consistency (LocalTorch)")
    print("=" * 60)

    # Install the project-owned real-model backend so the
    # node exercises an actual UNet + VAE forward pass.
    register_default_image_backend()

    character = _ref("char-001", AssetType.CHARACTER)

    # --- 1. character_apply ---
    # 64x96 keeps the LocalTorch TINY UNet+VAE forward cheap on CI.
    print("\n[1] character_apply (64x96)...")
    p1 = (
        PipelineBuilder("character_apply_demo")
        .node(
            "character_apply",
            id="apply",
            character=character,
            prompt="一位穿着未来感外套的少女",
            width=64,
            height=96,
        )
        .build()
    )
    out1 = p1.run(NodeContext())["apply"]
    img1 = out1.get("image", {})
    print(f"    output keys: {sorted(out1.keys())}")
    if isinstance(img1, dict):
        print(f"    character:   {img1.get('character', '?')}")
    else:
        # Real backend: image is a torch.Tensor.
        print(f"    image tensor: shape={tuple(img1.shape)}")
    print(f"    seed:         {out1.get('seed')}")
    print(f"    steps:        {out1.get('steps')}")

    # --- 2. character_five_view ---
    # 64x64 keeps 5-view cost in check on CI (5 forward passes
    # in the LocalTorch TINY preset).
    print("\n[2] character_five_view (5 views, 64x64)...")
    p2 = (
        PipelineBuilder("five_view_demo")
        .node(
            "character_five_view",
            id="fv",
            reference_image={"kind": "portrait"},
            character_name="主角-艾莉",
            width=64,
            height=64,
        )
        .build()
    )
    out2 = p2.run(NodeContext())["fv"]
    views = out2.get("five_views", [])
    print(f"    output keys: {sorted(out2.keys())}")
    print(f"    num views:   {len(views)}")
    for v in views:
        result = v.get("result", {})
        image = result.get("image", None) if isinstance(result, dict) else None
        if isinstance(image, dict):
            print(f"      - {v.get('view', '?')}: kind={image.get('kind', '?')}")
        elif image is not None:
            # Real backend: image is a torch.Tensor.
            print(f"      - {v.get('view', '?'):14s} tensor shape={tuple(image.shape)}")
        else:
            print(f"      - {v.get('view', '?')}: (no image)")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
