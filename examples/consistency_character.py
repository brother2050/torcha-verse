"""Consistency pipeline via the L4 node layer.

Demonstrates the consistency framework: ``character_apply`` produces an
image conditioned on a character asset, and ``character_five_view``
expands a single portrait into five canonical views.  Without a
registered image model the nodes fall back to the echo backend in
:mod:`nodes._helpers`.

Run with::

    python examples/consistency_character.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assets.base import AssetRef
from assets.types import AssetType
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
    print("TorchaVerse — L4 Consistency (character_apply + five_view)")
    print("=" * 60)

    character = _ref("char-001", AssetType.CHARACTER)

    # --- 1. character_apply ---
    print("\n[1] character_apply (512x768)...")
    p1 = (
        PipelineBuilder("character_apply_demo")
        .node(
            "character_apply",
            id="apply",
            character=character,
            prompt="一位穿着未来感外套的少女",
            width=512,
            height=768,
        )
        .build()
    )
    out1 = p1.run(NodeContext())["apply"]
    print(f"    output keys: {sorted(out1.keys())}")
    print(f"    character:   {out1.get('image', {}).get('character', '?')}")

    # --- 2. character_five_view ---
    print("\n[2] character_five_view (5 views)...")
    p2 = (
        PipelineBuilder("five_view_demo")
        .node(
            "character_five_view",
            id="fv",
            reference_image={"kind": "portrait"},
            character_name="主角-艾莉",
        )
        .build()
    )
    out2 = p2.run(NodeContext())["fv"]
    views = out2.get("five_views", [])
    print(f"    output keys: {sorted(out2.keys())}")
    print(f"    num views:   {len(views)}")
    for v in views:
        print(f"      - {v.get('view', '?')}: kind={v.get('kind', '?')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
