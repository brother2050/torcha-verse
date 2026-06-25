"""Digital-human pipeline via the L4 node layer (v0.4.x P0 real-model path).

Demonstrates ``dh_lip_sync``: re-animate a video's mouth to match a
driving audio clip.  With the project-owned
:class:`LocalTorchVideoProvider` installed as the fallback backend
the demo runs a real VideoDiT + VideoVAE forward pass instead of
the echo stub.

Run with::

    python examples/dh_lipsync.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._helpers import register_default_video_backend
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — v0.4.x P0 Digital Human (LocalTorch)")
    print("=" * 60)

    # Install the project-owned real-model backend so the
    # node exercises an actual VideoDiT + VideoVAE forward
    # pass (no echo).
    register_default_video_backend()

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
    video = out.get("video", {})
    print(f"\n[output keys] {sorted(out.keys())}")
    print(f"[video kind]  {video.get('kind', '?')}")
    print(f"[method]      {video.get('method', '?')}")
    print(f"[path]        {video.get('path', '?')}")
    print(f"[sync_score]  {out.get('sync_score', '?')}")
    # The real LocalTorchVideoProvider returns a torch.Tensor
    # for the ``frames`` key; the echo stub returns a dict.
    frames = video.get("frames")
    if hasattr(frames, "shape"):
        print(f"[frames]      tensor shape={tuple(frames.shape)}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
