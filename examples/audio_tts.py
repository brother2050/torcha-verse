"""Text-to-speech via the L4 node layer.

Demonstrates the ``audio_tts`` node.  Without a registered TTS model
the node falls back to the echo backend in :mod:`nodes._helpers` and
returns a stub payload.

Run with::

    python examples/audio_tts.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — L4 Audio TTS (audio_tts)")
    print("=" * 60)

    pipeline = (
        PipelineBuilder("tts_demo")
        .node(
            "audio_tts",
            id="tts",
            text="Hello from TorchaVerse!",
            voice="default",
            language="en",
            speed=1.0,
            sample_rate=22050,
        )
        .build()
    )

    out = pipeline.run(NodeContext())["tts"]
    print(f"\n[output keys]  {sorted(out.keys())}")
    print(f"[audio kind]   {out.get('audio', {}).get('kind', '?')}")
    print(f"[sample_rate]  {out.get('sample_rate')}")
    print(f"[duration_s]   {out.get('audio', {}).get('duration_s', '?')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
