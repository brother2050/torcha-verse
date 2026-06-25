"""Text-to-speech via the L4 node layer (v0.4.x P0 real-model path).

Demonstrates the ``audio_tts`` node **with the project-owned
:class:`LocalTorchAudioProvider`** installed as the fallback
backend -- so the demo actually runs a real TTS-Transformer +
HiFi-GAN forward pass instead of the echo stub.

Run with::

    python examples/audio_tts.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._helpers import register_default_audio_backend
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — v0.4.x P0 Audio TTS (LocalTorch)")
    print("=" * 60)

    # Install the project-owned real-model backend so the
    # node exercises an actual TTS-Transformer + HiFi-GAN
    # forward pass (no echo).
    register_default_audio_backend()

    pipeline = (
        PipelineBuilder("tts_demo")
        .node(
            "audio_tts",
            id="tts",
            text="Hello from TorchaVerse!",
            voice="default",
            language="en",
            speed=1.0,
            sample_rate=16000,
        )
        .build()
    )

    t0 = time.time()
    out = pipeline.run(NodeContext())["tts"]
    elapsed = time.time() - t0
    print(f"\n[output keys]  {sorted(out.keys())}")
    print(f"[audio kind]   {out.get('audio', {}).get('kind', '?')}")
    print(f"[sample_rate]  {out.get('sample_rate')}")
    print(f"[duration_s]   {out.get('audio', {}).get('duration_s', '?')}")
    print(f"[elapsed]      {elapsed:.2f}s")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
