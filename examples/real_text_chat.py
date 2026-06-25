"""End-to-end real-model coverage of the L4 text_chat node (v0.4.0 P0).

This example is the headline demo of the v0.4.0 P0 milestone:
*it* proves the 30-node L4 capability layer can be exercised
**end-to-end with a real neural network** (no echo, no
passthrough) while remaining *pure torch, zero external
dependencies*.

The flow is:

1. (Optional) Train a tiny Transformer LM with
   :func:`models.providers.pretrain_tiny.train_tiny_transformer`
   (a few hundred AdamW steps on a tiny in-memory corpus;
   ~30 s on CPU for the ``"small"`` preset, ~2 s for the
   ``"tiny"`` preset).
2. Wrap the resulting checkpoint in a
   :class:`models.providers.local_text.LocalTorchTextProvider`,
   which satisfies the
   :class:`models.interfaces.llm_provider.LLMProvider` protocol.
3. Register the provider as the default text backend via
   :func:`nodes._helpers.register_default_text_backend`.  The
   L4 ``text_chat`` node resolves the backend through
   :func:`call_text_backend`, so *no* node code needs to change.
4. Build a 1-node pipeline with ``text_chat`` and run it; the
   output is a real text continuation produced by the project-
   owned Transformer.

Usage::

    # Tiny preset (~2s pretrain, ~0.3M params):
    python examples/real_text_chat.py --preset tiny --skip-pretrain

    # Small preset (~30s pretrain, ~10M params, better output):
    python examples/real_text_chat.py --preset small

    # Skip pretraining, load an existing checkpoint:
    python examples/real_text_chat.py --skip-pretrain \\
        --checkpoint assets/checkpoints/tiny-transformer-small.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="real_text_chat",
        description=(
            "Train (or load) a project-owned tiny Transformer and "
            "drive the L4 text_chat node with it."
        ),
    )
    parser.add_argument(
        "--preset", choices=("tiny", "small"), default="small",
        help="Model preset (default: small, ~10M params).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override the number of pretraining steps.",
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip training; build a random-init provider instead.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pt file to load instead of training.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Maximum new tokens to generate (default: 64).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device (default: cpu).",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("TorchaVerse — v0.4.0 P0 real-model text_chat")
    print("=" * 60)

    from models.providers import (
        TINY_CONFIG, SMALL_CONFIG,
        TrainConfig, train_tiny_transformer,
        LocalTorchTextProvider, fetch_and_load_text,
    )

    cfg = TINY_CONFIG if args.preset == "tiny" else SMALL_CONFIG

    # ---- 1. (Optional) pretrain ----------------------------------------
    if args.checkpoint is not None and not args.skip_pretrain:
        print(f"[step 1] Loading checkpoint from {args.checkpoint}")
        provider = LocalTorchTextProvider.from_file(
            args.checkpoint, device=args.device,
        )
    elif args.skip_pretrain:
        print("[step 1] Building random-init provider (no pretrain).")
        provider = fetch_and_load_text(
            repo_id="torcha-verse/tiny-transformer-{}".format(args.preset),
            config_name=args.preset,
            device=args.device,
        )
    else:
        # Pretrain into a temp file (don't pollute the repo).
        tmpdir = tempfile.mkdtemp(prefix="torcha-verse-p0-")
        ckpt = os.path.join(tmpdir, "tiny-transformer-{}.pt".format(args.preset))
        tcfg = TrainConfig(preset=cfg.name, out_path=ckpt)
        if args.steps is not None:
            tcfg.steps = int(args.steps)
        print(
            "[step 1] Pretraining {name} for {n} steps "
            "(~{p:.1f}M params) -> {ckpt}".format(
                name=cfg.name, n=tcfg.steps,
                p=cfg.approx_params_m(), ckpt=ckpt,
            )
        )
        train_tiny_transformer(
            config=cfg, train_cfg=tcfg, device=args.device, save=True,
        )
        provider = LocalTorchTextProvider.from_file(ckpt, device=args.device)

    print("[step 2] Provider ready: {!r}".format(provider))

    # ---- 2. Register as the L4 default text backend -------------------
    from nodes._helpers import (
        register_default_text_backend,
        call_text_backend,
    )
    from core.module_bus import ModuleBus
    from models.interfaces.llm_provider import LLMProvider

    # Adapt the LocalTorchTextProvider to the LLMProvider protocol the
    # backend layer expects: a ``generate(prompt, **kwargs) -> str``
    # method.  LocalTorchTextProvider already implements that
    # signature -- the registration only needs to make sure the bus
    # can resolve the backend through a zero-arg ``factory`` callable.
    register_default_text_backend(lambda: provider)
    print("[step 3] L4 text backend registered (factory -> provider).")

    # ---- 3. Drive the L4 text_chat node end-to-end --------------------
    from nodes.base import NodeContext
    from pipeline.composer import PipelineBuilder

    # Two prompts (English + Chinese) to prove the byte-level
    # tokenizer handles both without manual preprocessing.
    prompts = [
        "Hello, who are you?",
        "用一句话介绍 TorchaVerse 框架",
    ]

    for prompt in prompts:
        print("\n--- prompt: {!r} ---".format(prompt))
        # Direct call: bypass the Pipeline to make the backend
        # resolution path explicit.
        direct = call_text_backend(
            bus=None,
            name="default",
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        direct_text = direct.get("text", "") if isinstance(direct, dict) else str(direct)
        print("[direct] {!r}".format(direct_text[:200]))

        # End-to-end: build a 1-node pipeline so the L4 layer is
        # exercised through the public API.
        pipeline = (
            PipelineBuilder("real_text_chat_{}".format(abs(hash(prompt)) % 1000))
            .node(
                "text_chat",
                id="chat",
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            .build()
        )
        out = pipeline.run(NodeContext())["chat"]
        text = out.get("text", "")
        print("[node  ] {!r}".format(text[:200]))

    print("\nP0 demo complete. (Provider params: {}; vocab: {}.)".format(
        provider.num_parameters(), provider.config.vocab_size,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
