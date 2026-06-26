"""Local Transformers Demo — 自研 transformers 风格的端到端 demo (v0.10.0)。

本 example 演示如何使用项目自 v0.8 起累积的"自研 transformers 风格"
本地运行时,在**零外部依赖**的前提下,完成:

1. **下载**(可选):用 :class:`LocalModelHub.download` 走项目自己的
   :mod:`models.source` mirror / dedup / integrity 路径,无需
   ``huggingface_hub``。
2. **加载**:用 :func:`load_model_and_tokenizer` 一行加载模型 + 配套
   tokenizer。函数内部会自动:
   - 检测 model family (HunyuanDiT / FLUX / SD3 / Wan2 / MusicGen /
     TinyTransformer)
   - 选 upstream → local key rename table (HUNYUAN_DIT_KEY_MAP 等)
   - 走 :meth:`models.base.ModelMixin.from_pretrained` + key_renames
   - 从 ``vocab.json`` / ``merges.txt`` / ``sp.model`` 自动构建
     :class:`TokenizerBundle`
3. **推理**:用 :func:`pipeline` 一行构造类似 ``transformers.pipeline``
   的多模态推理管道:
   - ``text-generation`` → :class:`LocalTextGenerationPipeline`
   - ``text-to-image``  → :class:`LocalImageGenerationPipeline`
   - ``text-to-speech`` / ``music-generation`` → :class:`LocalAudioPipeline`
4. **节点串联**(可选):用 :func:`enable_local_runtime` 一行把"自研
   加载 + 真推理循环"注入 :class:`core.module_bus.ModuleBus`,
   让 39 个 L4 节点全部切到真模型真生成。

运行::

    python examples/local_transformers_demo.py --task text-generation
    python examples/local_transformers_demo.py --task text-to-image
    python examples/local_transformers_demo.py --task text-to-speech
    python examples/local_transformers_demo.py --task all
    python examples/local_transformers_demo.py --inject-runtime

依赖
----

零外部依赖。只用 ``torch`` + 项目自有代码。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quiet down the project's logger so the demo output stays compact.
import logging
logging.getLogger("models").setLevel(logging.WARNING)
logging.getLogger("nodes").setLevel(logging.WARNING)
logging.getLogger("infrastructure").setLevel(logging.WARNING)

from infrastructure.logger import get_logger

_logger = get_logger("examples.local_transformers_demo")


# ---------------------------------------------------------------------------
# TinyTransformer demo (works without any external weights)
# ---------------------------------------------------------------------------
def demo_text_generation(offline: bool = True) -> int:
    """End-to-end text generation demo using the project-owned
    TinyTransformer + :class:`LocalModelForCausalLM` wrapper.

    This path needs **no external weights** -- the model is
    initialised from a fresh ``TINY_CONFIG``.  The output text
    will be noisy (random init) but the round-trip is fully
    exercised, which is exactly the demo the v0.4.x P0 milestone
    shipped.
    """
    print("=" * 64)
    print("[1/3] Text generation (TinyTransformer, ~2M params, random init)")
    print("=" * 64)

    from models.runtime import (
        LocalModelForCausalLM,
        LocalTextGenerationPipeline,
    )
    from models.providers.tiny_transformer import (
        TINY_CONFIG,
        build_tiny_transformer,
    )

    t0 = time.time()
    model, tok = build_tiny_transformer(TINY_CONFIG)
    head = LocalModelForCausalLM(model, family="tiny_transformer")
    pipe = LocalTextGenerationPipeline(head)
    t_build = time.time() - t0
    print(f"  pipeline built in {t_build:.3f}s ({model.num_parameters_human()})")

    prompts = [
        "the quick brown fox",
        "once upon a time",
        "lorem ipsum dolor sit amet",
    ]
    out = pipe(prompts, max_new_tokens=24, do_sample=False)
    for rec in out:
        print(f"  prompt:    {rec['prompt']!r}")
        print(f"  generated: {rec['generated_text']!r}")
        print()
    return 0


# ---------------------------------------------------------------------------
# Image generation demo (works without real weights; returns random latents)
# ---------------------------------------------------------------------------
def demo_image_generation(offline: bool = True) -> int:
    """End-to-end image generation demo.

    The demo does *not* require a real HunyuanDiT / FLUX / SD3
    checkpoint -- the wrapper builds a tiny ``HunyuanDiT`` from
    ``HunyuanDiTConfig.tiny()`` (already in :mod:`models.image.dit`)
    and runs the v0.8.x diffusion loop helper.  When the helper
    is unavailable the wrapper falls back to a "random latents"
    path so the contract is still honoured.
    """
    print("=" * 64)
    print("[2/3] Image generation (HunyuanDiT-tiny, ~0.4M params, random init)")
    print("=" * 64)

    from models.runtime import (
        LocalModelForTextToImage,
        LocalImageGenerationPipeline,
    )
    from models.base import ModelMixin

    t0 = time.time()
    head = LocalModelForTextToImage(
        ModelMixin(), family="hunyuan_dit",
    )
    pipe = LocalImageGenerationPipeline(head)
    t_build = time.time() - t0
    print(f"  pipeline built in {t_build:.3f}s")

    out = pipe(
        "a serene mountain landscape at sunset, oil painting",
        height=256,
        width=256,
        num_inference_steps=4,  # tiny model, just to demo the loop
        guidance_scale=4.0,
        sampler="flow_match_euler",
    )
    for rec in out:
        print(f"  prompt:    {rec['prompt']!r}")
        print(f"  family:    {rec['family']!r}")
        if "latents" in rec:
            latents = rec["latents"]
            if hasattr(latents, "shape"):
                print(f"  latents:   shape={tuple(latents.shape)}")
            else:
                print(f"  latents:   {type(latents).__name__}")
        for k in ("note", "sampler"):
            if k in rec:
                print(f"  {k}:        {rec[k]!r}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Audio / TTS demo
# ---------------------------------------------------------------------------
def demo_text_to_speech(offline: bool = True) -> int:
    """End-to-end TTS demo using the project-owned
    :class:`LocalModelForTextToSpeech` wrapper.
    """
    print("=" * 64)
    print("[3/3] Text-to-speech (MusicGen-shaped, random init placeholder)")
    print("=" * 64)

    from models.runtime import (
        LocalModelForTextToSpeech,
        LocalAudioPipeline,
    )
    from models.base import ModelMixin

    t0 = time.time()
    head = LocalModelForTextToSpeech(
        ModelMixin(), family="musicgen",
    )
    pipe = LocalAudioPipeline(head)
    t_build = time.time() - t0
    print(f"  pipeline built in {t_build:.3f}s")

    out = pipe(
        "Hello world, this is a test of the local runtime.",
        sample_rate=22050,
    )
    for rec in out:
        print(f"  prompt:    {rec['prompt']!r}")
        print(f"  family:    {rec['family']!r}")
        if "mel" in rec:
            mel = rec["mel"]
            if hasattr(mel, "shape"):
                print(f"  mel:       shape={tuple(mel.shape)}")
            else:
                print(f"  mel:       {type(mel).__name__}")
        for k in ("sample_rate", "note"):
            if k in rec:
                print(f"  {k}:        {rec[k]!r}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Optional: runtime injection into the 39 L4 nodes
# ---------------------------------------------------------------------------
def demo_inject_runtime() -> int:
    """Demonstrate :func:`enable_local_runtime` -- one call that
    switches the 39 L4 nodes from "echo backends" to "real local
    backends".  This is the "**基础设施齐全但未串联**" → "**已串联**"
    bridge.
    """
    print("=" * 64)
    print("[*] Inject local runtime into the 39 L4 nodes")
    print("=" * 64)

    from models.runtime import (
        RuntimeConfig,
        enable_local_runtime,
        is_local_runtime_enabled,
        get_active_config,
        disable_local_runtime,
    )

    cfg = RuntimeConfig(
        prefer_local_text=True,
        prefer_local_image=True,
        prefer_local_video=True,
        prefer_local_audio=True,
        prefer_local_multimodal=True,
        torch_dtype=None,
        device="cpu",
        use_real_diffusion_loop=True,
        tags=["demo"],
    )
    enable_local_runtime(cfg)
    print(f"  enabled:   {is_local_runtime_enabled()}")
    print(f"  config:    {get_active_config().describe()}")

    # Now run a few L4 nodes end-to-end to prove the backend was swapped.
    from pipeline.composer import PipelineBuilder
    from nodes.base import NodeContext

    print("\n  running 3 representative nodes to prove the swap:")
    for node_id, kw in [
        ("text_chat", {"prompt": "hello, world!", "max_new_tokens": 12}),
        ("image_txt2img", {"prompt": "a tiny cat", "width": 64, "height": 64}),
        ("audio_tts", {"text": "hi from torcha-verse", "sample_rate": 22050}),
    ]:
        try:
            t0 = time.time()
            pipe = PipelineBuilder(f"demo-{node_id}").node(node_id, id=node_id, **kw).build()
            ctx = NodeContext()
            result = pipe.run(ctx)
            elapsed = time.time() - t0
            print(f"    {node_id:18s}  ok   {elapsed:.3f}s   result_keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {node_id:18s}  FAIL {exc!r}")

    # Disable the runtime to leave the demo's environment clean.
    disable_local_runtime()
    print(f"\n  disabled:  {not is_local_runtime_enabled()}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Family detection demo (no weights required)
# ---------------------------------------------------------------------------
def demo_family_detection() -> int:
    """Show :func:`detect_model_family` and the load_model_and_tokenizer
    entry point with a synthetic checkpoint."""
    print("=" * 64)
    print("[+] Family detection (synthetic HunyuanDiT-style checkpoint)")
    print("=" * 64)

    import torch
    from models.runtime import (
        detect_model_family,
        load_model_and_tokenizer,
        ModelFamily,
    )

    # Build a tiny fake "HunyuanDiT-style" state-dict and save it.
    fake = {
        "img_in.proj.weight": torch.zeros(8, 4, 3, 3),
        "img_in.proj.bias": torch.zeros(8),
        "x_embedder.weight": torch.zeros(8, 4),
        "x_embedder.bias": torch.zeros(4),
        "time_in.mlp.0.weight": torch.zeros(8, 4),
        "time_in.mlp.0.bias": torch.zeros(8),
        "time_in.mlp.2.weight": torch.zeros(8, 8),
        "time_in.mlp.2.bias": torch.zeros(8),
        "vector_in.proj.weight": torch.zeros(8, 4),
        "vector_in.proj.bias": torch.zeros(8),
        "style_embedder.weight": torch.zeros(8, 4),
        "size_embedder.weight": torch.zeros(8, 4),
    }
    tmp = tempfile.mkdtemp(prefix="torcha_verse_demo_")
    ckpt_path = os.path.join(tmp, "hunyuan_tiny.safetensors")
    from core.checkpoint_loader import save_safetensors
    save_safetensors(fake, ckpt_path)
    print(f"  fake ckpt:  {ckpt_path}")

    fam = detect_model_family(ckpt_path)
    print(f"  detected:  {fam!r}  ({fam.value!r})")
    assert fam == ModelFamily.HUNYUAN_DIT, f"expected HUNYUAN_DIT, got {fam!r}"

    # Now exercise the full load_model_and_tokenizer round-trip.
    # HunyuanDiTConfig.tiny() exists in models.image.dit (per the
    # v0.8.5 acceptance criteria).  We pass strict=False so missing
    # keys (this tiny fake has only the 11 keys above) do not error.
    try:
        mdl, tok, fam = load_model_and_tokenizer(
            tmp,
            torch_dtype=torch.float32,
            device="cpu",
            strict=False,
        )
        print(f"  loaded:    family={fam.value!r}  params={mdl.num_parameters_human()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  load failed (acceptable in dev env without real keys): {exc!r}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("text-generation", "text-to-image", "text-to-speech", "all"),
        default="all",
        help="which demo to run (default: all)",
    )
    parser.add_argument(
        "--inject-runtime",
        action="store_true",
        help="also run the enable_local_runtime() bridge demo",
    )
    parser.add_argument(
        "--no-family-detection",
        action="store_true",
        help="skip the synthetic-checkpoint family-detection demo",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("TorchaVerse — v0.10.0 Local Transformers Demo")
    print("(自研 transformers 风格, 零外部依赖)")
    print("=" * 64)
    print()

    rc = 0
    if not args.no_family_detection:
        rc |= demo_family_detection()

    if args.task in ("text-generation", "all"):
        rc |= demo_text_generation()
    if args.task in ("text-to-image", "all"):
        rc |= demo_image_generation()
    if args.task in ("text-to-speech", "all"):
        rc |= demo_text_to_speech()

    if args.inject_runtime:
        rc |= demo_inject_runtime()

    print("Demo complete.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
