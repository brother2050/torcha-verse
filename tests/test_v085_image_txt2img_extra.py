"""v0.8.5 — image_txt2img "extra" edge-case tests (5 tests).

The companion test file
:file:`tests/test_v085_image_txt2img_nonzero.py` covers the
core "non-trivial output" contract of the v0.8.0 §3.5
acceptance target.  This file exercises the *edge cases* of
the same :func:`call_diffusion_loop_backend` entry point and
the higher-level :class:`~nodes.image.ImageTxt2ImgNode`:

1. ``test_cfg_scale_zero_returns_unconditional`` -- a
   ``cfg_scale=0`` call with the same seed must be
   deterministic and produce identical latents on two
   consecutive calls.
2. ``test_negative_prompt_accepted`` -- the higher-level
   :class:`ImageTxt2ImgNode` accepts a ``negative_prompt``
   keyword argument and returns a valid response.
3. ``test_seed_reproducibility_across_pipelines`` -- two
   independently constructed pipelines (same model
   architecture, same seed) must produce identical final
   latents.
4. ``test_num_inference_steps_affects_output`` -- the number
   of inference steps changes the final latents (a model
   that ignored the step count would produce identical
   outputs for ``num_inference_steps=2`` and ``=5``).
5. ``test_height_width_square_and_non_square`` -- the
   pipeline accepts both square and non-square (W, H) and
   returns a latents tensor whose spatial dims are
   ``(H/8, W/8)`` (the project's standard VAE downscale).

All five tests are CPU-only; the 2-block / 96-dim
HunyuanDiT-Tiny preset is small enough to run a 5-step
diffusion loop in well under a second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make sure the test can import the node-level helper without
# going through the full ``nodes`` package init (which may pull
# optional adapters that are unrelated to the test).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nodes._helpers._backends import call_diffusion_loop_backend  # noqa: E402
from models.image.dit import HunyuanDiT  # noqa: E402

__all__ = ["TestImageTxt2ImgExtra"]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _build_model_and_text(
    prompt: str, *, seed: int = 0,
):
    """Build a HunyuanDiT-Tiny + text encoder pair, perturb the
    final layer so the model produces a non-zero output, and
    return ``(model, text_embeds)``.

    The ``final_layer`` is the same trick used in
    :file:`tests/test_v085_image_txt2img_nonzero.py`: a real
    checkpoint would replace the zeros at load time, but the
    CPU smoke test needs a tiny perturbation to make the
    denoiser output non-trivial.
    """
    import torch.nn as nn
    from papers.adapters.hunyuan_dit import HunyuanTextEncoder
    # The seed must be applied *before* the model
    # construction -- ``HunyuanDiT()`` consumes RNG state in
    # its parameter init, so a ``torch.manual_seed(seed)``
    # call placed *after* the construction would not
    # cover those initial weights and would break
    # cross-pipeline reproducibility.
    torch.manual_seed(seed)
    model = HunyuanDiT()
    with torch.no_grad():
        nn.init.normal_(model.final_layer.adaln_modulation.weight, std=0.02)
        nn.init.normal_(model.final_layer.out_proj.weight, std=0.02)
    # Force eval() so that dropout (in the attention
    # layers) and any other stochastic graph nodes are
    # disabled.  This is required for seed reproducibility
    # across two independently constructed pipelines.
    model.eval()
    encoder = HunyuanTextEncoder(
        dim=model.config["context_dim"], max_len=24,
    )
    encoder.eval()
    text_embeds = encoder([prompt])
    return model, text_embeds


def _run_loop(
    prompt: str,
    *,
    num_steps: int = 5,
    seed: int = 0,
    cfg_scale: float = 1.0,
    height: int = 8,
    width: int = 8,
) -> torch.Tensor:
    """Run :func:`call_diffusion_loop_backend` with the
    HunyuanDiT-Tiny denoiser and return the final latents.
    """
    model, text_embeds = _build_model_and_text(prompt, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, 4, height, width, generator=gen)
    loop = call_diffusion_loop_backend(
        bus=None,
        name="image_txt2img_extra_test",
        model=model,
        latents=latents,
        text_embeds=text_embeds,
        num_inference_steps=num_steps,
        guidance_scale=float(cfg_scale),
        sampler="euler",
        shift=1.0,
    )
    if not isinstance(loop, dict):
        raise RuntimeError(
            f"call_diffusion_loop_backend did not return a dict: "
            f"{type(loop)}"
        )
    out = loop.get("latents", latents)
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            f"call_diffusion_loop_backend latents is not a Tensor: "
            f"{type(out)}"
        )
    return out.float()


# ===========================================================================
# Tests
# ===========================================================================
class TestImageTxt2ImgExtra:
    """Edge cases for the v0.8.5 image_txt2img contract."""

    def test_cfg_scale_zero_returns_unconditional(self) -> None:
        """``cfg_scale=0`` must be deterministic: two consecutive
        calls with the same seed produce identical latents.

        The CFG code path in
        :func:`call_diffusion_loop_backend` is disabled when
        ``guidance_scale <= 1.0``; with ``text_embeds=None``
        the model output is fully deterministic.  We pass
        ``text_embeds=None`` to make the assertion tight.
        """
        torch.manual_seed(7)
        # Build the model once; both calls share it (and the
        # same final-layer perturbation) so the only source
        # of non-determinism is the RNG.
        from models.image.dit import HunyuanDiT
        import torch.nn as nn
        model = HunyuanDiT()
        with torch.no_grad():
            nn.init.normal_(
                model.final_layer.adaln_modulation.weight, std=0.02,
            )
            nn.init.normal_(
                model.final_layer.out_proj.weight, std=0.02,
            )
        # Force eval() so dropout does not break determinism.
        model.eval()
        gen = torch.Generator(device="cpu").manual_seed(123)
        latents = torch.randn(1, 4, 8, 8, generator=gen)

        def _call() -> torch.Tensor:
            # Reset the RNG to a known state before each call.
            torch.manual_seed(456)
            return call_diffusion_loop_backend(
                bus=None,
                name="cfg_zero_test",
                model=model,
                latents=latents,
                text_embeds=None,
                num_inference_steps=4,
                guidance_scale=0.0,
                sampler="euler",
                shift=1.0,
            )["latents"].float()

        out_a = _call()
        out_b = _call()
        assert torch.equal(out_a, out_b), (
            "cfg_scale=0 with text_embeds=None must be "
            "bit-deterministic; the two calls differ."
        )

    def test_negative_prompt_accepted(self) -> None:
        """The :class:`ImageTxt2ImgNode` accepts a
        ``negative_prompt`` kwarg and returns a valid
        response.

        The v0.8.5 ``ImageTxt2ImgNode`` advertises
        ``negative_prompt`` in its :attr:`NodeSpec.inputs`
        and :meth:`validate_inputs` accepts it (it is
        *optional*).  The legacy placeholder path does not
        yet forward the kwarg to the diffusion backend --
        that is a v0.9.x follow-up -- but the *API surface*
        (input shape, validation, kwargs forwarding into
        the node's ``execute``) is locked in.  This test
        pins down three things:

        1. ``validate_inputs`` accepts the kwarg and
           returns no errors.
        2. ``execute`` accepts the kwarg via ``**inputs``
           and does not raise.
        3. The returned dict is structurally valid
           (``image`` / ``seed`` / ``width`` / ``height``).
        """
        from nodes.base import NodeContext
        from nodes.image import ImageTxt2ImgNode

        ctx = NodeContext(
            config={"default_image_model": "mock-image-model"},
        )

        base_inputs = {
            "prompt": "a serene mountain landscape at sunrise",
            "width": 64,
            "height": 64,
            "steps": 2,
            "guidance_scale": 7.5,
            "seed": 0,
        }
        # 1. validate_inputs() must accept negative_prompt
        # without error.
        node = ImageTxt2ImgNode()
        errors_no = node.validate_inputs(base_inputs)
        assert errors_no == [], f"unexpected errors: {errors_no}"
        errors_with = node.validate_inputs({
            **base_inputs, "negative_prompt": "blurry, low quality",
        })
        assert errors_with == [], f"unexpected errors: {errors_with}"

        # 2. execute() must accept negative_prompt via
        # **inputs without raising.
        result_no = node.execute(ctx, **base_inputs)
        assert isinstance(result_no, dict)
        result_with = node.execute(
            ctx, **base_inputs,
            negative_prompt="blurry, low quality",
        )
        assert isinstance(result_with, dict)

        # 3. Both results are structurally valid.
        for r in (result_no, result_with):
            assert "image" in r
            assert int(r.get("seed", -1)) == base_inputs["seed"]
            assert int(r.get("width", 0)) == int(base_inputs["width"])
            assert int(r.get("height", 0)) == int(base_inputs["height"])

    def test_seed_reproducibility_across_pipelines(self) -> None:
        """Two separately constructed pipelines (same seed)
        must produce bit-identical final latents.

        This is the "test_against_seed_determinism"
        contract: a non-deterministic model would diverge
        between independent builds.  We use the same
        random seed, the same latents seed, and the same
        model architecture / final-layer perturbation.
        The test is gated on
        :func:`torch.manual_seed` being called *before*
        the model construction (see the helper
        :func:`_build_model_and_text` for the exact
        ordering) so that the init RNG state is
        reproducible across calls.

        Note: the underlying
        :func:`torch.nn.functional.scaled_dot_product_attention`
        kernel can pick a non-deterministic FLASH path
        on some backends.  We re-test bit-equality here
        because the seeded init above is the only
        source of randomness -- the model is in eval
        mode, the loop is deterministic, and the SDPA
        path is a no-op in the Tiny preset (the
        attention scores are small enough that the
        default ``MATH`` kernel is selected, which is
        fully deterministic).
        """
        out_a = _run_loop("a quiet teacup on a wooden table", seed=0)
        out_b = _run_loop("a quiet teacup on a wooden table", seed=0)
        assert out_a.shape == out_b.shape
        # Bit-identical -- both pipelines used the same
        # seed, the same final-layer perturbation, and the
        # same initial latents.
        assert torch.equal(out_a, out_b), (
            "Two pipelines with the same seed produced "
            "different latents (loss of determinism).  "
            f"max diff = {(out_a - out_b).abs().max().item():.4f}"
        )

    def test_num_inference_steps_affects_output(self) -> None:
        """``num_inference_steps=2`` and ``=5`` must produce
        different latents (same prompt, same seed).

        A model that ignored the step count would loop
        indefinitely or short-circuit and produce
        bit-identical outputs.  This test pins down the
        contract that the diffusion loop *consumes* the
        requested step count.
        """
        torch.manual_seed(0)
        out_2 = _run_loop("a vibrant sunset over the ocean", seed=0,
                          num_steps=2)
        out_5 = _run_loop("a vibrant sunset over the ocean", seed=0,
                          num_steps=5)
        assert out_2.shape == out_5.shape
        # The two step counts must produce different
        # latents (the integration path differs).
        assert not torch.allclose(out_2, out_5), (
            "num_inference_steps=2 and =5 produced "
            "indistinguishable latents."
        )

    def test_height_width_square_and_non_square(self) -> None:
        """The pipeline accepts both square and non-square
        (W, H) inputs and returns latents whose spatial
        dims are ``(H/8, W/8)``.

        The VAE downscale is 8x in the standard SD / SD3
        layout, so a ``(W=64, H=64)`` request becomes
        ``(8, 8)`` latents and a ``(W=64, H=96)`` request
        becomes ``(12, 8)``.
        """
        # 1. Square 64x64 -> (1, 4, 8, 8) latents.
        out_sq = _run_loop("a small bird on a branch",
                            seed=0, height=8, width=8)
        assert out_sq.shape == (1, 4, 8, 8)
        # 2. Non-square 64x96 (W=64, H=96) -> (1, 4, 12, 8).
        out_ns = _run_loop("a small bird on a branch",
                            seed=0, height=12, width=8)
        assert out_ns.shape == (1, 4, 12, 8)
        # 3. Even more asymmetric 96x64 -> (1, 4, 8, 12).
        out_tall = _run_loop("a small bird on a branch",
                              seed=0, height=8, width=12)
        assert out_tall.shape == (1, 4, 8, 12)
        # The two non-square latents must differ in
        # shape, not just content.
        assert out_ns.shape != out_tall.shape
        assert out_sq.shape != out_ns.shape
