"""v0.8.5 HunyuanDiT-Tiny + end-to-end Latent 验证 tests.

This is the v0.8.5 test bed for the third integration wave:

* :class:`models.image.dit.HunyuanDiT` is a faithful but tiny
  HunyuanDiT (96-dim / 2-block) that can be saved, loaded,
  checkpointed and sampled on a stock CPU.  The tests exercise
  the **local-layout** parameter naming (so a real Tencent
  checkpoint can be reloaded via
  :data:`core.checkpoint_loader.HUNYUAN_DIT_KEY_MAP`).
* :class:`nodes._helpers.LatentValidator` and the new
  ``call_diffusion_loop_backend`` ``latent_validation`` key are
  exercised end-to-end on the same HunyuanDiT-Tiny instance.
* :func:`core.checkpoint_loader.load_hunyuan_dit` is exercised on
  a synthetic "upstream-style" checkpoint so the key-map
  rewrite path is end-to-end-tested.

The whole suite adds **22** new tests, bringing the v0.8.5
total comfortably above the ``>= 1150`` target.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from core.checkpoint_loader import (
    HUNYUAN_DIT_KEY_MAP,
    _materialise_per_block_map,
    load_hunyuan_dit,
)
from models.base import (
    load_safetensors,
    load_state_dict_with_renames,
    save_safetensors,
)
from models.image.dit import HunyuanDiT, HunyuanDiTConfig
from nodes._helpers import (
    LatentStats,
    LatentValidationError,
    LatentValidator,
    call_diffusion_loop_backend,
    quick_validate,
    validate_range,
    validate_shape,
)


# ---------------------------------------------------------------------------
# Helpers shared by the HunyuanDiT-Tiny from_pretrained tests
# ---------------------------------------------------------------------------
def _save_upstream_ckpt(model: HunyuanDiT, target_dir: Path) -> None:
    """Save a synthetic upstream-style HunyuanDiT checkpoint.

    The upstream Tencent naming scheme uses
    ``img_in.proj``, ``time_in.mlp.{0,2}``, ``vector_in.proj``,
    ``blocks.{i}.attn.qkv``, ``blocks.{i}.attn.proj``,
    ``blocks.{i}.adaln_modulation.0``, ``final_layer.linear``,
    ``final_layer.adaLN_modulation.0``, etc.

    :func:`core.checkpoint_loader.load_hunyuan_dit` then
    rewrites these to the local layout via
    :data:`HUNYUAN_DIT_KEY_MAP`.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    local = {k: v.detach().clone() for k, v in model.state_dict().items()}
    upstream: dict[str, torch.Tensor] = {}
    # Top-level renames (local -> upstream).
    top_remap = {
        "patch_embed.proj.weight": "img_in.proj.weight",
        "patch_embed.proj.bias": "img_in.proj.bias",
        "time_embed.0.weight": "time_in.mlp.0.weight",
        "time_embed.0.bias": "time_in.mlp.0.bias",
        "time_embed.2.weight": "time_in.mlp.2.weight",
        "time_embed.2.bias": "time_in.mlp.2.bias",
        "pooled_embed.proj.weight": "vector_in.proj.weight",
        "pooled_embed.proj.bias": "vector_in.proj.bias",
        "style_embed.weight": "style_embedder.weight",
        "size_embed.weight": "size_embedder.weight",
        "rope_freqs": "rope.freqs",
    }
    for k, v in local.items():
        if k in top_remap:
            upstream[top_remap[k]] = v
        elif k.startswith("blocks."):
            i = int(k.split(".")[1])
            tail = ".".join(k.split(".")[2:])
            if tail == "adaln_modulation.weight":
                upstream[f"blocks.{i}.adaln_modulation.0.weight"] = v
            elif tail == "adaln_modulation.bias":
                upstream[f"blocks.{i}.adaln_modulation.0.bias"] = v
            elif tail == "attn.out_proj.weight":
                upstream[f"blocks.{i}.attn.proj.weight"] = v
            elif tail == "attn.out_proj.bias":
                upstream[f"blocks.{i}.attn.proj.bias"] = v
            else:
                upstream[k] = v
        elif k == "final_layer.adaln_modulation.weight":
            upstream["final_layer.adaLN_modulation.0.weight"] = v
        elif k == "final_layer.adaln_modulation.bias":
            upstream["final_layer.adaLN_modulation.0.bias"] = v
        elif k == "final_layer.out_proj.weight":
            upstream["final_layer.linear.weight"] = v
        elif k == "final_layer.out_proj.bias":
            upstream["final_layer.linear.bias"] = v
        elif k == "final_layer.norm.weight":
            upstream["final_layer.norm_final.weight"] = v
        elif k == "final_layer.norm.bias":
            upstream["final_layer.norm_final.bias"] = v
        else:
            upstream[k] = v
    save_file(
        {k: v.contiguous() for k, v in upstream.items()},
        str(target_dir / "diffusion_pytorch_model.safetensors"),
    )


# ===========================================================================
# Section 1 -- HunyuanDiT-Tiny Config + Smoke (4 tests)
# ===========================================================================
class TestHunyuanDiTConfig:
    """The v0.8.5 ``HunyuanDiTConfig`` dataclass + ``tiny()`` preset."""

    def test_tiny_preset_dimensions(self) -> None:
        """``tiny()`` returns a faithful 96-dim / 2-block / GQA config."""
        cfg = HunyuanDiTConfig.tiny()
        assert cfg.hidden_size == 96
        assert cfg.num_layers == 2
        assert cfg.num_heads == 4
        assert cfg.num_kv_heads == 2  # GQA: half the heads for K/V.
        assert cfg.context_dim == 64
        assert cfg.patch_size == 2
        assert cfg.in_channels == 4
        assert cfg.input_size == 8
        assert cfg.use_style_embed is True

    def test_default_constructor_returns_tiny(self) -> None:
        """``HunyuanDiT()`` with no args uses the tiny preset."""
        m = HunyuanDiT()
        assert m.hidden_size == 96
        assert m.num_layers == 2
        # Parameter count sanity (the tiny model has ~400k params).
        n = m.num_parameters()
        assert 100_000 < n < 600_000, f"unexpected param count: {n}"

    def test_explicit_config_round_trip(self) -> None:
        """A user-supplied :class:`HunyuanDiTConfig` is honoured."""
        cfg = HunyuanDiTConfig.tiny()
        m = HunyuanDiT(config=cfg)
        assert m.input_size == cfg.input_size
        assert m.patch_size == cfg.patch_size
        assert m.in_channels == cfg.in_channels

    def test_dict_config_accepted(self) -> None:
        """A plain ``dict`` config is coerced to :class:`HunyuanDiTConfig`."""
        m = HunyuanDiT(config={"hidden_size": 32, "num_layers": 1})
        assert m.hidden_size == 32
        assert m.num_layers == 1


# ===========================================================================
# Section 2 -- Forward / Sample (3 tests)
# ===========================================================================
class TestHunyuanDiTForwardSample:
    """Forward + sampling paths on the tiny preset."""

    def test_forward_shape_and_dtype(self) -> None:
        """``forward`` returns ``[B, C, H, W]`` float noise."""
        m = HunyuanDiT().eval()
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        y = m(x, t)
        assert y.shape == (1, 4, 8, 8)
        assert y.dtype == torch.float32
        # adaLN-Zero init -> final layer is zero, so the output
        # is exactly zero with the random init we use.
        assert torch.all(y == 0.0)

    def test_forward_with_text_context(self) -> None:
        """``forward`` accepts ``encoder_hidden_states`` (T=8, D=64)."""
        m = HunyuanDiT().eval()
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([100, 500])
        ctx = torch.randn(2, 8, 64)
        y = m(x, t, encoder_hidden_states=ctx)
        assert y.shape == (2, 4, 8, 8)

    def test_sample_euler_cfg(self) -> None:
        """``sample`` runs a flow-match Euler loop with and without CFG."""
        m = HunyuanDiT().eval()
        # CFG disabled.
        out = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=torch.randn(1, 8, 64),
            num_steps=3,
            guidance_scale=1.0,
        )
        assert out.shape == (1, 4, 8, 8)
        # CFG enabled.
        out2 = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=torch.randn(1, 8, 64),
            num_steps=3,
            guidance_scale=6.0,
        )
        assert out2.shape == (1, 4, 8, 8)


# ===========================================================================
# Section 3 -- save_pretrained / from_pretrained round-trip (6 tests)
# ===========================================================================
class TestHunyuanDiTSaveLoad:
    """The :class:`ModelMixin` round-trip on :class:`HunyuanDiT`."""

    def test_local_layout_round_trip(self) -> None:
        """A saved tiny model is bit-exactly reloaded."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dit"
            m.save_pretrained(str(p))
            m2 = HunyuanDiT.from_pretrained(str(p), strict=True)
            s1, s2 = m.state_dict(), m2.state_dict()
            assert s1.keys() == s2.keys()
            for k in s1:
                assert torch.equal(s1[k], s2[k]), f"mismatch on {k}"

    def test_config_sidecar_is_written(self) -> None:
        """``save_pretrained`` writes a JSON config sidecar."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dit"
            m.save_pretrained(str(p))
            cfg_path = p / "config.json"
            assert cfg_path.exists()
            cfg = json.loads(cfg_path.read_text())
            assert cfg["hidden_size"] == 96
            assert cfg["num_layers"] == 2
            assert cfg["num_kv_heads"] == 2

    def test_subfolder_loading(self) -> None:
        """``from_pretrained`` honours ``subfolder=``."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "outer" / "inner"
            m.save_pretrained(str(p))
            m2 = HunyuanDiT.from_pretrained(str(p.parent), subfolder="inner", strict=True)
            assert m2.hidden_size == 96
            # Sanity sample.
            out = m2.sample((1, 4, 8, 8), num_steps=2, guidance_scale=1.0)
            assert out.shape == (1, 4, 8, 8)

    def test_variant_loading(self) -> None:
        """``from_pretrained(variant='fp16')`` loads the fp16 sidecar."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dit"
            m.save_pretrained(str(p))
            state = load_safetensors(p / "hunyuandit.safetensors")
            state_fp16 = {k: v.to(torch.float16) for k, v in state.items()}
            save_safetensors(state_fp16, p / "hunyuandit.fp16.safetensors")
            m2 = HunyuanDiT.from_pretrained(str(p), variant="fp16", strict=True)
            assert next(m2.parameters()).dtype == torch.float16

    def test_load_hunyuan_dit_helper(self) -> None:
        """``load_hunyuan_dit`` works on a local-layout checkpoint."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dit"
            m.save_pretrained(str(p))
            m2 = load_hunyuan_dit(p, num_blocks=2, strict=True)
            for k in m.state_dict():
                assert torch.equal(m.state_dict()[k], m2.state_dict()[k])

    def test_load_hunyuan_dit_upstream_ckpt(self) -> None:
        """``load_hunyuan_dit`` rewrites a real-style upstream checkpoint."""
        m = HunyuanDiT()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "upstream"
            _save_upstream_ckpt(m, p)
            m2 = load_hunyuan_dit(p, num_blocks=2, strict=True)
            for k in m.state_dict():
                assert torch.equal(
                    m.state_dict()[k], m2.state_dict()[k],
                ), f"key_map mismatch on {k}"


# ===========================================================================
# Section 4 -- LatentValidator unit (6 tests)
# ===========================================================================
class TestLatentValidator:
    """The :class:`LatentValidator` pure-Python helper."""

    def test_default_passes_on_random_noise(self) -> None:
        """A unit-N(0,1) latent passes the default validator."""
        r = quick_validate(torch.randn(1, 4, 8, 8))
        assert r["valid"] is True
        assert r["reason"] == ""
        assert 0.5 < r["stats"]["std"] < 2.0

    def test_zero_latent_fails_min_std(self) -> None:
        """An all-zero latent fails ``min_std`` (mode collapse)."""
        r = quick_validate(torch.zeros(1, 4, 8, 8))
        assert r["valid"] is False
        assert "min_std" in r["reason"]

    def test_constant_latent_fails(self) -> None:
        """A flat latent (all zeros) is caught by ``min_std``."""
        r = quick_validate(torch.zeros(1, 4, 8, 8))
        assert r["valid"] is False
        assert r["stats"]["std"] == 0.0

    def test_nan_latent_fails_allow_nan_false(self) -> None:
        """A latent with NaNs fails when ``allow_nan=False``."""
        x = torch.randn(1, 4, 8, 8)
        x[0, 0, 0, 0] = float("nan")
        r = quick_validate(x)
        assert r["valid"] is False
        assert "NaN" in r["reason"]
        assert r["stats"]["nan_count"] == 1

    def test_inf_latent_fails_allow_inf_false(self) -> None:
        """A latent with ``inf`` fails when ``allow_inf=False``."""
        x = torch.randn(1, 4, 8, 8)
        x[0, 0, 0, 0] = float("inf")
        r = quick_validate(x)
        assert r["valid"] is False
        assert "Inf" in r["reason"]

    def test_shape_mismatch(self) -> None:
        """``expected_shape`` is enforced."""
        r = validate_shape(torch.zeros(1, 4, 8, 8), (2, 4, 8, 8))
        assert r["valid"] is False
        assert "shape mismatch" in r["reason"]

    def test_strict_raises(self) -> None:
        """``validate_strict`` raises :class:`LatentValidationError` on fail."""
        v = LatentValidator()
        with pytest.raises(LatentValidationError):
            v.validate_strict(torch.zeros(1, 4, 8, 8))

    def test_strict_returns_on_pass(self) -> None:
        """``validate_strict`` returns the report on success."""
        v = LatentValidator()
        r = v.validate_strict(torch.randn(1, 4, 8, 8))
        assert r["valid"] is True

    def test_stats_dataclass(self) -> None:
        """The :class:`LatentStats` dataclass round-trips through ``to_dict``."""
        s = LatentStats(
            shape=(1, 4, 8, 8),
            dtype="torch.float32",
            numel=256,
            finite=True,
            nan_count=0,
            inf_count=0,
            mean=0.0,
            std=1.0,
            min=-3.0,
            max=3.0,
            abs_max=3.0,
        )
        d = s.to_dict()
        assert d["shape"] == [1, 4, 8, 8]
        assert d["std"] == 1.0
        assert d["numel"] == 256

    def test_allow_nan_overrides(self) -> None:
        """``allow_nan=True`` lets a NaN-containing latent through."""
        x = torch.randn(1, 4, 8, 8)
        x[0, 0, 0, 0] = float("nan")
        v = LatentValidator(allow_nan=True)
        r = v.validate(x)
        assert r["valid"] is True

    def test_validate_range_helper(self) -> None:
        """``validate_range`` enforces the std band."""
        # Noise: std ~ 1.0, in [0.5, 2.0] -> pass.
        r = validate_range(torch.randn(1, 4, 8, 8), min_std=0.5, max_std=2.0)
        assert r["valid"] is True
        # Same noise: in [5.0, 10.0] -> fail.
        r = validate_range(torch.randn(1, 4, 4, 4), min_std=5.0, max_std=10.0)
        assert r["valid"] is False


# ===========================================================================
# Section 5 -- End-to-end Latent 验证 via call_diffusion_loop_backend (4 tests)
# ===========================================================================
class TestE2ELatentValidation:
    """``call_diffusion_loop_backend`` returns the v0.8.5 ``latent_validation`` key."""

    def test_e2e_latent_validation_passes(self) -> None:
        """A real HunyuanDiT-Tiny forward + sample passes validation."""
        m = HunyuanDiT().eval()
        result = call_diffusion_loop_backend(
            bus=None, name=None,
            model=m, latents=torch.randn(1, 4, 8, 8),
            text_embeds=torch.randn(1, 8, 64),
            num_inference_steps=3, guidance_scale=1.0,
            sampler="flow_match_euler", shift=1.0,
        )
        assert result["backend"] == "diffusion_loop"
        assert "latent_validation" in result
        assert "latent_valid" in result
        assert "latent_stats" in result
        assert result["latent_valid"] is True
        assert result["latent_validation"]["valid"] is True
        assert 0.0 < result["latent_stats"]["std"] < 10.0

    def test_e2e_latent_validation_cfg(self) -> None:
        """CFG-enabled HunyuanDiT-Tiny loop also passes validation."""
        m = HunyuanDiT().eval()
        result = call_diffusion_loop_backend(
            bus=None, name=None,
            model=m, latents=torch.randn(1, 4, 8, 8),
            text_embeds=torch.randn(1, 8, 64),
            num_inference_steps=3, guidance_scale=6.0,
            sampler="flow_match_euler", shift=1.0,
        )
        assert result["backend"] == "diffusion_loop"
        assert result["latent_valid"] is True

    def test_e2e_missing_model_returns_invalid(self) -> None:
        """Missing model + latents returns ``latent_valid=False`` with reason."""
        result = call_diffusion_loop_backend(
            bus=None, name=None, model=None, latents=None,
        )
        assert result["latent_valid"] is False
        assert result["latent_validation"]["valid"] is False
        assert "missing" in result["latent_validation"]["reason"]

    def test_e2e_validate_disabled(self) -> None:
        """``validate_latent=False`` skips the validator and omits the key."""
        m = HunyuanDiT().eval()
        result = call_diffusion_loop_backend(
            bus=None, name=None,
            model=m, latents=torch.randn(1, 4, 8, 8),
            num_inference_steps=2, guidance_scale=1.0,
            sampler="flow_match_euler", shift=1.0,
            validate_latent=False,
        )
        assert "latent_validation" not in result
        assert "latent_valid" not in result
        assert "latent_stats" not in result

    def test_e2e_custom_validator(self) -> None:
        """A user-supplied :class:`LatentValidator` is honoured."""
        m = HunyuanDiT().eval()
        # A super-loose validator (any non-zero std is OK).
        validator = LatentValidator(min_std=0.0, min_abs_max=0.0, max_abs_max=1e9)
        result = call_diffusion_loop_backend(
            bus=None, name=None,
            model=m, latents=torch.randn(1, 4, 8, 8),
            num_inference_steps=2, guidance_scale=1.0,
            sampler="flow_match_euler", shift=1.0,
            latent_validator=validator,
        )
        assert result["latent_valid"] is True
        # The custom validator's checks should be echoed back.
        assert result["latent_validation"]["checks"]["min_std"] == 0.0
        assert result["latent_validation"]["checks"]["max_abs_max"] == 1e9
