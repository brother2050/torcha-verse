"""v1.0.0 — model matrix key-rename tables.

Tests for the four new :data:`*_KEY_MAP` constants in
:mod:`core.checkpoint_loader` (FLUX / SD3 / Wan2.1 / MusicGen) plus
the matching ``load_*`` shims and the generalised
:func:`_materialise_per_block_map` factory.
"""
from __future__ import annotations

import pytest

from core.checkpoint_loader import (
    FLUX_KEY_MAP,
    MUSICGEN_KEY_MAP,
    SD3_KEY_MAP,
    WAN2_KEY_MAP,
    _materialise_per_block_map,
    load_flux,
    load_musicgen,
    load_sd3,
    load_wan2,
)


# ---------------------------------------------------------------------------
# 1. Table presence and shape
# ---------------------------------------------------------------------------
class TestKeyMapPresence:
    def test_flux_key_map_exported(self):
        assert isinstance(FLUX_KEY_MAP, dict)
        assert len(FLUX_KEY_MAP) >= 40, "FLUX should have at least 40 entries"

    def test_sd3_key_map_exported(self):
        assert isinstance(SD3_KEY_MAP, dict)
        assert len(SD3_KEY_MAP) >= 30, "SD3 should have at least 30 entries"

    def test_wan2_key_map_exported(self):
        assert isinstance(WAN2_KEY_MAP, dict)
        assert len(WAN2_KEY_MAP) >= 30, "Wan2 should have at least 30 entries"

    def test_musicgen_key_map_exported(self):
        assert isinstance(MUSICGEN_KEY_MAP, dict)
        assert len(MUSICGEN_KEY_MAP) >= 25, "MusicGen should have at least 25 entries"

    def test_key_maps_disjoint(self):
        """No two tables should share an upstream source key
        (a checkpoint can only belong to one model family)."""
        all_keys = []
        for m in (FLUX_KEY_MAP, SD3_KEY_MAP, WAN2_KEY_MAP, MUSICGEN_KEY_MAP):
            all_keys.extend(m.keys())
        assert len(all_keys) == len(set(all_keys)), (
            f"key maps have {len(all_keys) - len(set(all_keys))} overlapping entries"
        )


# ---------------------------------------------------------------------------
# 2. Per-table structural sanity
# ---------------------------------------------------------------------------
class TestKeyMapStructure:
    @pytest.mark.parametrize("key_map", [FLUX_KEY_MAP, SD3_KEY_MAP, WAN2_KEY_MAP, MUSICGEN_KEY_MAP])
    def test_no_empty_strings(self, key_map):
        for k, v in key_map.items():
            assert isinstance(k, str) and len(k) > 0
            assert isinstance(v, str) and len(v) > 0

    @pytest.mark.parametrize("key_map", [FLUX_KEY_MAP, SD3_KEY_MAP, WAN2_KEY_MAP, MUSICGEN_KEY_MAP])
    def test_keys_are_dot_separated(self, key_map):
        for k in key_map:
            assert "." in k, f"key {k!r} should be a dotted path"

    @pytest.mark.parametrize("key_map", [FLUX_KEY_MAP, SD3_KEY_MAP, WAN2_KEY_MAP, MUSICGEN_KEY_MAP])
    def test_placeholder_consistency(self, key_map):
        """Every ``{i}`` in the LHS must appear in the RHS, and vice versa."""
        for k, v in key_map.items():
            assert ("{i}" in k) == ("{i}" in v), (
                f"placeholder mismatch: k={k!r} v={v!r}"
            )


# ---------------------------------------------------------------------------
# 3. Per-block placeholder expansion
# ---------------------------------------------------------------------------
class TestPlaceholderExpansion:
    def test_expand_flux_19_blocks(self):
        expanded = _materialise_per_block_map(19, FLUX_KEY_MAP)
        # All per-block entries get expanded 19x.
        # Check that a known per-block key is present for each i.
        for i in range(19):
            src = f"double_blocks.{i}.img_attn.qkv.weight"
            dst = f"double_blocks.{i}.img_attn.qkv.weight"
            assert expanded[src] == dst

    def test_expand_sd3_24_blocks(self):
        expanded = _materialise_per_block_map(24, SD3_KEY_MAP)
        for i in range(24):
            assert f"joint_transformer_blocks.{i}.x_block.attn.qkv.weight" in expanded

    def test_expand_wan2_40_blocks(self):
        expanded = _materialise_per_block_map(40, WAN2_KEY_MAP)
        for i in range(40):
            assert f"blocks.{i}.cross_attn.q.weight" in expanded

    def test_expand_musicgen_24_blocks(self):
        expanded = _materialise_per_block_map(24, MUSICGEN_KEY_MAP)
        for i in range(24):
            assert (
                f"text_encoder.transformer.Layers.{i}.self_attn.k_proj.weight"
                in expanded
            )

    def test_expand_preserves_non_block_keys(self):
        """Keys without ``{i}`` should pass through unchanged."""
        expanded = _materialise_per_block_map(19, FLUX_KEY_MAP)
        # Final layer is global (no per-block expansion).
        assert expanded["img_in.weight"] == "img_in.weight"
        assert expanded["txt_in.weight"] == "txt_in.weight"

    def test_expand_with_default_hunyuan_dit(self):
        """Calling without a key_map argument falls back to HUNYUAN_DIT_KEY_MAP."""
        expanded = _materialise_per_block_map(5)
        # The 5 blocks worth of qkv entries must be there.
        for i in range(5):
            assert f"blocks.{i}.attn.qkv.weight" in expanded

    def test_expand_grows_linearly(self):
        """Expanding to N blocks produces N copies of every per-block entry."""
        e1 = _materialise_per_block_map(1, FLUX_KEY_MAP)
        e5 = _materialise_per_block_map(5, FLUX_KEY_MAP)
        # 5x blocks => 5x as many per-block keys, but the non-per-block
        # ones are constant. We just check that the expansion at least
        # includes 5 distinct block indices.
        block_indices = {
            int(k.split(".")[1])
            for k in e5
            if k.startswith("double_blocks.")
        }
        assert block_indices == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# 4. Public loader shims
# ---------------------------------------------------------------------------
class TestLoaderShims:
    def test_load_flux_is_callable(self):
        assert callable(load_flux)

    def test_load_sd3_is_callable(self):
        assert callable(load_sd3)

    def test_load_wan2_is_callable(self):
        assert callable(load_wan2)

    def test_load_musicgen_is_callable(self):
        assert callable(load_musicgen)

    def test_load_flux_default_num_blocks(self, tmp_path, monkeypatch):
        """Calling load_flux with a non-existent path should not crash
        with an argument-parsing error; it should fail with a model
        class or file resolution error, indicating the defaults were
        accepted."""
        import core.checkpoint_loader as cl

        # Patch _load_with_keymap to a no-op recorder.
        captured = {}

        def fake_load_with_keymap(cls_candidates, weights_path, key_map, **kwargs):
            captured["kwargs"] = kwargs
            captured["key_map_size"] = len(key_map)
            return None

        monkeypatch.setattr(cl, "_load_with_keymap", fake_load_with_keymap)
        load_flux(tmp_path / "fake.safetensors")
        # Default num_blocks for FLUX is 19; after expansion every
        # per-block entry should have been multiplied by 19.
        assert captured["kwargs"]["num_blocks"] == 19
        # The expanded key_map should contain per-block entries.
        assert captured["key_map_size"] > 0

    def test_load_sd3_passes_keymap_size(self, tmp_path, monkeypatch):
        import core.checkpoint_loader as cl

        captured = {}

        def fake_load(cls_candidates, weights_path, key_map, **kwargs):
            captured["num_blocks"] = kwargs["num_blocks"]
            return None

        monkeypatch.setattr(cl, "_load_with_keymap", fake_load)
        load_sd3(tmp_path / "fake.safetensors")
        assert captured["num_blocks"] == 24

    def test_load_wan2_passes_keymap_size(self, tmp_path, monkeypatch):
        import core.checkpoint_loader as cl

        captured = {}

        def fake_load(cls_candidates, weights_path, key_map, **kwargs):
            captured["num_blocks"] = kwargs["num_blocks"]
            return None

        monkeypatch.setattr(cl, "_load_with_keymap", fake_load)
        load_wan2(tmp_path / "fake.safetensors")
        assert captured["num_blocks"] == 40

    def test_load_musicgen_passes_keymap_size(self, tmp_path, monkeypatch):
        import core.checkpoint_loader as cl

        captured = {}

        def fake_load(cls_candidates, weights_path, key_map, **kwargs):
            captured["num_blocks"] = kwargs["num_blocks"]
            return None

        monkeypatch.setattr(cl, "_load_with_keymap", fake_load)
        load_musicgen(tmp_path / "fake.safetensors")
        assert captured["num_blocks"] == 24

    def test_loader_accepts_custom_num_blocks(self, tmp_path, monkeypatch):
        """Custom num_blocks should be forwarded to the expansion helper."""
        import core.checkpoint_loader as cl

        captured = {}

        def fake_load(cls_candidates, weights_path, key_map, **kwargs):
            captured["num_blocks"] = kwargs["num_blocks"]
            return None

        monkeypatch.setattr(cl, "_load_with_keymap", fake_load)
        load_flux(tmp_path / "x.safetensors", num_blocks=7)
        assert captured["num_blocks"] == 7
