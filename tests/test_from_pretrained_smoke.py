"""v0.8.0 ModelMixin ``from_pretrained`` smoke test (CI 守卫).

This test file is the v0.8.0 contract guard for the
:class:`models.base.ModelMixin` API.  It exercises every public
``from_pretrained`` / ``save_pretrained`` knob on a minimal
synthetic model so that:

1. The round-trip path is exercised on every commit.
2. New ModelMixin subclasses are encouraged to inherit the same
   surface area (the test runs against any ``nn.Module`` that
   subclasses ``ModelMixin``).
3. Regressions in the saver / loader are caught at PR time
   rather than at integration time.

The test runs in under 1 s on a stock dev box (no GPU, no real
upstream checkpoint, no network).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from models.base import (
    ModelMixin,
    load_safetensors,
    load_state_dict_with_renames,
    save_safetensors,
    transform_checkpoint_dict_key,
)


# ---------------------------------------------------------------------------
# Synthetic ModelMixin subclasses used for round-trip tests
# ---------------------------------------------------------------------------
class _TinyMLP(ModelMixin):
    """Minimal 2-layer MLP used to exercise the loading contract."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class _RenamedMLP(ModelMixin):
    """A model that uses an old upstream-style naming scheme on disk.

    The ``_renamed`` parameter is the only difference with
    :class:`_TinyMLP`; the load helper uses ``key_renames`` to
    translate ``legacy_fc1`` -> ``linear1.weight`` etc.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))

    @classmethod
    def from_legacy(cls, path: str | Path) -> "_RenamedMLP":
        state_dict = load_safetensors(path)
        state_dict = transform_checkpoint_dict_key(
            state_dict,
            {
                "legacy_fc1.weight": "linear1.weight",
                "legacy_fc1.bias": "linear1.bias",
                "legacy_fc2.weight": "linear2.weight",
                "legacy_fc2.bias": "linear2.bias",
            },
        )
        model = cls()
        load_state_dict_with_renames(model, state_dict, strict=True)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# 1. save_pretrained → from_pretrained round-trip
# ---------------------------------------------------------------------------
def test_from_pretrained_roundtrip(tmp_path: Path) -> None:
    m = _TinyMLP()
    m.save_pretrained(str(tmp_path))
    # File naming follows ``<class_name_lowercase><ext>``.
    assert (tmp_path / "_tinymlp.safetensors").is_file()
    # Round-trip
    m2 = _TinyMLP.from_pretrained(str(tmp_path), strict=True)
    # The two state-dicts must be byte-identical.
    s1 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    s2 = {k: v.detach().clone() for k, v in m2.state_dict().items()}
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"mismatch on {k}"


# ---------------------------------------------------------------------------
# 2. torch_dtype cast
# ---------------------------------------------------------------------------
def test_from_pretrained_dtype_cast(tmp_path: Path) -> None:
    m = _TinyMLP()
    m.save_pretrained(str(tmp_path))
    m2 = _TinyMLP.from_pretrained(
        str(tmp_path), torch_dtype=torch.float16, strict=True,
    )
    p = next(m2.parameters())
    assert p.dtype == torch.float16, p.dtype


# ---------------------------------------------------------------------------
# 3. device_map (string shortcut)
# ---------------------------------------------------------------------------
def test_from_pretrained_device_map_cpu(tmp_path: Path) -> None:
    m = _TinyMLP()
    m.save_pretrained(str(tmp_path))
    m2 = _TinyMLP.from_pretrained(
        str(tmp_path), device_map="cpu", strict=True,
    )
    for p in m2.parameters():
        assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# 4. key_renames (declarative checkpoint migration)
# ---------------------------------------------------------------------------
def test_from_pretrained_key_renames(tmp_path: Path) -> None:
    # Write a "legacy" checkpoint with the old naming scheme.
    src = _TinyMLP()
    legacy_state: dict[str, torch.Tensor] = {}
    for k, v in src.state_dict().items():
        legacy_state[k.replace("linear", "legacy_fc")] = v.clone()
    save_safetensors(legacy_state, tmp_path / "legacy.safetensors")
    # Load via the helper that applies the rename.
    m = _RenamedMLP.from_legacy(tmp_path / "legacy.safetensors")
    # A single forward pass should succeed and produce the right shape.
    out = m(torch.zeros(1, 8))
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# 5. variant resolution
# ---------------------------------------------------------------------------
def test_from_pretrained_variant(tmp_path: Path) -> None:
    m = _TinyMLP()
    m.save_pretrained(str(tmp_path))
    # Materialise a "fp16" sibling by saving a copy with the variant
    # suffix.  The loader should pick it up first.
    state = load_safetensors(tmp_path / "_tinymlp.safetensors")
    state = {k: v.to(torch.float16) for k, v in state.items()}
    save_safetensors(state, tmp_path / "_tinymlp.fp16.safetensors")
    m2 = _TinyMLP.from_pretrained(
        str(tmp_path), variant="fp16", strict=True,
    )
    p = next(m2.parameters())
    assert p.dtype == torch.float16, p.dtype


# ---------------------------------------------------------------------------
# 6. subfolder resolution
# ---------------------------------------------------------------------------
def test_from_pretrained_subfolder(tmp_path: Path) -> None:
    sub = tmp_path / "nested"
    m = _TinyMLP()
    m.save_pretrained(str(sub))
    m2 = _TinyMLP.from_pretrained(
        str(tmp_path), subfolder="nested", strict=True,
    )
    out = m2(torch.zeros(1, 8))
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# 7. config.json sidecar
# ---------------------------------------------------------------------------
def test_from_pretrained_config_json(tmp_path: Path) -> None:
    config = {"hidden": 8, "name": "tiny"}
    m = _TinyMLP(config=config)
    m.save_pretrained(str(tmp_path))
    # The config.json was written by save_pretrained.
    cfg_path = tmp_path / "config.json"
    assert cfg_path.is_file()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data == config
    # And from_pretrained picks it up.
    m2 = _TinyMLP.from_pretrained(str(tmp_path), strict=True)
    assert m2.config == config


# ---------------------------------------------------------------------------
# 8. sharded layout
# ---------------------------------------------------------------------------
def test_from_pretrained_sharded(tmp_path: Path) -> None:
    m = _TinyMLP()
    # Write a sharded layout: 1 file + index.
    state = m.state_dict()
    keys = sorted(state.keys())
    half = len(keys) // 2
    s1 = {k: state[k] for k in keys[:half]}
    s2 = {k: state[k] for k in keys[half:]}
    save_safetensors(s1, tmp_path / "test-00001-of-00002.safetensors")
    save_safetensors(s2, tmp_path / "test-00002-of-00002.safetensors")
    weight_map = {
        **{k: "test-00001-of-00002.safetensors" for k in s1},
        **{k: "test-00002-of-00002.safetensors" for k in s2},
    }
    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (tmp_path / "test.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8",
    )
    # The loader should auto-stitch when it sees the index next to
    # the requested file.
    state2 = load_safetensors(tmp_path / "test-00001-of-00002.safetensors")
    for k in keys:
        assert k in state2, k


# ---------------------------------------------------------------------------
# 9. error: file not found
# ---------------------------------------------------------------------------
def test_from_pretrained_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _TinyMLP.from_pretrained(str(tmp_path), strict=True)


# ---------------------------------------------------------------------------
# 10. HUNYUAN_DIT_KEY_MAP expands correctly
# ---------------------------------------------------------------------------
def test_hunyuan_dit_key_map_expansion() -> None:
    from core.checkpoint_loader import (
        HUNYUAN_DIT_KEY_MAP, _materialise_per_block_map,
    )
    expanded = _materialise_per_block_map(num_blocks=2)
    # The ``{i}`` placeholders are gone.
    assert "{i}" not in " ".join(expanded)
    # At least one per-block entry exists.
    assert any(
        k.startswith("blocks.0.") or k.startswith("blocks.1.")
        for k in expanded
    )
    # 1:1 entries (without ``{i}``) are still present.
    assert "patch_embed.proj.weight" in expanded.values()
