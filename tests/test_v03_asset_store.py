"""Tests for v0.3.0 AssetStore and asset serialisation round-trips.

Covers the core CRUD surface of :class:`AssetStore` (put / get / exists /
delete / list / search / fork / verify) as well as the ``to_dict`` /
``from_dict`` round-trip for every concrete :class:`Asset` subclass.
All tests use the ``tmp_path`` fixture so that no on-disk state leaks
between test runs.
"""
from __future__ import annotations

import pytest

from assets.base import Asset, AssetRef
from assets.store import AssetStore
from assets.types import AssetStatus, AssetType
from assets.model_asset import (
    CharacterAsset,
    DepthAsset,
    ModelAsset,
    OutfitAsset,
    SceneAsset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path):
    """Return a fresh AssetStore rooted in *tmp_path*."""
    s = AssetStore(base_dir=tmp_path / "assets")
    yield s
    s.close()


@pytest.fixture()
def content_file(tmp_path):
    """Create a small dummy content file and return its path."""
    p = tmp_path / "content.bin"
    p.write_bytes(b"hello-torcha-verse")
    return p


# ---------------------------------------------------------------------------
# AssetStore core operations
# ---------------------------------------------------------------------------
class TestAssetStoreCRUD:
    """put / get / exists / delete / list / search / fork / verify."""

    def test_put_and_get(self, store, content_file):
        """put() returns an AssetRef; get() retrieves the asset + path."""
        asset = ModelAsset(
            id="sd-xl",
            name="Stable Diffusion XL",
            architecture="dit",
            format="safetensors",
            size_gb=6.5,
            source="local",
        )
        ref = store.put(asset, content_file)
        assert isinstance(ref, AssetRef)
        assert ref.asset_id == "sd-xl"
        assert ref.asset_type == AssetType.MODEL

        retrieved, path = store.get(ref)
        assert retrieved.id == "sd-xl"
        assert retrieved.name == "Stable Diffusion XL"
        assert path.exists()

    def test_exists(self, store, content_file):
        """exists() returns True for a stored ref, False for a bogus one."""
        asset = ModelAsset(id="m1", name="Model 1", architecture="unet")
        ref = store.put(asset, content_file)
        assert store.exists(ref) is True

        bogus = AssetRef(
            asset_id="nonexistent",
            asset_type=AssetType.MODEL,
            revision="r1",
            content_hash="0" * 64,
        )
        assert store.exists(bogus) is False

    def test_delete_soft(self, store, content_file):
        """delete() soft-deletes (archives) the asset."""
        asset = ModelAsset(id="del-me", name="Delete Me", architecture="unet")
        ref = store.put(asset, content_file)
        assert store.delete(ref) is True

        archived = store.list(status=AssetStatus.ARCHIVED)
        assert any(a.id == "del-me" for a in archived)

    def test_list_with_filters(self, store, content_file):
        """list() supports filtering by type, tags and status."""
        m1 = ModelAsset(
            id="m-list-1", name="List Model 1", architecture="unet",
            tags=["anime", "base"],
        )
        m2 = ModelAsset(
            id="m-list-2", name="List Model 2", architecture="dit",
            tags=["photo"],
        )
        store.put(m1, content_file)
        store.put(m2, content_file)

        all_models = store.list(asset_type=AssetType.MODEL)
        assert len(all_models) >= 2

        anime = store.list(tags=["anime"])
        assert any(a.id == "m-list-1" for a in anime)

    def test_search(self, store, content_file):
        """search() finds assets by substring in name/description/tags."""
        asset = ModelAsset(
            id="searchable",
            name="My Awesome Model",
            architecture="unet",
            description="A great model for anime art",
            tags=["anime"],
        )
        store.put(asset, content_file)

        results = store.search("awesome")
        assert len(results) >= 1
        assert results[0].id == "searchable"

        results = store.search("anime")
        assert any(a.id == "searchable" for a in results)

    def test_fork(self, store, content_file):
        """fork() creates a new asset referencing the same content blob."""
        original = ModelAsset(
            id="orig", name="Original", architecture="unet",
        )
        ref = store.put(original, content_file)
        forked_ref = store.fork(ref, "Forked Copy")

        assert forked_ref.asset_id != "orig"
        forked_asset, forked_path = store.get(forked_ref)
        assert forked_asset.name == "Forked Copy"
        # Content is shared (content-addressed).
        _, orig_path = store.get(ref)
        assert forked_path == orig_path

    def test_verify(self, store, content_file):
        """verify() returns True when the content hash matches."""
        asset = ModelAsset(id="verify-me", name="Verify", architecture="dit")
        ref = store.put(asset, content_file)
        assert store.verify(ref) is True

    def test_get_missing_raises(self, store):
        """get() raises KeyError for a non-existent asset."""
        bogus = AssetRef(
            asset_id="ghost",
            asset_type=AssetType.MODEL,
            revision="r1",
            content_hash="0" * 64,
        )
        with pytest.raises(KeyError):
            store.get(bogus)

    def test_put_missing_content_raises(self, store, tmp_path):
        """put() raises FileNotFoundError when the content file is missing."""
        asset = ModelAsset(id="bad", name="Bad", architecture="unet")
        with pytest.raises(FileNotFoundError):
            store.put(asset, tmp_path / "does-not-exist.bin")


# ---------------------------------------------------------------------------
# Asset subclass round-trip tests
# ---------------------------------------------------------------------------
class TestAssetRoundTrip:
    """to_dict / from_dict round-trips for every asset subclass."""

    def test_model_asset_roundtrip(self):
        """ModelAsset survives a to_dict -> from_dict round-trip."""
        original = ModelAsset(
            id="rt-model",
            name="Round Trip Model",
            architecture="dit",
            format="safetensors",
            size_gb=4.2,
            source="huggingface",
            config={"layers": 32, "dim": 2048},
            tags=["test", "roundtrip"],
        )
        d = original.to_dict()
        restored = Asset.from_dict(d)
        assert isinstance(restored, ModelAsset)
        assert restored.id == original.id
        assert restored.architecture == "dit"
        assert restored.format == "safetensors"
        assert restored.size_gb == 4.2
        assert restored.source == "huggingface"
        assert restored.config == {"layers": 32, "dim": 2048}

    def test_character_asset_roundtrip(self):
        """CharacterAsset survives a to_dict -> from_dict round-trip."""
        original = CharacterAsset(
            id="rt-char",
            name="Round Trip Character",
            consistency_seed=12345,
            tags=["hero"],
        )
        d = original.to_dict()
        restored = Asset.from_dict(d)
        assert isinstance(restored, CharacterAsset)
        assert restored.id == original.id
        assert restored.consistency_seed == 12345

    def test_outfit_asset_roundtrip(self):
        """OutfitAsset survives a to_dict -> from_dict round-trip."""
        original = OutfitAsset(
            id="rt-outfit",
            name="Round Trip Outfit",
            tags=["casual"],
        )
        d = original.to_dict()
        restored = Asset.from_dict(d)
        assert isinstance(restored, OutfitAsset)
        assert restored.id == original.id

    def test_scene_asset_roundtrip(self):
        """SceneAsset survives a to_dict -> from_dict round-trip."""
        original = SceneAsset(
            id="rt-scene",
            name="Round Trip Scene",
            tags=["indoor"],
        )
        d = original.to_dict()
        restored = Asset.from_dict(d)
        assert isinstance(restored, SceneAsset)
        assert restored.id == original.id

    def test_depth_asset_roundtrip(self):
        """DepthAsset survives a to_dict -> from_dict round-trip."""
        original = DepthAsset(
            id="rt-depth",
            name="Round Trip Depth",
            method="zoedepth",
            tags=["depth"],
        )
        d = original.to_dict()
        restored = Asset.from_dict(d)
        assert isinstance(restored, DepthAsset)
        assert restored.id == original.id
        assert restored.method == "zoedepth"
