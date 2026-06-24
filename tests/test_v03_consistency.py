"""Tests for v0.3.0 consistency framework (profiles, scores, pipeline).

Covers ConsistencyProfile weight validation, ConsistencyManager CRUD,
ConsistencyScore round-trip serialisation and ConsistencyPipeline
construction.
"""
from __future__ import annotations

import pytest

from consistency.profile import ConsistencyManager, ConsistencyProfile
from consistency.score import ConsistencyScore, ScoreCalculator
from consistency.pipeline import ConsistencyPipeline


# ---------------------------------------------------------------------------
# ConsistencyProfile
# ---------------------------------------------------------------------------
class TestConsistencyProfile:
    """Weight-range validation and serialisation."""

    def test_default_weights_in_range(self):
        """The default profile has all weights in [0, 1]."""
        p = ConsistencyProfile()
        assert 0.0 <= p.character_weight <= 1.0
        assert 0.0 <= p.outfit_weight <= 1.0
        assert 0.0 <= p.scene_weight <= 1.0
        assert 0.0 <= p.depth_weight <= 1.0

    def test_weight_below_zero_rejected(self):
        """A negative weight raises ValueError."""
        with pytest.raises(ValueError):
            ConsistencyProfile(character_weight=-0.1)

    def test_weight_above_one_rejected(self):
        """A weight > 1 raises ValueError."""
        with pytest.raises(ValueError):
            ConsistencyProfile(outfit_weight=1.5)

    def test_to_dict_from_dict_roundtrip(self):
        """Profile survives a to_dict -> from_dict round-trip."""
        original = ConsistencyProfile(
            character_weight=0.9,
            outfit_weight=0.3,
            scene_weight=0.7,
            depth_weight=0.2,
            temporal_consistency=True,
            drift_threshold=0.25,
            reframe_interval=4,
        )
        d = original.to_dict()
        restored = ConsistencyProfile.from_dict(d)
        assert restored.character_weight == 0.9
        assert restored.outfit_weight == 0.3
        assert restored.scene_weight == 0.7
        assert restored.depth_weight == 0.2
        assert restored.temporal_consistency is True
        assert restored.drift_threshold == 0.25
        assert restored.reframe_interval == 4

    def test_drift_threshold_range(self):
        """drift_threshold must be in [0, 1]."""
        with pytest.raises(ValueError):
            ConsistencyProfile(drift_threshold=-0.5)
        with pytest.raises(ValueError):
            ConsistencyProfile(drift_threshold=2.0)


# ---------------------------------------------------------------------------
# ConsistencyManager
# ---------------------------------------------------------------------------
class TestConsistencyManager:
    """CRUD operations on named profiles."""

    def test_create_and_get_profile(self):
        """create_profile() registers a profile; get_profile() retrieves it."""
        mgr = ConsistencyManager()
        profile = mgr.create_profile("cinematic", character_weight=0.9)
        assert isinstance(profile, ConsistencyProfile)
        assert mgr.get_profile("cinematic") is profile

    def test_create_duplicate_raises(self):
        """Creating a profile with an existing name raises ValueError."""
        mgr = ConsistencyManager()
        mgr.create_profile("dup", character_weight=0.5)
        with pytest.raises(ValueError):
            mgr.create_profile("dup", character_weight=0.6)

    def test_get_unknown_raises(self):
        """get_profile() raises KeyError for an unknown name."""
        mgr = ConsistencyManager()
        with pytest.raises(KeyError):
            mgr.get_profile("nonexistent")

    def test_list_profiles(self):
        """list_profiles() returns sorted profile names."""
        mgr = ConsistencyManager()
        mgr.create_profile("beta", character_weight=0.5)
        mgr.create_profile("alpha", character_weight=0.5)
        names = mgr.list_profiles()
        assert names == ["alpha", "beta"]

    def test_remove_profile(self):
        """remove_profile() returns True when found, False otherwise."""
        mgr = ConsistencyManager()
        mgr.create_profile("temp", character_weight=0.5)
        assert mgr.remove_profile("temp") is True
        assert mgr.remove_profile("temp") is False


# ---------------------------------------------------------------------------
# ConsistencyScore
# ---------------------------------------------------------------------------
class TestConsistencyScore:
    """Score dataclass round-trip."""

    def test_to_dict_from_dict_roundtrip(self):
        """ConsistencyScore survives a to_dict -> from_dict round-trip."""
        original = ConsistencyScore(
            character_score=0.9,
            outfit_score=0.8,
            scene_score=0.7,
            depth_score=0.6,
            temporal_score=0.5,
            overall=0.75,
        )
        d = original.to_dict()
        restored = ConsistencyScore.from_dict(d)
        assert restored.character_score == 0.9
        assert restored.outfit_score == 0.8
        assert restored.scene_score == 0.7
        assert restored.depth_score == 0.6
        assert restored.temporal_score == 0.5
        assert restored.overall == 0.75

    def test_from_dict_defaults(self):
        """from_dict() uses 0.0 for missing keys."""
        restored = ConsistencyScore.from_dict({})
        assert restored.character_score == 0.0
        assert restored.overall == 0.0


# ---------------------------------------------------------------------------
# ConsistencyPipeline
# ---------------------------------------------------------------------------
class TestConsistencyPipeline:
    """Pipeline construction and basic generate."""

    def test_construct_with_profile_only(self):
        """A pipeline can be constructed with just a profile."""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        assert pipe.profile is profile
        assert pipe.character is None
        assert pipe.outfit is None

    def test_generate_returns_expected_keys(self):
        """generate() returns image, consistency_scores and metadata."""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        result = pipe.generate("a cat", width=64, height=64)
        assert "image" in result
        assert "consistency_scores" in result
        assert "metadata" in result

    def test_generate_invalid_dimensions_raises(self):
        """generate() raises ValueError for out-of-range dimensions."""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        with pytest.raises(ValueError):
            pipe.generate("test", width=0, height=64)
