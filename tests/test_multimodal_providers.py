"""Tests for the v0.4.x P0 multi-modal local-torch providers.

Covers the four new providers introduced in this milestone:
:class:`LocalTorchImageProvider`,
:class:`LocalTorchAudioProvider`,
:class:`LocalTorchVideoProvider`,
:class:`LocalTorchMultimodalProvider`, plus the
:mod:`models.interfaces.media_providers` echo / protocol
definitions and the matching ``fetch_and_load_*`` /
``get_default_*_provider`` factory entry points.

The tests are deliberately **lightweight**: they exercise the
end-to-end forward pass (UNet + VAE / TTS + HiFi-GAN /
VideoDiT + VideoVAE / OmniModel + TinyCausalLM) on the TINY
presets and verify the contract (output shape, dtype, key
presence) without trying to assert anything about the *quality*
of the output.  The aim is to prove the pipelines can be
chained end-to-end on a single CPU thread within a second or
two -- that is the v0.4.x P0 CI smoke-test goal.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# `models.providers` re-exports the public surface we want to
# assert against.  Importing it has the side-effect of loading
# the four new ``local_*`` modules; we are *not* triggering a
# default-singleton build here (the factories are only
# consulted when ``get_default_*_provider()`` is called).
from models.interfaces.media_providers import (  # noqa: E402
    AudioProvider,
    EchoAudioProvider,
    EchoImageProvider,
    EchoMultimodalProvider,
    EchoVideoProvider,
    ImageProvider,
    MultimodalProvider,
    VideoProvider,
)
from models.providers import (  # noqa: E402
    SMALL_AUDIO_CONFIG,
    SMALL_IMAGE_CONFIG,
    SMALL_MULTIMODAL_CONFIG,
    SMALL_VIDEO_CONFIG,
    TINY_AUDIO_CONFIG,
    TINY_IMAGE_CONFIG,
    TINY_MULTIMODAL_CONFIG,
    TINY_VIDEO_CONFIG,
    fetch_and_load_audio,
    fetch_and_load_image,
    fetch_and_load_omni,
    fetch_and_load_video,
    get_default_audio_provider,
    get_default_image_provider,
    get_default_omni_provider,
    get_default_video_provider,
)
from models.providers.local_audio import LocalTorchAudioProvider  # noqa: E402
from models.providers.local_image import LocalTorchImageProvider  # noqa: E402
from models.providers.local_multimodal import LocalTorchMultimodalProvider  # noqa: E402
from models.providers.local_video import LocalTorchVideoProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Protocol + echo
# ---------------------------------------------------------------------------
class TestEchoProviders:
    """The :class:`Echo*Provider` classes are the v0.4.0 reference
    implementations of the four new provider protocols.  They
    must satisfy the protocol (via ``runtime_checkable``) and
    produce the expected output shape / dtype for the unit
    tests below.
    """

    def test_echo_image_satisfies_protocol(self) -> None:
        p = EchoImageProvider()
        assert isinstance(p, ImageProvider)
        out = p.generate("a tiny cat", width=8, height=8)
        assert "image" in out
        assert tuple(out["image"].shape) == (3, 8, 8)
        assert out["image"].dtype == torch.float32
        assert out["width"] == 8
        assert out["height"] == 8

    def test_echo_audio_satisfies_protocol(self) -> None:
        p = EchoAudioProvider()
        assert isinstance(p, AudioProvider)
        out = p.generate("hello", sample_rate=16000, duration_s=0.1)
        assert "waveform" in out
        assert tuple(out["waveform"].shape) == (1, 1600)
        assert out["sample_rate"] == 16000

    def test_echo_video_satisfies_protocol(self) -> None:
        p = EchoVideoProvider()
        assert isinstance(p, VideoProvider)
        out = p.generate("a flying bird", num_frames=2, height=8, width=8)
        assert "frames" in out
        assert tuple(out["frames"].shape) == (2, 3, 8, 8)
        assert out["num_frames"] == 2

    def test_echo_multimodal_satisfies_protocol(self) -> None:
        p = EchoMultimodalProvider()
        assert isinstance(p, MultimodalProvider)
        out = p.generate("hello world")
        assert "text" in out
        # The text should at least contain the prompt.
        assert "hello" in out["text"].lower()


# ---------------------------------------------------------------------------
# LocalTorchImageProvider
# ---------------------------------------------------------------------------
class TestLocalTorchImageProvider:
    """End-to-end UNet + VAE + CLIP forward pass on TINY preset."""

    def test_from_random_constructs(self) -> None:
        p = LocalTorchImageProvider.from_random()
        # TINY preset is small enough to construct in well under 5s
        # on CI; the param count is at most a few million.
        n = p.num_parameters()
        assert 0 < n < 50_000_000
        assert p.config.name == "tiny"

    def test_generate_shape_and_keys(self) -> None:
        p = LocalTorchImageProvider.from_random()
        t0 = time.time()
        out = p.generate(
            "a tiny cat", width=16, height=16, steps=2, seed=42,
        )
        elapsed = time.time() - t0
        assert elapsed < 30, "tiny image generate took too long: {:.2f}s".format(elapsed)
        assert "image" in out
        assert tuple(out["image"].shape) == (3, 16, 16)
        assert out["image"].dtype == torch.float32
        # Image is in [0, 1].
        assert 0.0 <= float(out["image"].min()) <= 1.0
        assert 0.0 <= float(out["image"].max()) <= 1.0
        assert out["seed"] == 42
        assert out["steps"] == 2
        assert out["width"] == 16
        assert out["height"] == 16

    def test_seed_determinism(self) -> None:
        p = LocalTorchImageProvider.from_random()
        a = p.generate("hello", width=16, height=16, steps=2, seed=7)
        b = p.generate("hello", width=16, height=16, steps=2, seed=7)
        # Same seed -> same noise -> same residual -> same image.
        assert torch.allclose(a["image"], b["image"])

    def test_from_file_round_trip(self, tmp_path: Path) -> None:
        p = LocalTorchImageProvider.from_random()
        out_file = tmp_path / "img.pt"
        p.save(out_file)
        # Reload and check the model is still usable.
        q = LocalTorchImageProvider.from_file(out_file)
        # Shapes match (we re-init via TINY_IMAGE_CONFIG, so the
        # parameter count is the same).
        assert p.num_parameters() == q.num_parameters()
        out = q.generate("hello", width=16, height=16, steps=1, seed=1)
        assert tuple(out["image"].shape) == (3, 16, 16)


# ---------------------------------------------------------------------------
# LocalTorchAudioProvider
# ---------------------------------------------------------------------------
class TestLocalTorchAudioProvider:
    """End-to-end TTS-Transformer + HiFi-GAN forward pass on TINY preset."""

    def test_from_random_constructs(self) -> None:
        p = LocalTorchAudioProvider.from_random()
        assert p.config.name == "tiny"
        n = p.num_parameters()
        assert 0 < n < 50_000_000

    def test_generate_shape_and_keys(self) -> None:
        p = LocalTorchAudioProvider.from_random()
        t0 = time.time()
        out = p.generate("hello", duration_s=0.1, sample_rate=16000)
        elapsed = time.time() - t0
        assert elapsed < 30
        assert "waveform" in out
        # 0.1s at 16kHz -> 1600 samples; upsample factor is
        # 16 (4 * 4), so the produced waveform will be
        # ``mel_len * 16`` long.  We do not assert a fixed
        # number of samples (it is implementation-defined) --
        # only that the shape is ``(1, N)``.
        wav = out["waveform"]
        assert wav.ndim == 2
        assert wav.shape[0] == 1
        assert wav.shape[1] > 0
        assert out["sample_rate"] == 16000
        # The waveform should be finite (no NaN/Inf).
        assert torch.isfinite(wav).all()

    def test_from_file_round_trip(self, tmp_path: Path) -> None:
        p = LocalTorchAudioProvider.from_random()
        out_file = tmp_path / "aud.pt"
        p.save(out_file)
        q = LocalTorchAudioProvider.from_file(out_file)
        assert p.num_parameters() == q.num_parameters()


# ---------------------------------------------------------------------------
# LocalTorchVideoProvider
# ---------------------------------------------------------------------------
class TestLocalTorchVideoProvider:
    """End-to-end VideoDiT + VideoVAE forward pass on TINY preset."""

    def test_from_random_constructs(self) -> None:
        p = LocalTorchVideoProvider.from_random()
        assert p.config.name == "tiny"
        n = p.num_parameters()
        assert 0 < n < 50_000_000

    def test_generate_shape_and_keys(self) -> None:
        p = LocalTorchVideoProvider.from_random()
        t0 = time.time()
        out = p.generate(
            "a flying bird", num_frames=2, height=8, width=8, steps=1, seed=42,
        )
        elapsed = time.time() - t0
        assert elapsed < 30
        assert "frames" in out
        frames = out["frames"]
        assert frames.ndim == 4
        # (T, 3, H, W).
        assert frames.shape[1] == 3
        assert frames.shape[2] == 8
        assert frames.shape[3] == 8
        assert frames.shape[0] >= 1  # at least 1 frame
        assert out["fps"] == 8
        assert out["seed"] == 42

    def test_from_file_round_trip(self, tmp_path: Path) -> None:
        p = LocalTorchVideoProvider.from_random()
        out_file = tmp_path / "vid.pt"
        p.save(out_file)
        q = LocalTorchVideoProvider.from_file(out_file)
        assert p.num_parameters() == q.num_parameters()


# ---------------------------------------------------------------------------
# LocalTorchMultimodalProvider
# ---------------------------------------------------------------------------
class TestLocalTorchMultimodalProvider:
    """End-to-end OmniModel + TinyCausalLM forward pass on TINY preset."""

    def test_from_random_constructs(self) -> None:
        p = LocalTorchMultimodalProvider.from_random()
        assert p.config.name == "tiny"
        n = p.num_parameters()
        assert 0 < n < 50_000_000

    def test_text_only_generation(self) -> None:
        p = LocalTorchMultimodalProvider.from_random()
        t0 = time.time()
        out = p.generate("hello", max_new_tokens=4, seed=42)
        elapsed = time.time() - t0
        assert elapsed < 30
        assert "text" in out
        # Even with random init, the LM produces a 4-token
        # byte string; the bytes must all be < 256.
        for ch in out["text"]:
            assert 0 <= ord(ch) < 256

    def test_text_plus_image_plus_audio(self) -> None:
        p = LocalTorchMultimodalProvider.from_random()
        out = p.generate(
            {
                "text": "a cat",
                "image": torch.rand(3, 8, 8),
                "audio": torch.zeros(1, 64),
            },
            max_new_tokens=2,
        )
        assert "text" in out
        # The vision tower was actually invoked (returns
        # ``(features, cls_token)``; we record the feature
        # shape).
        assert "image_emb_shape" in out
        # The audio tower was actually invoked.
        assert "audio_emb_shape" in out
        # Both embedding shapes are tuples.
        assert isinstance(out["image_emb_shape"], tuple)
        assert isinstance(out["audio_emb_shape"], tuple)

    def test_from_file_round_trip(self, tmp_path: Path) -> None:
        p = LocalTorchMultimodalProvider.from_random()
        out_file = tmp_path / "omni.pt"
        p.save(out_file)
        q = LocalTorchMultimodalProvider.from_file(out_file)
        assert p.num_parameters() == q.num_parameters()


# ---------------------------------------------------------------------------
# factory.py entry points
# ---------------------------------------------------------------------------
class TestFactoryEntrypoints:
    """The :func:`fetch_and_load_*` and :func:`get_default_*_provider`
    helpers are the public API the L4 nodes and CLI scripts call.
    """

    def test_fetch_image(self) -> None:
        p = fetch_and_load_image()
        assert isinstance(p, LocalTorchImageProvider)
        assert p.config.name == "tiny"

    def test_fetch_audio(self) -> None:
        p = fetch_and_load_audio()
        assert isinstance(p, LocalTorchAudioProvider)

    def test_fetch_video(self) -> None:
        p = fetch_and_load_video()
        assert isinstance(p, LocalTorchVideoProvider)

    def test_fetch_omni(self) -> None:
        p = fetch_and_load_omni()
        assert isinstance(p, LocalTorchMultimodalProvider)

    def test_get_default_image_is_singleton(self) -> None:
        a = get_default_image_provider()
        b = get_default_image_provider()
        assert a is b

    def test_get_default_audio_is_singleton(self) -> None:
        a = get_default_audio_provider()
        b = get_default_audio_provider()
        assert a is b

    def test_get_default_video_is_singleton(self) -> None:
        a = get_default_video_provider()
        b = get_default_video_provider()
        assert a is b

    def test_get_default_omni_is_singleton(self) -> None:
        a = get_default_omni_provider()
        b = get_default_omni_provider()
        assert a is b


# ---------------------------------------------------------------------------
# Preset sanity checks
# ---------------------------------------------------------------------------
class TestPresets:
    """The TINY / SMALL presets should be coherent: same TINY_CONFIG
    -> same param count when constructed twice."""

    def test_tiny_image_is_stable(self) -> None:
        a = LocalTorchImageProvider.from_random(TINY_IMAGE_CONFIG)
        b = LocalTorchImageProvider.from_random(TINY_IMAGE_CONFIG)
        assert a.num_parameters() == b.num_parameters()

    def test_tiny_audio_is_stable(self) -> None:
        a = LocalTorchAudioProvider.from_random(TINY_AUDIO_CONFIG)
        b = LocalTorchAudioProvider.from_random(TINY_AUDIO_CONFIG)
        assert a.num_parameters() == b.num_parameters()

    def test_tiny_video_is_stable(self) -> None:
        a = LocalTorchVideoProvider.from_random(TINY_VIDEO_CONFIG)
        b = LocalTorchVideoProvider.from_random(TINY_VIDEO_CONFIG)
        assert a.num_parameters() == b.num_parameters()

    def test_tiny_omni_is_stable(self) -> None:
        a = LocalTorchMultimodalProvider.from_random(TINY_MULTIMODAL_CONFIG)
        b = LocalTorchMultimodalProvider.from_random(TINY_MULTIMODAL_CONFIG)
        assert a.num_parameters() == b.num_parameters()

    def test_small_presets_constructible(self) -> None:
        # Construction-only smoke test -- we do *not* run a full
        # forward pass for SMALL presets in CI (they are larger).
        p_img = LocalTorchImageProvider.from_random(SMALL_IMAGE_CONFIG)
        p_aud = LocalTorchAudioProvider.from_random(SMALL_AUDIO_CONFIG)
        p_vid = LocalTorchVideoProvider.from_random(SMALL_VIDEO_CONFIG)
        p_omni = LocalTorchMultimodalProvider.from_random(SMALL_MULTIMODAL_CONFIG)
        assert p_img.num_parameters() > 0
        assert p_aud.num_parameters() > 0
        assert p_vid.num_parameters() > 0
        assert p_omni.num_parameters() > 0
