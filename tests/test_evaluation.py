"""Tests for the TorchaVerse evaluation framework (v0.4.0).

These tests are deliberately self-contained -- they only require
``torch`` (and ``Pillow`` for the directory-loader test), so they
run in any CI environment regardless of which optional dependencies
are installed.

Coverage
--------

* :mod:`evaluation.metrics` -- PSNR / SSIM / LPIPS placeholder.
* :mod:`evaluation.fid` -- Frechet distance math, Inception
  placeholder, ``FidCalculator``, ``image_fid`` entry point.
* :mod:`evaluation.prompt_recall` -- tokenizer, dual encoder,
  ``PromptRecallCalculator``, ``score`` / ``prompt_recall``.
* :mod:`evaluation.runner` -- ``EvaluationRunner`` facade and
  ``load_image_dir``.
* :mod:`evaluation` -- the top-level re-exports.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from evaluation import (
    DualEncoderPlaceholder,
    EvaluationReport,
    EvaluationRunner,
    FidCalculator,
    InceptionPlaceholder,
    LpipPlaceholder,
    PromptRecallCalculator,
    PromptRecallResult,
    compute_statistics,
    frechet_distance,
    image_fid,
    load_image_dir,
    lpips,
    prompt_recall,
    psnr,
    score,
    ssim,
)

# Mark every test in this module with the `eval` marker so that
# `pytest -m eval` runs the full evaluation suite and
# `pytest -m "not eval"` skips it.
pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng() -> torch.Generator:
    """Deterministic RNG so tests are reproducible across machines."""
    g = torch.Generator()
    g.manual_seed(20260625)
    return g


@pytest.fixture
def images(rng) -> List[torch.Tensor]:
    """A small batch of deterministic ``(3, 32, 32)`` images in ``[0, 1]``."""
    return [torch.rand(3, 32, 32, generator=rng) for _ in range(4)]


@pytest.fixture
def similar_images(rng) -> List[torch.Tensor]:
    """A batch of images with controlled pairwise similarity.

    The first two images are identical (modulo a tiny noise); the rest
    are independent.  Useful for asserting that FID / SSIM / PSNR
    correctly report "small distance" on near-duplicate inputs and
    "larger distance" on independent inputs.
    """
    base = torch.rand(3, 32, 32, generator=rng)
    g = torch.Generator()
    g.manual_seed(42)
    near = base + 0.01 * torch.randn(3, 32, 32, generator=g)
    near = near.clamp(0.0, 1.0)
    far1 = torch.rand(3, 32, 32, generator=rng)
    far2 = torch.rand(3, 32, 32, generator=rng)
    return [base, near, far1, far2]


@pytest.fixture
def prompts() -> List[str]:
    return [
        "a cat sitting on a chair",
        "a dog running in the park",
        "a small bird on a branch",
        "a horse in a green field",
    ]


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------
class TestPsnr:
    def test_identical_returns_inf(self, images) -> None:
        """PSNR of an image with itself is the cap value, not NaN/inf-from-log."""
        a = images[0]
        # The cap avoids log10(0); just assert it's "very large".
        assert psnr(a, a) >= 50.0

    def test_higher_when_closer(self, similar_images) -> None:
        """PSNR of near-duplicate images is higher than PSNR of independent ones."""
        base, near, far1, _ = similar_images
        assert psnr(base, near) > psnr(base, far1)

    def test_monotone_in_mse(self, rng) -> None:
        """PSNR strictly decreases as we add more noise."""
        base = torch.rand(3, 32, 32, generator=rng)
        psnrs = []
        for std in (0.01, 0.05, 0.10, 0.20):
            noisy = (base + std * torch.randn_like(base)).clamp(0.0, 1.0)
            psnrs.append(psnr(base, noisy))
        for i in range(len(psnrs) - 1):
            assert psnrs[i] > psnrs[i + 1], psnrs

    def test_max_value_uint8(self, rng) -> None:
        """With max_value=255 the same MSE produces a higher PSNR."""
        a = torch.rand(3, 8, 8, generator=rng)
        b = (a + 0.05 * torch.randn_like(a)).clamp(0.0, 1.0)
        p_low = psnr(a, b, max_value=1.0)
        p_high = psnr(a, b, max_value=255.0)
        assert p_high > p_low


class TestSsim:
    def test_identical_returns_one(self, images) -> None:
        assert ssim(images[0], images[0]) == pytest.approx(1.0, abs=1e-3)

    def test_independent_lower(self, similar_images) -> None:
        base, near, far1, _ = similar_images
        assert ssim(base, near) > ssim(base, far1)

    def test_bounded_zero_one(self, images) -> None:
        a, b = images[0], images[1]
        value = ssim(a, b)
        assert 0.0 <= value <= 1.0

    def test_window_size_kwarg(self, images) -> None:
        a, b = images[0], images[1]
        # Custom window size should not blow up.
        v = ssim(a, b, window_size=7)
        assert 0.0 <= v <= 1.0


class TestLpips:
    def test_returns_bounded_scalar(self, images) -> None:
        a, b = images[0], images[1]
        v = lpips(a, b)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_identical_near_zero(self, images) -> None:
        a = images[0]
        # Identical inputs produce identical features; cos_sim == 1,
        # so LPIPS distance should be ~0.
        v = lpips(a, a)
        assert v < 1e-4

    def test_placeholder_module_lazy_initialised(self) -> None:
        """Repeated calls share the same module instance (caching)."""
        from evaluation.metrics import _get_lpips_placeholder
        a = _get_lpips_placeholder()
        b = _get_lpips_placeholder()
        assert a is b

    def test_placeholder_forward_shape(self) -> None:
        net = LpipPlaceholder()
        x = torch.rand(2, 3, 32, 32)
        out = net(x)
        assert out.shape == (2, net.feature_dim)
        # L2-normalised rows.
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# fid.py
# ---------------------------------------------------------------------------
class TestFrechetDistance:
    def test_zero_for_identical_gaussians(self) -> None:
        mu = torch.tensor([1.0, 2.0, 3.0])
        sigma = torch.eye(3)
        assert frechet_distance(mu, sigma, mu, sigma) < 1e-6

    def test_positive_for_different_means(self) -> None:
        mu1 = torch.tensor([0.0, 0.0])
        mu2 = torch.tensor([1.0, 0.0])
        sigma = torch.eye(2)
        assert frechet_distance(mu1, sigma, mu2, sigma) == pytest.approx(1.0, abs=1e-3)

    def test_non_negative(self, rng) -> None:
        mu1 = torch.randn(8, generator=rng)
        mu2 = torch.randn(8, generator=rng)
        a = torch.randn(8, 8, generator=rng)
        sigma1 = a @ a.transpose(0, 1) + torch.eye(8) * 0.1
        b = torch.randn(8, 8, generator=rng)
        sigma2 = b @ b.transpose(0, 1) + torch.eye(8) * 0.1
        value = frechet_distance(mu1, sigma1, mu2, sigma2)
        assert value >= 0.0


class TestComputeStatistics:
    def test_shape_and_length(self, images) -> None:
        mu, sigma = compute_statistics(images)
        assert mu.ndim == 1
        assert sigma.ndim == 2
        assert mu.shape[0] == sigma.shape[0] == sigma.shape[1]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_statistics([])


class TestInceptionPlaceholder:
    def test_forward_shape(self) -> None:
        net = InceptionPlaceholder(feature_dim=64, image_size=32)
        x = torch.rand(2, 3, 32, 32)
        out = net(x)
        assert out.shape == (2, 64)

    def test_resizes_arbitrary_input(self) -> None:
        """Larger / non-square inputs are resized to ``image_size``."""
        net = InceptionPlaceholder(feature_dim=64, image_size=32)
        x = torch.rand(1, 3, 100, 200)
        out = net(x)
        assert out.shape == (1, 64)


class TestFidCalculator:
    def test_smoke(self, images) -> None:
        calc = FidCalculator(image_size=32, feature_dim=64)
        feats = calc.features(images)
        assert feats.shape == (len(images), 64)

    def test_fid_zero_for_same_set(self, images) -> None:
        calc = FidCalculator(image_size=32, feature_dim=64)
        assert calc.fid(images, images) < 1e-4

    def test_fid_non_negative_for_different_sets(self, images, rng) -> None:
        calc = FidCalculator(image_size=32, feature_dim=64)
        other = [torch.rand(3, 32, 32, generator=rng) for _ in range(4)]
        v = calc.fid(images, other)
        assert v >= 0.0

    def test_empty_set_raises(self, images) -> None:
        calc = FidCalculator(image_size=32, feature_dim=64)
        with pytest.raises(ValueError):
            calc.fid([], images)
        with pytest.raises(ValueError):
            calc.fid(images, [])

    def test_features_empty_raises(self) -> None:
        calc = FidCalculator(image_size=32, feature_dim=64)
        with pytest.raises(ValueError):
            calc.features([])


class TestImageFid:
    def test_zero_for_identical_sets(self, images) -> None:
        assert image_fid(images, images) < 1e-3

    def test_non_negative(self, images, rng) -> None:
        other = [torch.rand(3, 32, 32, generator=rng) for _ in range(4)]
        assert image_fid(images, other) >= 0.0

    def test_empty_raises(self, images) -> None:
        with pytest.raises(ValueError):
            image_fid([], images)
        with pytest.raises(ValueError):
            image_fid(images, [])


# ---------------------------------------------------------------------------
# prompt_recall.py
# ---------------------------------------------------------------------------
class TestTokenizer:
    def test_basic(self) -> None:
        from evaluation.prompt_recall import _tokenize
        tokens = _tokenize("a cat and a dog")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < 4096 for t in tokens)

    def test_deterministic(self) -> None:
        from evaluation.prompt_recall import _tokenize
        assert _tokenize("hello world") == _tokenize("hello world")

    def test_empty_prompt(self) -> None:
        from evaluation.prompt_recall import _tokenize
        assert _tokenize("") == []


class TestDualEncoder:
    def test_encode_image_shape(self) -> None:
        net = DualEncoderPlaceholder(feature_dim=64)
        x = torch.rand(2, 3, 32, 32)
        out = net.encode_image(x)
        assert out.shape == (2, 64)
        assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)

    def test_encode_text_shape(self) -> None:
        net = DualEncoderPlaceholder(feature_dim=64)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
        out = net.encode_text(tokens)
        assert out.shape == (2, 64)
        assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)

    def test_encode_text_empty(self) -> None:
        net = DualEncoderPlaceholder(feature_dim=64)
        # All-zero (no real tokens) -> "no-signal" unit embedding.
        tokens = torch.zeros(1, 4, dtype=torch.long)
        out = net.encode_text(tokens)
        assert out.shape == (1, 64)


class TestPromptRecallCalculator:
    def test_score_bounded(self, images, prompts) -> None:
        calc = PromptRecallCalculator(feature_dim=64)
        v = calc.score(images[0], prompts[0])
        assert 0.0 <= v <= 1.0

    def test_prompt_recall_returns_result(self, images, prompts) -> None:
        calc = PromptRecallCalculator(feature_dim=64)
        result = calc.prompt_recall(images, prompts)
        assert isinstance(result, PromptRecallResult)
        assert len(result.scores) == len(images)
        assert 0.0 <= result.mean <= 1.0
        assert result.std >= 0.0

    def test_length_mismatch_raises(self, images) -> None:
        calc = PromptRecallCalculator(feature_dim=64)
        with pytest.raises(ValueError):
            calc.prompt_recall(images, ["only one"])

    def test_empty_raises(self) -> None:
        calc = PromptRecallCalculator(feature_dim=64)
        with pytest.raises(ValueError):
            calc.prompt_recall([], [])


class TestModuleLevelScoreAndPromptRecall:
    def test_score_bounded(self, images) -> None:
        v = score(images[0], "a cat")
        assert 0.0 <= v <= 1.0

    def test_prompt_recall_bounded(self, images, prompts) -> None:
        result = prompt_recall(images, prompts)
        assert isinstance(result, PromptRecallResult)
        assert 0.0 <= result.mean <= 1.0


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------
class TestLoadImageDir:
    def test_loads_pngs(self, tmp_path, rng) -> None:
        """load_image_dir should decode PNGs into float tensors."""
        from PIL import Image
        import numpy as np
        for i in range(3):
            arr = (torch.rand(16, 16, 3, generator=rng) * 255).to(torch.uint8).numpy()
            Image.fromarray(arr).save(tmp_path / "img_{}.png".format(i))
        tensors = load_image_dir(tmp_path)
        assert len(tensors) == 3
        for t in tensors:
            assert t.shape == (3, 16, 16)
            assert 0.0 <= t.min().item() <= t.max().item() <= 1.0

    def test_skips_hidden_and_non_image(self, tmp_path) -> None:
        (tmp_path / "real.png").write_bytes(b"")
        (tmp_path / ".hidden.png").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("ignore me")
        # Only the visible PNG is *attempted* -- but it's empty so we
        # fall back to a deterministic tensor; the test verifies the
        # *count* is right.
        tensors = load_image_dir(tmp_path)
        assert len(tensors) == 1

    def test_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image_dir("/path/that/does/not/exist/xyz")

    def test_not_a_directory_raises(self, tmp_path) -> None:
        f = tmp_path / "a_file.png"
        f.write_bytes(b"")
        with pytest.raises(NotADirectoryError):
            load_image_dir(f)


class TestEvaluationRunner:
    def test_run_fid_only(self, images, rng) -> None:
        runner = EvaluationRunner(compute_per_image=False)
        other = [torch.rand(3, 32, 32, generator=rng) for _ in range(4)]
        report = runner.run(real=images, generated=other)
        assert isinstance(report, EvaluationReport)
        assert report.fid >= 0.0
        assert report.n_real == len(images)
        assert report.n_generated == len(other)
        assert report.prompt_recall is None
        assert report.per_image is None

    def test_run_with_prompts(self, images, prompts, rng) -> None:
        runner = EvaluationRunner(compute_per_image=False)
        # Generated set has the same length as prompts.
        gen = [torch.rand(3, 32, 32, generator=rng) for _ in range(len(prompts))]
        report = runner.run(real=images, generated=gen, prompts=prompts)
        assert report.prompt_recall is not None
        assert 0.0 <= report.prompt_recall["mean"] <= 1.0

    def test_run_with_per_image(self, images) -> None:
        runner = EvaluationRunner(compute_per_image=True)
        report = runner.run(real=images, generated=images)
        assert report.per_image is not None
        assert "psnr" in report.per_image
        assert "ssim" in report.per_image
        assert "lpips" in report.per_image
        assert len(report.per_image["psnr"]) == len(images)

    def test_run_prompts_length_must_match_generated(self, images) -> None:
        runner = EvaluationRunner(compute_per_image=False)
        # images has 4 entries; pass only 3 prompts -> mismatch.
        with pytest.raises(ValueError):
            runner.run(real=images, generated=images, prompts=["a", "b", "c"])

    def test_run_empty_raises(self) -> None:
        runner = EvaluationRunner(compute_per_image=False)
        with pytest.raises(ValueError):
            runner.run(real=[], generated=[torch.rand(3, 4, 4)])

    def test_report_to_dict_round_trip(self, images, rng) -> None:
        runner = EvaluationRunner(compute_per_image=False)
        other = [torch.rand(3, 32, 32, generator=rng) for _ in range(4)]
        report = runner.run(real=images, generated=other)
        d = report.to_dict()
        assert "fid" in d
        assert d["n_real"] == len(images)
        assert d["n_generated"] == len(other)


class TestFromDirs:
    def test_end_to_end(self, tmp_path, rng) -> None:
        """End-to-end: write two image directories, run the runner."""
        from PIL import Image
        import numpy as np
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        for i in range(3):
            arr = (torch.rand(16, 16, 3, generator=rng) * 255).to(torch.uint8).numpy()
            Image.fromarray(arr).save(real_dir / "r_{}.png".format(i))
            Image.fromarray(arr).save(gen_dir / "g_{}.png".format(i))
        report = EvaluationRunner.from_dirs(real_dir, gen_dir)
        # Identical sets -> very small FID.
        assert report.fid < 1.0
        assert report.n_real == 3
        assert report.n_generated == 3


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------
class TestPublicAPI:
    def test_imports(self) -> None:
        """All public names are importable from the package root."""
        # Re-import everything to catch stale / shadowed names.
        from evaluation import (  # noqa: F401
            DualEncoderPlaceholder,
            EvaluationReport,
            EvaluationRunner,
            FidCalculator,
            InceptionPlaceholder,
            LpipPlaceholder,
            PromptRecallCalculator,
            PromptRecallResult,
            compute_statistics,
            frechet_distance,
            image_fid,
            load_image_dir,
            lpips,
            prompt_recall,
            psnr,
            score,
            ssim,
        )

    def test_version_attribute(self) -> None:
        from evaluation import __version__
        assert __version__ == "0.4.0"
