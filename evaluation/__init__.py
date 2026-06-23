"""Evaluation module for TorchaVerse.

This package provides comprehensive evaluation utilities for all
framework modalities:

* :mod:`torcha_verse.evaluation.text_eval` -- TextEvaluator for text
  generation quality (perplexity, BLEU, ROUGE, diversity, toxicity).
* :mod:`torcha_verse.evaluation.image_eval` -- ImageEvaluator for image
  generation quality (FID, Inception Score, CLIP Score, LPIPS).
* :mod:`torcha_verse.evaluation.benchmark_runner` -- BenchmarkRunner for
  standardised benchmark suites (OpenCompass / lm-evaluation-harness
  style).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
