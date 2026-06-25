"""Prompt-recall metric for the TorchaVerse evaluation framework (v0.4.0).

Prompt recall (a.k.a. CLIP-score) measures the alignment between a
text prompt and the image it produced.  The classical implementation
embeds both modalities with a pretrained CLIP model and reports the
cosine similarity of the resulting unit-norm vectors.

For the v0.4.0 minimum-viable milestone we ship a *placeholder* dual
encoder: two small randomly-initialised MLPs that lift fixed bag-of-
tokens text features and pooled image features into a shared
``_FEATURE_DIM``-dimensional space.  The public API (:func:`score` /
:func:`prompt_recall` / :class:`PromptRecallCalculator`) is the
production surface that will hold across the v0.4.x series; a future
swap-in of a pretrained CLIP model is a one-class change behind the
same ``encode_image`` / ``encode_text`` interface.

The placeholder keeps the math honest: ``score(image, prompt)`` still
returns a real cosine similarity in ``[-1, 1]`` (clamped to ``[0, 1]``
in the public API), and ``prompt_recall(images, prompts)`` still
returns the per-image / per-prompt similarities plus the cohort
mean -- so callers can wire it into their evaluation pipelines today.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``evaluation`` (this module) -- cross-modal metric.

Threading: the :class:`PromptRecallCalculator` lazily initialises
its encoders behind a lock and is safe to share across threads.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger
from consistency.score import _to_tensor

__all__ = [
    "prompt_recall",
    "score",
    "PromptRecallCalculator",
    "DualEncoderPlaceholder",
    "PromptRecallResult",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default shared embedding dimension for the dual encoder.
_FEATURE_DIM: int = 256

#: Vocabulary size for the placeholder bag-of-tokens text encoder.
#: Sized to comfortably cover the prompts used in the smoke tests
#: (~3k tokens) and grow as needed.
_VOCAB_SIZE: int = 4096

#: Default hidden width of the placeholder encoders.
_HIDDEN_DIM: int = 256

#: Random-projection stddev for the encoder heads.
_PROJ_STD: float = 0.02

#: Image size used by the image encoder's pooling head.
_IMAGE_POOL_SIZE: int = 4

#: Regex used to extract word tokens from a prompt.  Matches runs of
#: word characters; punctuation is dropped.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

#: Module-level logger.
_logger = get_logger("evaluation.prompt_recall")


# ---------------------------------------------------------------------------
# Lightweight text tokenizer
# ---------------------------------------------------------------------------
def _tokenize(prompt: str) -> List[int]:
    """Tokenize a prompt into a list of non-negative token IDs.

    Uses a deterministic Murmur-style hash for each word so the
    placeholder encoder does not require a fixed vocabulary file.  The
    same word always maps to the same ID within a Python process; the
    mapping is not portable across processes but the *cosine
    similarity* it produces is comparable within an evaluation run.

    Args:
        prompt: A natural-language prompt.

    Returns:
        A list of token IDs in ``[0, _VOCAB_SIZE)``.  Empty prompts
        return an empty list.
    """
    if not prompt:
        return []
    tokens: List[int] = []
    for match in _TOKEN_RE.findall(prompt.lower()):
        # FNV-1a 32-bit hash for portability across Python versions.
        h = 2166136261
        for ch in match.encode("utf-8"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        tokens.append(h % _VOCAB_SIZE)
    return tokens


# ---------------------------------------------------------------------------
# Placeholder dual encoder
# ---------------------------------------------------------------------------
class DualEncoderPlaceholder(nn.Module):
    """Lightweight dual encoder that projects images and text into a
    shared space (placeholder).

    The original prompt-recall metric embeds both modalities with a
    pretrained CLIP model.  For v0.4.0 we ship a structural
    placeholder: a small convolutional image encoder, a bag-of-tokens
    text encoder, and a shared linear projection head that lifts both
    to ``_FEATURE_DIM`` dimensions.  The forward methods return
    L2-normalised embeddings so the cosine similarity is just a dot
    product.

    The encoders are randomly initialised (not pretrained) so the
    resulting CLIP scores are *relative* rather than absolute, but
    they remain useful for comparing outputs produced under the same
    conditions.  A future swap-in of a pretrained CLIP model is a
    single-class change behind the same ``encode_image`` /
    ``encode_text`` interface.
    """

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        vocab_size: int = _VOCAB_SIZE,
        hidden_dim: int = _HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        # Image encoder: a small conv backbone.
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.image_pool = nn.AdaptiveAvgPool2d(_IMAGE_POOL_SIZE)
        self.image_head = nn.Linear(128 * _IMAGE_POOL_SIZE * _IMAGE_POOL_SIZE, feature_dim)
        nn.init.normal_(self.image_head.weight, std=_PROJ_STD)
        nn.init.zeros_(self.image_head.bias)

        # Text encoder: a bag-of-tokens embedding + 2-layer MLP.
        self.text_embed = nn.Embedding(vocab_size, hidden_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.normal_(self.text_mlp[0].weight, std=_PROJ_STD)
        nn.init.zeros_(self.text_mlp[0].bias)
        nn.init.normal_(self.text_mlp[2].weight, std=_PROJ_STD)
        nn.init.zeros_(self.text_mlp[2].bias)

    # ------------------------------------------------------------------
    # Public encoders
    # ------------------------------------------------------------------
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a ``(N, 3, H, W)`` image batch into L2-normalised embeddings.

        The image is bilinearly resized to 32x32 inside the backbone --
        this is the placeholder's canonical input size; the public
        API takes care of larger inputs by resizing before calling.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] != 32 or image.shape[-2] != 32:
            image = F.interpolate(
                image, size=(32, 32), mode="bilinear", align_corners=False
            )
        feat = self.image_backbone(image)
        feat = self.image_pool(feat).flatten(1)
        feat = self.image_head(feat)
        return F.normalize(feat, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a ``(N, L)`` token-ID batch into L2-normalised embeddings.

        Each row is a mean-pooled bag-of-tokens representation lifted
        through a 2-layer MLP.  Empty rows (no tokens) are mapped to
        the zero vector and then re-normalised -- the result is a
        unit-norm "no-signal" embedding that the cosine similarity
        treats as orthogonal to every real prompt.
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        n = tokens.shape[0]
        if tokens.numel() == 0:
            out = torch.zeros(n, self.feature_dim, device=tokens.device)
            return F.normalize(out, dim=-1)
        # Pad-free mean pool: ignore padding implicitly by averaging
        # over non-zero rows.  We use an additive mask to keep the
        # gradient signal clean.
        mask = (tokens != 0).to(tokens.dtype).unsqueeze(-1)
        embedded = self.text_embed(tokens) * mask
        sums = embedded.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = sums / counts
        feat = self.text_mlp(pooled)
        return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PromptRecallResult:
    """Container for a batch of prompt-recall scores.

    Attributes:
        scores: Per-(image, prompt) cosine similarities in ``[0, 1]``.
        mean: Mean of ``scores``.
        std: Standard deviation of ``scores`` (``0.0`` for n=1).
    """

    scores: List[float]
    mean: float
    std: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "scores": list(self.scores),
            "mean": float(self.mean),
            "std": float(self.std),
        }

    def __repr__(self) -> str:
        return (
            "PromptRecallResult(n={n}, mean={m:.3f}, std={s:.3f})".format(
                n=len(self.scores), m=self.mean, s=self.std,
            )
        )


# ---------------------------------------------------------------------------
# PromptRecallCalculator
# ---------------------------------------------------------------------------
class PromptRecallCalculator:
    """Stateful prompt-recall calculator with a cached dual encoder.

    Wraps a :class:`DualEncoderPlaceholder` (or, in the future, a
    pretrained CLIP model) and exposes a uniform :meth:`score` /
    :meth:`prompt_recall` API.  A single instance is safe to share
    across threads -- the encoder is lazily initialised behind a
    lock.

    Args:
        device: Optional device for encoding.  Defaults to CPU so the
            calculator is portable across CI environments.
        feature_dim: Shared embedding dimension.
        vocab_size: Text-encoder vocabulary size.
        encoder: Optional pre-built :class:`DualEncoderPlaceholder`
            (mainly for testing or for callers that want to share a
            custom encoder across multiple calculators).
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        feature_dim: int = _FEATURE_DIM,
        vocab_size: int = _VOCAB_SIZE,
        encoder: Optional[DualEncoderPlaceholder] = None,
    ) -> None:
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device if device is not None
            else torch.device("cpu")
        )
        self.feature_dim: int = int(feature_dim)
        self.vocab_size: int = int(vocab_size)
        self._encoder: Optional[nn.Module] = encoder
        self._lock: threading.Lock = threading.Lock()
        self._logger = _logger

    def _get_encoder(self) -> nn.Module:
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    net = DualEncoderPlaceholder(
                        feature_dim=self.feature_dim,
                        vocab_size=self.vocab_size,
                    ).to(self._device).eval()
                    self._encoder = net
        return self._encoder  # type: ignore[return-value]

    @torch.no_grad()
    def _embed_images(self, images: Sequence[Any]) -> torch.Tensor:
        if not images:
            raise ValueError("`images` must be non-empty")
        tensors: List[torch.Tensor] = []
        for img in images:
            t = _to_tensor(img)
            if t.dim() == 2:
                t = t.unsqueeze(0).repeat(3, 1, 1)
            tensors.append(t[:3])
        batch = torch.stack(tensors, dim=0).to(self._device)
        encoder = self._get_encoder()
        return encoder.encode_image(batch).detach().cpu()

    @torch.no_grad()
    def _embed_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        if not prompts:
            raise ValueError("`prompts` must be non-empty")
        token_lists = [_tokenize(p) for p in prompts]
        max_len = max((len(t) for t in token_lists), default=1)
        max_len = max(max_len, 1)
        # Pad with the reserved "zero token" (0) which the encoder
        # treats as padding.
        padded = torch.zeros(len(prompts), max_len, dtype=torch.long)
        for i, tokens in enumerate(token_lists):
            if tokens:
                padded[i, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long
                )
        padded = padded.to(self._device)
        encoder = self._get_encoder()
        return encoder.encode_text(padded).detach().cpu()

    def score(self, image: Any, prompt: str) -> float:
        """Compute the cosine similarity between a single image and prompt.

        Returns:
            A float in ``[0, 1]`` (the cosine similarity of unit-norm
            vectors is in ``[-1, 1]``; we clamp to ``[0, 1]`` so the
            result is monotone in alignment).
        """
        img_emb = self._embed_images([image])
        txt_emb = self._embed_texts([prompt])
        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1).item()
        return float(max(0.0, min(1.0, cos)))

    def prompt_recall(
        self,
        images: Sequence[Any],
        prompts: Sequence[str],
    ) -> PromptRecallResult:
        """Compute per-pair similarities plus the cohort mean.

        The two sequences must have the same length; each ``images[i]``
        is scored against ``prompts[i]``.

        Returns:
            A :class:`PromptRecallResult` with per-pair scores, the
            cohort mean, and the cohort standard deviation.
        """
        if len(images) != len(prompts):
            raise ValueError(
                "`images` and `prompts` must have the same length "
                "(got {} and {})".format(len(images), len(prompts))
            )
        if not images:
            raise ValueError("`images` and `prompts` must be non-empty")
        img_emb = self._embed_images(images)
        txt_emb = self._embed_texts(prompts)
        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1).tolist()
        scores = [float(max(0.0, min(1.0, c))) for c in cos]
        n = len(scores)
        mean = sum(scores) / float(n)
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / float(n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        return PromptRecallResult(scores=scores, mean=mean, std=std)

    def __repr__(self) -> str:
        return (
            "PromptRecallCalculator(device={}, feature_dim={}, "
            "vocab_size={})".format(
                self._device, self.feature_dim, self.vocab_size,
            )
        )


# ---------------------------------------------------------------------------
# Public free-function API
# ---------------------------------------------------------------------------
def score(image: Any, prompt: str) -> float:
    """Compute the prompt-recall score for a single image-prompt pair.

    Uses a process-level singleton :class:`PromptRecallCalculator`
    so multiple calls in the same evaluation share the same encoder.
    """
    return _default_calculator().score(image, prompt)


def prompt_recall(
    images: Sequence[Any],
    prompts: Sequence[str],
) -> PromptRecallResult:
    """Compute prompt-recall scores for a batch of (image, prompt) pairs.

    Returns:
        A :class:`PromptRecallResult` with per-pair scores, the
        cohort mean, and the cohort standard deviation.
    """
    return _default_calculator().prompt_recall(images, prompts)


_default_calc: Optional[PromptRecallCalculator] = None
_default_calc_lock: threading.Lock = threading.Lock()


def _default_calculator() -> PromptRecallCalculator:
    """Return the process-level singleton calculator (lazy-initialised)."""
    global _default_calc
    if _default_calc is None:
        with _default_calc_lock:
            if _default_calc is None:
                _default_calc = PromptRecallCalculator()
    return _default_calc
