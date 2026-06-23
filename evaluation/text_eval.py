"""Text generation quality evaluation for TorchaVerse.

This module provides :class:`TextEvaluator`, a comprehensive evaluator
for text generation models.  It implements standard metrics used in NLP
evaluation:

* **Perplexity** -- measures how well a model predicts a dataset.
* **Accuracy** -- exact-match and normalised-match accuracy against
  reference answers.
* **BLEU** -- n-gram precision-based translation/generation score.
* **ROUGE** -- recall-oriented n-gram overlap (ROUGE-1, ROUGE-2,
  ROUGE-L).
* **Diversity** -- distinct-n and entropy-based lexical diversity.
* **Toxicity** -- a lightweight keyword-based toxicity detector.

All metrics are implemented in pure Python/NumPy so that the evaluation
pipeline has no hard external dependencies beyond PyTorch (used for
perplexity computation).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from infrastructure.logger import get_logger

__all__ = ["TextEvaluator"]

logger = get_logger("TextEvaluator")


# ===========================================================================
# Utility functions
# ===========================================================================
def _tokenize(text: str) -> List[str]:
    """Tokenise ``text`` into lowercase word tokens.

    Args:
        text: Input text.

    Returns:
        A list of lowercase tokens.
    """
    # Simple whitespace + punctuation tokenisation.
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    """Extract n-grams from a token list.

    Args:
        tokens: Sequence of tokens.
        n: N-gram order.

    Returns:
        A list of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute the length of the longest common subsequence.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        The LCS length.
    """
    m, n = len(a), len(b)
    # Use a rolling array to save memory.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _normalise_text(text: str) -> str:
    """Normalise text for fair comparison.

    Lowercases, strips whitespace, and removes punctuation.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ===========================================================================
# Toxicity keyword list (simplified)
# ===========================================================================
_TOXIC_KEYWORDS: List[str] = [
    "hate", "kill", "stupid", "idiot", "damn", "shut up",
    "racist", "nazi", "terrorist", "violent", "attack",
    "destroy", "threat", "enemy", "war", "weapon",
]


# ===========================================================================
# TextEvaluator
# ===========================================================================
class TextEvaluator:
    """Evaluate text generation quality across multiple metrics.

    This evaluator provides standard NLP metrics without requiring heavy
    external libraries.  All methods accept plain Python strings or lists
    thereof, making the evaluator easy to integrate into any pipeline.

    Example::

        evaluator = TextEvaluator()
        bleu = evaluator.evaluate_bleu(predictions, references)
        results = evaluator.evaluate_all(predictions, references)
    """

    def __init__(
        self,
        max_order: int = 4,
        smooth: bool = True,
    ) -> None:
        """Initialise the evaluator.

        Args:
            max_order: Maximum n-gram order for BLEU computation.
            smooth: Whether to apply smoothing to BLEU (avoids zero
                scores when a precision is zero).
        """
        self.max_order: int = max_order
        self.smooth: bool = smooth
        self._logger = logger

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    def evaluate_perplexity(
        self,
        model: Any,
        dataset: Any,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> float:
        """Evaluate the perplexity of a model on a dataset.

        Perplexity is defined as ``exp(average negative log-likelihood)``.
        The model is expected to have a ``forward`` method that returns
        logits of shape ``(batch, seq_len, vocab_size)`` and a
        ``tokenize`` method (or a ``tokenizer`` with ``encode``).

        Args:
            model: A model with ``forward`` and ``tokenize``/``tokenizer``.
            dataset: An iterable of text strings or a list of token-id
                tensors.
            batch_size: Batch size for evaluation.
            max_length: Maximum sequence length.

        Returns:
            The perplexity score (lower is better).  Returns ``float('inf')``
            if the dataset is empty.
        """
        self._logger.info("Computing perplexity...")

        total_loss = 0.0
        total_tokens = 0

        # Resolve the tokenisation function.
        tokenize_fn = getattr(model, "tokenize", None)
        if tokenize_fn is None:
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None:
                tokenize_fn = lambda text: tokenizer.encode(text, return_tensors=True)  # type: ignore[union-attr]

        if tokenize_fn is None:
            self._logger.warning(
                "No tokenize/tokenizer method found on model; "
                "cannot compute perplexity."
            )
            return float("inf")

        device = getattr(model, "_device", None) or torch.device("cpu")

        model.eval()
        with torch.no_grad():
            for item in dataset:
                # Tokenise.
                if isinstance(item, str):
                    input_ids = tokenize_fn(item)
                elif isinstance(item, torch.Tensor):
                    input_ids = item
                elif isinstance(item, (list, tuple)):
                    input_ids = torch.tensor(item, dtype=torch.long)
                else:
                    continue

                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long)

                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(device)

                if input_ids.shape[1] < 2:
                    continue

                # Truncate.
                input_ids = input_ids[:, :max_length]

                # Forward pass.
                try:
                    logits = model.forward(input_ids)
                except Exception:
                    self._logger.warning("Forward pass failed; skipping item.")
                    continue

                if not isinstance(logits, torch.Tensor):
                    continue

                # Shift for next-token prediction.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

                num_tokens = shift_labels.numel()
                total_loss += loss.item()
                total_tokens += num_tokens

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_loss / total_tokens
        perplexity = math.exp(avg_nll)
        self._logger.info("Perplexity: %.4f", perplexity)
        return perplexity

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    def evaluate_accuracy(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> Dict[str, float]:
        """Evaluate accuracy of predictions against references.

        Computes exact match (EM) and normalised match (after
        lowercasing and removing punctuation).

        Args:
            predictions: List of predicted strings.
            references: List of reference strings.

        Returns:
            A dictionary with ``exact_match`` and ``normalized_match``
            scores in ``[0, 1]``.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(references)} references."
            )

        if not predictions:
            return {"exact_match": 0.0, "normalized_match": 0.0}

        exact = 0
        normalised = 0

        for pred, ref in zip(predictions, references):
            if pred.strip() == ref.strip():
                exact += 1
            if _normalise_text(pred) == _normalise_text(ref):
                normalised += 1

        n = len(predictions)
        result = {
            "exact_match": exact / n,
            "normalized_match": normalised / n,
        }
        self._logger.info("Accuracy: %s", result)
        return result

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------
    def evaluate_bleu(
        self,
        predictions: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
        max_order: Optional[int] = None,
    ) -> float:
        """Evaluate the BLEU score.

        Implements the standard corpus-level BLEU with brevity penalty.

        Args:
            predictions: List of predicted strings.
            references: List of reference strings (or lists of reference
                strings for multi-reference BLEU).
            max_order: Maximum n-gram order.  Defaults to
                ``self.max_order``.

        Returns:
            The BLEU score in ``[0, 1]``.
        """
        order = max_order or self.max_order
        if len(predictions) != len(references):
            raise ValueError("Length mismatch between predictions and references.")

        matches_by_order = [0] * order
        possible_by_order = [0] * order
        reference_length = 0
        translation_length = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = _tokenize(pred)
            # Normalise references to a list of token lists.
            if isinstance(ref, str):
                ref_list = [_tokenize(ref)]
            else:
                ref_list = [_tokenize(r) if isinstance(r, str) else list(r) for r in ref]

            reference_length += min(len(r) for r in ref_list)
            translation_length += len(pred_tokens)

            for n in range(1, order + 1):
                pred_ngrams = Counter(_ngrams(pred_tokens, n))
                # Union of reference n-grams (clipped).
                max_ref_ngrams: Counter = Counter()
                for r in ref_list:
                    ref_ngrams = Counter(_ngrams(r, n))
                    for gram, count in ref_ngrams.items():
                        max_ref_ngrams[gram] = max(max_ref_ngrams[gram], count)

                matches = sum(
                    min(count, max_ref_ngrams[gram])
                    for gram, count in pred_ngrams.items()
                )
                possible = max(len(pred_tokens) - n + 1, 0)

                matches_by_order[n - 1] += matches
                possible_by_order[n - 1] += possible

        # Compute precisions with optional smoothing.
        precisions: List[float] = []
        for i in range(order):
            if self.smooth and possible_by_order[i] == 0:
                precisions.append(0.0)
            elif possible_by_order[i] == 0:
                precisions.append(0.0)
            else:
                if self.smooth:
                    # Add-one smoothing.
                    precisions.append(
                        (matches_by_order[i] + 1) / (possible_by_order[i] + 1)
                    )
                else:
                    precisions.append(matches_by_order[i] / possible_by_order[i])

        # Geometric mean of precisions.
        if min(precisions) > 0:
            log_precision = sum(math.log(p) for p in precisions) / order
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0.0

        # Brevity penalty.
        if translation_length > reference_length:
            bp = 1.0
        elif translation_length == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - reference_length / translation_length)

        bleu = bp * geo_mean
        self._logger.info("BLEU-%d: %.4f", order, bleu)
        return bleu

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------
    def evaluate_rouge(
        self,
        predictions: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
    ) -> Dict[str, float]:
        """Evaluate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

        Args:
            predictions: List of predicted strings.
            references: List of reference strings (or lists thereof).

        Returns:
            A dictionary with ``rouge_1``, ``rouge_2``, and ``rouge_l``
            F-measure scores in ``[0, 1]``.
        """
        if len(predictions) != len(references):
            raise ValueError("Length mismatch between predictions and references.")

        rouge_1_scores: List[float] = []
        rouge_2_scores: List[float] = []
        rouge_l_scores: List[float] = []

        for pred, ref in zip(predictions, references):
            pred_tokens = _tokenize(pred)
            if isinstance(ref, str):
                ref_tokens = _tokenize(ref)
            else:
                # Use the first reference.
                ref_tokens = _tokenize(ref[0]) if ref else []

            # ROUGE-1 (unigram).
            r1 = self._rouge_n(pred_tokens, ref_tokens, 1)
            rouge_1_scores.append(r1)

            # ROUGE-2 (bigram).
            r2 = self._rouge_n(pred_tokens, ref_tokens, 2)
            rouge_2_scores.append(r2)

            # ROUGE-L (LCS-based).
            rl = self._rouge_l(pred_tokens, ref_tokens)
            rouge_l_scores.append(rl)

        n = len(predictions) if predictions else 1
        result = {
            "rouge_1": sum(rouge_1_scores) / n,
            "rouge_2": sum(rouge_2_scores) / n,
            "rouge_l": sum(rouge_l_scores) / n,
        }
        self._logger.info("ROUGE: %s", result)
        return result

    @staticmethod
    def _rouge_n(pred_tokens: Sequence[str], ref_tokens: Sequence[str], n: int) -> float:
        """Compute ROUGE-N F-measure for a single prediction/reference pair."""
        pred_ngrams = Counter(_ngrams(pred_tokens, n))
        ref_ngrams = Counter(_ngrams(ref_tokens, n))

        if not ref_ngrams:
            return 0.0

        overlap = sum(
            min(count, ref_ngrams[gram])
            for gram, count in pred_ngrams.items()
        )

        precision = overlap / max(sum(pred_ngrams.values()), 1)
        recall = overlap / sum(ref_ngrams.values())

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _rouge_l(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
        """Compute ROUGE-L F-measure using LCS."""
        if not pred_tokens or not ref_tokens:
            return 0.0

        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------
    def evaluate_diversity(self, texts: Sequence[str]) -> Dict[str, float]:
        """Evaluate lexical diversity of generated texts.

        Computes:

        * ``distinct_1`` -- ratio of unique unigrams to total unigrams.
        * ``distinct_2`` -- ratio of unique bigrams to total bigrams.
        * ``entropy`` -- Shannon entropy of the unigram distribution.

        Args:
            texts: List of generated text strings.

        Returns:
            A dictionary with diversity metrics.
        """
        all_tokens: List[str] = []
        for text in texts:
            all_tokens.extend(_tokenize(text))

        if not all_tokens:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "entropy": 0.0}

        # Distinct-1.
        unigrams = set(all_tokens)
        distinct_1 = len(unigrams) / len(all_tokens)

        # Distinct-2.
        bigrams = _ngrams(all_tokens, 2)
        unique_bigrams = set(bigrams)
        distinct_2 = len(unique_bigrams) / max(len(bigrams), 1)

        # Shannon entropy of unigram distribution.
        counter = Counter(all_tokens)
        total = len(all_tokens)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        result = {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "entropy": entropy,
        }
        self._logger.info("Diversity: %s", result)
        return result

    # ------------------------------------------------------------------
    # Toxicity
    # ------------------------------------------------------------------
    def evaluate_toxicity(self, texts: Sequence[str]) -> float:
        """Evaluate the toxicity ratio of generated texts.

        Uses a simplified keyword-based approach: a text is flagged as
        toxic if it contains any keyword from a curated list.  The score
        is the fraction of toxic texts.

        Args:
            texts: List of generated text strings.

        Returns:
            The toxicity ratio in ``[0, 1]`` (lower is better).
        """
        if not texts:
            return 0.0

        toxic_count = 0
        for text in texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in _TOXIC_KEYWORDS):
                toxic_count += 1

        ratio = toxic_count / len(texts)
        self._logger.info("Toxicity ratio: %.4f", ratio)
        return ratio

    # ------------------------------------------------------------------
    # Comprehensive evaluation
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        predictions: Sequence[str],
        references: Optional[Sequence[Union[str, Sequence[str]]]] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run all applicable metrics in one call.

        Args:
            predictions: List of predicted strings.
            references: Optional list of reference strings.  When
                provided, accuracy, BLEU, and ROUGE are computed.
            texts: Optional list of texts for diversity and toxicity.
                Defaults to ``predictions`` when not provided.

        Returns:
            A dictionary containing all computed metrics.
        """
        results: Dict[str, Any] = {}

        eval_texts = texts if texts is not None else predictions

        # Diversity and toxicity always available.
        results["diversity"] = self.evaluate_diversity(eval_texts)
        results["toxicity"] = self.evaluate_toxicity(eval_texts)

        # Reference-based metrics.
        if references is not None and len(references) == len(predictions):
            results["accuracy"] = self.evaluate_accuracy(predictions, references)
            results["bleu"] = self.evaluate_bleu(predictions, references)
            results["rouge"] = self.evaluate_rouge(predictions, references)

        self._logger.info("Comprehensive evaluation complete: %d metrics", len(results))
        return results

    # ------------------------------------------------------------------
    # Batch streaming evaluation
    # ------------------------------------------------------------------
    def evaluate_streaming(
        self,
        prediction_stream: Iterator[str],
        reference_stream: Optional[Iterator[str]] = None,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Evaluate predictions in streaming batches.

        Useful for evaluating very large datasets without holding all
        predictions in memory.

        Args:
            prediction_stream: An iterator yielding prediction strings.
            reference_stream: An optional iterator yielding reference
                strings.
            batch_size: Number of items per batch.

        Yields:
            A dictionary of metrics for each batch.
        """
        batch_preds: List[str] = []
        batch_refs: List[str] = []

        for i, pred in enumerate(prediction_stream):
            batch_preds.append(pred)
            if reference_stream is not None:
                batch_refs.append(next(reference_stream, ""))

            if len(batch_preds) >= batch_size:
                refs = batch_refs if reference_stream is not None else None
                yield self.evaluate_all(batch_preds, refs)
                batch_preds = []
                batch_refs = []

        # Flush remaining.
        if batch_preds:
            refs = batch_refs if reference_stream is not None else None
            yield self.evaluate_all(batch_preds, refs)

    def __repr__(self) -> str:
        return (
            f"TextEvaluator(max_order={self.max_order}, smooth={self.smooth})"
        )
