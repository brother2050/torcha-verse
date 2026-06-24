"""Synthetic data generation for TorchaVerse.

This module provides :class:`SyntheticDataGenerator`, which leverages a
:class:`TextEngine` to produce training data programmatically.  It can
generate:

* **Instructions** -- diverse task prompts for a given topic.
* **Responses** -- model-generated answers to a set of instructions.
* **Preference pairs** -- chosen/rejected response pairs for DPO.
* **Conversations** -- multi-turn dialogues.

A simple heuristic quality filter (:meth:`filter_quality`) scores each
sample and removes low-quality entries.  Data can be persisted to and
loaded from JSONL files.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# engines removed - use Pipeline
from infrastructure.logger import get_logger

__all__ = ["SyntheticDataGenerator", "SyntheticDataConfig"]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class SyntheticDataConfig:
    """Configuration for :class:`SyntheticDataGenerator`.

    Args:
        max_tokens: Maximum tokens for each generation.
        temperature: Sampling temperature (higher = more diverse).
        top_p: Nucleus sampling threshold.
        seed: Random seed.
        diversity_threshold: Minimum edit-distance ratio between two
            generated instructions for them to be considered distinct.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: int = 42,
        diversity_threshold: float = 0.3,
    ) -> None:
        self.max_tokens: int = max(16, int(max_tokens))
        self.temperature: float = float(temperature)
        self.top_p: float = float(top_p)
        self.seed: int = int(seed)
        self.diversity_threshold: float = float(diversity_threshold)


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    """Generate synthetic training data using a :class:`TextEngine`.

    The generator wraps a text engine and provides high-level methods
    for producing instructions, responses, preference pairs, and
    multi-turn conversations.  All methods return plain Python
    dictionaries that can be directly consumed by the dataset classes
    or saved to JSONL.

    Args:
        text_engine: A configured :class:`TextEngine` used for generation.
        config: Optional :class:`SyntheticDataConfig`.  When ``None``
            sensible defaults are used.
    """

    def __init__(
        self,
        text_engine: TextEngine,
        config: Optional[SyntheticDataConfig] = None,
    ) -> None:
        self.engine: TextEngine = text_engine
        self.config: SyntheticDataConfig = config or SyntheticDataConfig()
        self._logger = get_logger(self.__class__.__name__)
        self._rng: random.Random = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Instruction generation
    # ------------------------------------------------------------------
    def generate_instructions(
        self,
        topic: str,
        num_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate diverse instruction prompts for ``topic``.

        Args:
            topic: The subject/domain for the instructions.
            num_samples: Number of instructions to generate.

        Returns:
            A list of dictionaries, each with ``id``, ``topic``,
            ``instruction``, and ``quality`` keys.
        """
        instructions: List[Dict[str, Any]] = []
        seen: set = set()

        prompt = self._build_instruction_prompt(topic, count=num_samples)

        attempts = 0
        max_attempts = num_samples * 3

        while len(instructions) < num_samples and attempts < max_attempts:
            attempts += 1
            text = self._generate(prompt)
            # Parse the generated text into individual instructions.
            candidates = self._parse_instructions(text, topic)
            for candidate in candidates:
                instruction = candidate["instruction"].strip()
                if not instruction or len(instruction) < 8:
                    continue
                key = self._normalise(instruction)
                if key in seen:
                    continue
                if not self._is_diverse(key, seen):
                    continue
                seen.add(key)
                instructions.append(candidate)
                if len(instructions) >= num_samples:
                    break

        # If the model did not produce enough, synthesise fallbacks.
        while len(instructions) < num_samples:
            instructions.append(
                self._fallback_instruction(topic, len(instructions))
            )

        self._logger.info(
            "Generated %d instructions for topic '%s'.", len(instructions), topic
        )
        return instructions[:num_samples]

    def _build_instruction_prompt(self, topic: str, count: int) -> str:
        """Build the prompt used to elicit instructions."""
        return (
            f"You are a helpful assistant that generates diverse, high-quality "
            f"task instructions about the topic: '{topic}'.\n"
            f"Generate {count} distinct instructions. "
            f"Write each instruction on a new line prefixed with a number and "
            f"a period (e.g. '1. <instruction>').\n"
            f"Make the instructions varied in difficulty and style.\n\n"
            f"Instructions:\n"
        )

    def _parse_instructions(
        self, text: str, topic: str
    ) -> List[Dict[str, Any]]:
        """Parse numbered instructions from generated text.

        Args:
            text: The raw model output.
            topic: The topic for tagging.

        Returns:
            A list of instruction dictionaries.
        """
        results: List[Dict[str, Any]] = []
        # Match lines like "1. Do something" or "- Do something".
        pattern = re.compile(r"^(?:\d+[\.\)]\s*|[-*]\s*)(.+)$")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            instruction = match.group(1).strip() if match else line
            if len(instruction) < 5:
                continue
            results.append(
                {
                    "id": f"inst-{topic[:20]}-{len(results)}",
                    "topic": topic,
                    "instruction": instruction,
                    "quality": self._score_quality(instruction),
                }
            )
        return results

    def _fallback_instruction(self, topic: str, index: int) -> Dict[str, Any]:
        """Generate a deterministic fallback instruction.

        Args:
            topic: The topic.
            index: A disambiguating index.

        Returns:
            An instruction dictionary.
        """
        templates = [
            "Explain the key concepts of {topic}.",
            "What are the main applications of {topic}?",
            "Describe a common challenge in {topic} and how to solve it.",
            "Compare and contrast different approaches to {topic}.",
            "Provide a step-by-step guide for {topic}.",
            "What are the best practices for {topic}?",
            "Summarise the current state of research in {topic}.",
            "Give an example of {topic} in a real-world scenario.",
        ]
        instruction = templates[index % len(templates)].format(topic=topic)
        return {
            "id": f"inst-{topic[:20]}-{index}",
            "topic": topic,
            "instruction": instruction,
            "quality": 0.6,
        }

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------
    def generate_responses(
        self,
        instructions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate responses for a list of instructions.

        Args:
            instructions: A list of instruction dictionaries (as produced
                by :meth:`generate_instructions`).

        Returns:
            A list of dictionaries with ``instruction``, ``response``,
            and ``quality`` keys.
        """
        results: List[Dict[str, Any]] = []
        for item in instructions:
            instruction = item.get("instruction", item.get("text", ""))
            if not instruction:
                continue
            prompt = (
                f"Instruction: {instruction}\n\n"
                f"Provide a clear, accurate, and helpful response.\n\n"
                f"Response:"
            )
            response = self._generate(prompt)
            results.append(
                {
                    "instruction": instruction,
                    "response": response.strip(),
                    "quality": self._score_quality(response),
                    "topic": item.get("topic", ""),
                }
            )
        self._logger.info("Generated %d responses.", len(results))
        return results

    # ------------------------------------------------------------------
    # Preference pair generation (for DPO)
    # ------------------------------------------------------------------
    def generate_preference_pairs(
        self,
        instructions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate chosen/rejected preference pairs for DPO.

        For each instruction two responses are generated: one with a
        high-quality prompt (chosen) and one with a degraded prompt
        (rejected).  The pair is tagged with a quality differential.

        Args:
            instructions: A list of instruction dictionaries.

        Returns:
            A list of preference-pair dictionaries with ``prompt``,
            ``chosen``, ``rejected``, and ``quality_diff`` keys.
        """
        results: List[Dict[str, Any]] = []
        for item in instructions:
            instruction = item.get("instruction", item.get("text", ""))
            if not instruction:
                continue

            # Chosen: high-quality, detailed response.
            chosen_prompt = (
                f"You are an expert assistant. Provide a thorough, accurate, "
                f"and well-structured answer.\n\n"
                f"Question: {instruction}\n\nAnswer:"
            )
            chosen = self._generate(chosen_prompt).strip()

            # Rejected: lower-quality, terse response.
            rejected_prompt = (
                f"Answer briefly and with minimal detail.\n\n"
                f"Question: {instruction}\n\nAnswer:"
            )
            rejected = self._generate(
                rejected_prompt,
                temperature=max(0.1, self.config.temperature - 0.3),
            ).strip()

            chosen_q = self._score_quality(chosen)
            rejected_q = self._score_quality(rejected)

            # Ensure chosen is actually better; swap if needed.
            if chosen_q < rejected_q:
                chosen, rejected = rejected, chosen
                chosen_q, rejected_q = rejected_q, chosen_q

            results.append(
                {
                    "prompt": instruction,
                    "chosen": chosen,
                    "rejected": rejected,
                    "quality_diff": chosen_q - rejected_q,
                }
            )

        self._logger.info("Generated %d preference pairs.", len(results))
        return results

    # ------------------------------------------------------------------
    # Conversation generation
    # ------------------------------------------------------------------
    def generate_conversation(
        self,
        num_turns: int = 5,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a multi-turn conversation.

        Args:
            num_turns: Total number of user/assistant turns.
            topic: Optional topic to seed the conversation.

        Returns:
            A list of message dictionaries with ``role`` and ``content``
            keys (OpenAI-style).
        """
        messages: List[Dict[str, Any]] = []
        topic_hint = f" about {topic}" if topic else ""

        # Seed with a user message.
        seed_prompt = (
            f"Generate a natural opening question{topic_hint} that a user "
            f"might ask an assistant. Respond with only the question."
        )
        first_user = self._generate(seed_prompt, temperature=0.9).strip()
        if not first_user:
            first_user = f"Can you tell me about {topic or 'a topic'}?"
        messages.append({"role": "user", "content": first_user})

        for turn in range(num_turns):
            # Assistant turn.
            assistant_prompt = self._build_conversation_prompt(messages)
            assistant_reply = self._generate(assistant_prompt).strip()
            messages.append({"role": "assistant", "content": assistant_reply})

            if turn == num_turns - 1:
                break

            # User follow-up.
            followup_prompt = (
                f"Given this conversation so far:\n"
                f"{self._format_messages(messages)}\n\n"
                f"Generate a natural follow-up question from the user. "
                f"Respond with only the question."
            )
            user_reply = self._generate(followup_prompt, temperature=0.9).strip()
            if not user_reply:
                user_reply = "Can you elaborate on that?"
            messages.append({"role": "user", "content": user_reply})

        self._logger.info(
            "Generated a %d-turn conversation.", len(messages)
        )
        return messages

    def _build_conversation_prompt(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """Build a prompt for generating the next assistant turn.

        Args:
            messages: The conversation so far.

        Returns:
            The formatted prompt string.
        """
        return self._format_messages(messages) + "\n[ASSISTANT]"

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """Render messages into a text transcript."""
        parts: List[str] = []
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}] {msg['content']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------
    def filter_quality(
        self,
        data: List[Dict[str, Any]],
        min_quality: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Filter out low-quality samples.

        Each item is scored (or re-scored) and only those with a quality
        score >= ``min_quality`` are kept.

        Args:
            data: A list of sample dictionaries.
            min_quality: Minimum quality threshold in ``[0, 1]``.

        Returns:
            The filtered list of samples.
        """
        kept: List[Dict[str, Any]] = []
        for item in data:
            quality = item.get("quality")
            if quality is None:
                # Score based on the longest text field.
                text = self._longest_text(item)
                quality = self._score_quality(text)
                item["quality"] = quality
            if quality >= min_quality:
                kept.append(item)
        self._logger.info(
            "Filtered %d -> %d samples (min_quality=%.2f).",
            len(data), len(kept), min_quality,
        )
        return kept

    @staticmethod
    def _longest_text(item: Dict[str, Any]) -> str:
        """Return the longest string value in ``item``."""
        longest = ""
        for value in item.values():
            if isinstance(value, str) and len(value) > len(longest):
                longest = value
        return longest

    def _score_quality(self, text: str) -> float:
        """Score the quality of ``text`` on a ``[0, 1]`` scale.

        The heuristic combines length adequacy, lexical diversity, and
        absence of repetition.

        Args:
            text: The text to score.

        Returns:
            A quality score in ``[0, 1]``.
        """
        if not text or not text.strip():
            return 0.0

        words = text.split()
        n = len(words)

        # Length score: peaks around 30-150 words.
        if n < 5:
            length_score = n / 5.0 * 0.5
        elif n <= 150:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (n - 150) / 500.0)

        # Lexical diversity: unique words / total words.
        unique = len(set(w.lower() for w in words))
        diversity = unique / max(1, n)

        # Repetition penalty: detect repeated trigrams.
        trigrams = [tuple(words[i : i + 3]) for i in range(max(1, n - 2))]
        if trigrams:
            repeat_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
        else:
            repeat_ratio = 0.0

        score = 0.4 * length_score + 0.4 * diversity + 0.2 * (1.0 - repeat_ratio)
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Diversity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise text for deduplication (lowercase, strip punctuation)."""
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _is_diverse(self, candidate: str, existing: set) -> bool:
        """Check that ``candidate`` is sufficiently different from existing.

        Uses a simple token-overlap ratio as a proxy for edit distance.

        Args:
            candidate: Normalised candidate text.
            existing: Set of normalised existing texts.

        Returns:
            ``True`` if the candidate is diverse enough.
        """
        cand_tokens = set(candidate.split())
        if not cand_tokens:
            return False
        for other in existing:
            other_tokens = set(other.split())
            if not other_tokens:
                continue
            overlap = len(cand_tokens & other_tokens) / max(
                len(cand_tokens), len(other_tokens)
            )
            if overlap > (1.0 - self.config.diversity_threshold):
                return False
        return True

    # ------------------------------------------------------------------
    # Generation wrapper
    # ------------------------------------------------------------------
    def _generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text via the engine with error handling.

        Args:
            prompt: The input prompt.
            temperature: Optional temperature override.

        Returns:
            The generated text (empty string on failure).
        """
        temp = temperature if temperature is not None else self.config.temperature
        try:
            text = self.engine.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=temp,
                top_p=self.config.top_p,
            )
            if isinstance(text, str):
                return text
            # If streaming was accidentally enabled, drain the iterator.
            return "".join(text)  # type: ignore[arg-type]
        except Exception as exc:
            self._logger.warning("Generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save_to_jsonl(data: List[Dict[str, Any]], path: PathLike) -> Path:
        """Save a list of dictionaries to a JSONL file.

        Args:
            data: The list of dictionaries to save.
            path: Target file path.

        Returns:
            The resolved :class:`~pathlib.Path` of the saved file.
        """
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return target

    @staticmethod
    def load_from_jsonl(path: PathLike) -> List[Dict[str, Any]]:
        """Load a list of dictionaries from a JSONL file.

        Args:
            path: Path to the JSONL file.

        Returns:
            A list of dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"JSONL file not found: {source}")
        data: List[Dict[str, Any]] = []
        with open(source, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
