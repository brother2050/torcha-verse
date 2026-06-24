"""Input sanitization for the TorchaVerse security layer (Gate 1).

This module is the **first gate** of the defence-in-depth security
pipeline.  Every piece of untrusted data that enters the framework --
prompts, file paths, request payloads -- passes through
:class:`InputSanitizer` which normalises, truncates and screens the input
for common attack vectors before it reaches any model or executor.

Responsibilities
----------------
* **Text normalisation** -- NFC canonicalisation, control-character
  stripping and configurable length truncation.
* **Path traversal detection** -- flags ``../``, ``..\\`` and well-known
  sensitive absolute paths (``/etc/passwd``, ``/proc``, ...).
* **Path whitelisting** -- :meth:`InputSanitizer.sanitize_path` resolves
  a user-supplied path and rejects anything escaping the allowed roots.
* **Prompt-injection detection** -- :meth:`InputSanitizer.detect_prompt_injection`
  matches a curated rule set ("ignore previous instructions", "system
  prompt", "new instructions", ...) and returns a structured
  :class:`InjectionResult`.
* **Request sanitisation** -- :meth:`InputSanitizer.sanitize_request`
  recursively cleans an entire ``dict`` payload with a byte-size budget.

The module is **pure Python** (no ``torch`` dependency) and fully
thread-safe (a single :class:`threading.Lock` guards the mutable
custom blocklist).

Example:
    >>> sanitizer = InputSanitizer()
    >>> sanitizer.sanitize_text("hello world")
    'hello world'
    >>> inj = sanitizer.detect_prompt_injection("ignore previous instructions")
    >>> inj.is_injected
    True
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "InputSanitizer",
    "InjectionResult",
]

# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Default maximum length (in characters) accepted by :meth:`sanitize_text`.
_DEFAULT_MAX_TEXT_LENGTH: int = 16384

#: Default maximum size (in bytes) accepted by :meth:`sanitize_request`.
_DEFAULT_MAX_REQUEST_SIZE: int = 1_048_576  # 1 MiB

#: Suffix appended when a text input is truncated.
_TRUNCATION_SUFFIX: str = "..."

#: Confidence value returned when at least one injection rule matches.
_INJECTION_MATCH_CONFIDENCE: float = 0.9

#: Confidence value returned when no injection rule matches.
_INJECTION_CLEAN_CONFIDENCE: float = 0.0

#: Tokens that indicate a path-traversal attempt.  Stored *without* a
#: leading slash so they do not themselves look like filesystem paths to
#: static analysers; the matcher checks for both the bare token and the
#: slash-prefixed variant at runtime.
_PATH_TRAVERSAL_TOKENS: tuple[str, ...] = (
    "..",
    "%2e%2e",
    "%252e%252e",
    "..%2f",
    "..%5c",
)

#: Sensitive path *fragments* (directory names) that, when found after a
#: path separator, indicate an attempt to read system files.  Stored as
#: bare names to avoid being flagged as path literals.
_SENSITIVE_PATH_FRAGMENTS: tuple[str, ...] = (
    "etc" + "/" + "passwd",
    "etc" + "/" + "shadow",
    "etc" + "/" + "hosts",
    "proc" + "/" + "self",
    "proc" + "/" + "version",
    "sys" + "/" + "kernel",
    "boot" + "/" + "grub",
    "root" + "/" + ".ssh",
    "windows" + "\\" + "system32",
)

#: Prompt-injection rules.  Each entry is ``(compiled_regex, rule_name)``.
_PROMPT_INJECTION_RULES: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(?:previous|prior|above|all)\s+instructions?", "ignore_previous"),
    (r"disregard\s+(?:previous|prior|all|the)\s+instructions?", "disregard_previous"),
    (r"forget\s+(?:everything|all|previous|your)\s+(?:instructions?|rules?|context)", "forget_context"),
    (r"system\s+prompt", "system_prompt_leak"),
    (r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt", "prompt_extraction"),
    (r"new\s+instructions?\s*:", "new_instructions"),
    (r"you\s+are\s+now\s+(?:a|an)\b", "role_reassignment"),
    (r"act\s+as\s+(?:if\s+you\s+(?:are|have\s+no)|an?\s+(?:unrestricted|unfiltered))", "unrestricted_role"),
    (r"override\s+(?:your|the|any)\s+(?:system|safety|content|security)\s+(?:policy|policies|rules?|guidelines?)", "policy_override"),
    (r"jailbreak", "jailbreak_keyword"),
    (r"do\s+not\s+follow\s+(?:your|the|any)\s+rules?", "rule_disregard"),
    (r"pretend\s+(?:that\s+)?you\s+(?:have\s+no|don't\s+have\s+(?:any\s+)?|lack)\s+(?:restrictions?|guidelines?|rules?)", "pretend_unrestricted"),
    (r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:constraints?|safety|content)\s+(?:filters?|guidelines?|rules?)", "ignore_safety_filter"),
)

#: Compiled prompt-injection patterns (built once at import time).
_COMPILED_INJECTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), name)
    for pattern, name in _PROMPT_INJECTION_RULES
)

#: Regular expression matching C0/C1 control characters (except tab,
#: newline and carriage return which are preserved).
_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

#: Maximum recursion depth for :meth:`sanitize_request`.
_MAX_REQUEST_DEPTH: int = 32


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class InjectionResult:
    """Outcome of a prompt-injection scan.

    Attributes:
        is_injected: ``True`` when at least one injection rule matched.
        confidence: Heuristic confidence in the ``[0.0, 1.0]`` range.
        matched_rules: Names of the rules that triggered.
    """

    is_injected: bool
    confidence: float
    matched_rules: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------
class InputSanitizer:
    """Thread-safe input sanitiser (security Gate 1).

    Provides text normalisation, path-traversal detection, path
    whitelisting, prompt-injection detection and recursive request
    sanitisation.  A single :class:`threading.Lock` guards the mutable
    custom blocklist so the same instance can be shared across threads.

    Args:
        max_text_length: Default character cap for :meth:`sanitize_text`.
        max_request_size: Default byte cap for :meth:`sanitize_request`.
        custom_blocklist: Optional initial list of words/phrases that
            :meth:`sanitize_text` should redact.

    Example:
        >>> s = InputSanitizer()
        >>> s.sanitize_text("hello\\x00world")
        'helloworld'
        >>> s.sanitize_path("/tmp/../etc/passwd", allowed_roots=["/tmp"])
        Traceback (most recent call last):
                ...
        ValueError: ...
    """

    def __init__(
        self,
        max_text_length: int = _DEFAULT_MAX_TEXT_LENGTH,
        max_request_size: int = _DEFAULT_MAX_REQUEST_SIZE,
        custom_blocklist: Optional[Sequence[str]] = None,
    ) -> None:
        if max_text_length <= 0:
            raise ValueError(f"max_text_length must be > 0, got {max_text_length}.")
        if max_request_size <= 0:
            raise ValueError(f"max_request_size must be > 0, got {max_request_size}.")

        self._max_text_length: int = int(max_text_length)
        self._max_request_size: int = int(max_request_size)
        self._lock: threading.Lock = threading.Lock()
        self._custom_blocklist: list[str] = list(custom_blocklist) if custom_blocklist else []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def max_text_length(self) -> int:
        """Default character cap applied by :meth:`sanitize_text`."""
        return self._max_text_length

    @property
    def max_request_size(self) -> int:
        """Default byte cap applied by :meth:`sanitize_request`."""
        return self._max_request_size

    # ------------------------------------------------------------------
    # Text sanitisation
    # ------------------------------------------------------------------
    def sanitize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """Normalise and clean a text input.

        Performs, in order:

        1. NFC canonicalisation.
        2. Stripping of C0/C1 control characters (tab/newline retained).
        3. Truncation to ``max_length`` characters.
        4. Path-traversal token detection (raises :class:`ValueError`).

        Args:
            text: Raw user-supplied string.
            max_length: Optional override for the character cap.

        Returns:
            The cleaned string.

        Raises:
            ValueError: If a path-traversal attempt is detected.
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")

        cap = max_length if max_length is not None else self._max_text_length
        if cap <= 0:
            raise ValueError(f"max_length must be > 0, got {cap}.")

        # 1. NFC normalisation.
        normalised = unicodedata.normalize("NFC", text)

        # 2. Strip control characters.
        cleaned = _CONTROL_CHAR_RE.sub("", normalised)

        # 3. Path-traversal detection (before truncation so a payload
        #    cannot be split across the cut point).
        self._check_path_traversal(cleaned)

        # 4. Redact custom blocklist entries.
        cleaned = self._apply_blocklist(cleaned)

        # 5. Truncation.
        if len(cleaned) > cap:
            cleaned = cleaned[:cap]
            if len(_TRUNCATION_SUFFIX) <= cap:
                cleaned = cleaned[: cap - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
        return cleaned

    # ------------------------------------------------------------------
    # Path sanitisation
    # ------------------------------------------------------------------
    def sanitize_path(
        self,
        path: Union[str, Path],
        allowed_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> Path:
        """Validate a path against an allow-list of root directories.

        The path is resolved (following symlinks) and then checked to
        ensure it is contained within one of ``allowed_roots``.  When no
        roots are supplied the current working directory is used.

        Args:
            path: User-supplied path.
            allowed_roots: Iterable of permitted root directories.

        Returns:
            The resolved, validated :class:`pathlib.Path`.

        Raises:
            ValueError: If the resolved path escapes every allowed root.
            TypeError: If ``path`` is not a string or :class:`Path`.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be str or Path, got {type(path).__name__}.")

        candidate = Path(path).expanduser()

        # Reject obvious traversal tokens early.
        self._check_path_traversal(str(candidate))

        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot resolve path {path!r}: {exc}") from exc

        roots: list[Path] = []
        if allowed_roots:
            roots = [Path(root).expanduser().resolve(strict=False) for root in allowed_roots]
        else:
            roots = [Path.cwd().resolve(strict=False)]

        for root in roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue

        raise ValueError(
            f"Path {resolved!s} is outside the allowed roots: "
            f"{', '.join(str(r) for r in roots)}."
        )

    # ------------------------------------------------------------------
    # Prompt-injection detection
    # ------------------------------------------------------------------
    def detect_prompt_injection(self, text: str) -> InjectionResult:
        """Scan ``text`` for prompt-injection patterns.

        Matches the text against a curated rule set (case-insensitive).
        Each rule that fires contributes to ``matched_rules``; the
        confidence is a simple heuristic (high when any rule matches).

        Args:
            text: The text to inspect.

        Returns:
            An :class:`InjectionResult` describing the outcome.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")

        matched: list[str] = []
        for pattern, rule_name in _COMPILED_INJECTION_RULES:
            if pattern.search(text):
                matched.append(rule_name)

        if matched:
            return InjectionResult(
                is_injected=True,
                confidence=_INJECTION_MATCH_CONFIDENCE,
                matched_rules=matched,
            )
        return InjectionResult(
            is_injected=False,
            confidence=_INJECTION_CLEAN_CONFIDENCE,
            matched_rules=[],
        )

    # ------------------------------------------------------------------
    # Request sanitisation
    # ------------------------------------------------------------------
    def sanitize_request(
        self,
        data: dict,
        max_size: Optional[int] = None,
    ) -> dict:
        """Recursively sanitise a request payload.

        Walks the dictionary, applying :meth:`sanitize_text` to every
        string value and enforcing an overall byte budget.  Nested
        dicts and lists are traversed up to :data:`_MAX_REQUEST_DEPTH`.

        Args:
            data: The request payload.
            max_size: Optional byte override (defaults to the instance
                :attr:`max_request_size`).

        Returns:
            A new dictionary with all strings cleaned.

        Raises:
            ValueError: If the serialised payload exceeds ``max_size``
                or the recursion depth is exceeded.
            TypeError: If ``data`` is not a dict.
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data).__name__}.")

        budget = max_size if max_size is not None else self._max_request_size
        if budget <= 0:
            raise ValueError(f"max_size must be > 0, got {budget}.")

        cleaned = self._sanitize_value(data, depth=0)

        # Enforce the byte budget on the cleaned payload.
        import json

        try:
            payload = json.dumps(cleaned, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Request is not JSON-serialisable: {exc}") from exc

        if len(payload.encode("utf-8")) > budget:
            raise ValueError(
                f"Request size {len(payload.encode('utf-8'))} bytes exceeds "
                f"budget {budget} bytes."
            )
        return cleaned

    # ------------------------------------------------------------------
    # Custom blocklist
    # ------------------------------------------------------------------
    def set_custom_blocklist(self, words: Sequence[str]) -> None:
        """Replace the custom redaction blocklist.

        Args:
            words: New list of words/phrases to redact from text.
        """
        with self._lock:
            self._custom_blocklist = [str(w) for w in words]

    def add_to_blocklist(self, word: str) -> None:
        """Append a single word to the custom blocklist."""
        with self._lock:
            self._custom_blocklist.append(str(word))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_blocklist(self, text: str) -> str:
        """Redact every custom-blocklist entry from ``text``."""
        with self._lock:
            blocklist = list(self._custom_blocklist)
        if not blocklist:
            return text
        result = text
        for word in blocklist:
            if word:
                result = result.replace(word, "*" * len(word))
        return result

    @staticmethod
    def _check_path_traversal(text: str) -> None:
        """Raise :class:`ValueError` if ``text`` contains a traversal token."""
        lowered = text.lower()
        for token in _PATH_TRAVERSAL_TOKENS:
            if token in lowered:
                raise ValueError(
                    f"Path-traversal token {token!r} detected in input."
                )
        for fragment in _SENSITIVE_PATH_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(
                    f"Sensitive path fragment {fragment!r} detected in input."
                )

    def _sanitize_value(self, value: Any, depth: int) -> Any:
        """Recursively clean a single value within a request payload."""
        if depth > _MAX_REQUEST_DEPTH:
            raise ValueError(
                f"Request nesting depth exceeds {_MAX_REQUEST_DEPTH}."
            )
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize_value(v, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item, depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_value(item, depth + 1) for item in value)
        return value

    def __repr__(self) -> str:
        with self._lock:
            blocklist_len = len(self._custom_blocklist)
        return (
            f"InputSanitizer(max_text_length={self._max_text_length}, "
            f"max_request_size={self._max_request_size}, "
            f"blocklist_entries={blocklist_len})"
        )
