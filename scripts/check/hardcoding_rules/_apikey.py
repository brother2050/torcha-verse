"""Rule #9: a string literal that looks like a hardcoded API key."""

from __future__ import annotations

import re
from typing import List, Tuple

from ._helpers import truncate
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["ApiKeyPatternRule"]


class ApiKeyPatternRule(Rule):
    """Rule #9: a string literal that looks like a hardcoded API key.

    Fires on any string literal that matches one of the
    well-known API-key prefixes for popular public APIs:

    * ``sk-...`` (OpenAI / OpenAI-compat)
    * ``sk-ant-...`` (Anthropic)
    * ``ghp_...`` / ``gho_...`` / ``ghs_...`` / ``ghu_...`` /
      ``ghr_...`` (GitHub personal / OAuth / server / user / refresh)
    * ``AKIA`` + 16 uppercase alphanumerics (AWS access key id)
    * ``AIza`` + 35 alphanumerics (Google API key)
    * ``xoxb-...`` / ``xoxp-...`` (Slack bot / user tokens)
    * ``hf_...`` (Hugging Face)

    Docstrings and ``__all__`` entries are exempt.  The default
    severity is ``"critical"`` -- a leaked API key is a security
    incident waiting to happen, so the scanner should make the
    operator fix the violation rather than whitelist it.
    """

    name = "api_key_pattern"
    description = "string literal that looks like a hardcoded API key (critical)"
    default_severity = "critical"

    #: Regexes for the well-known API-key prefixes.
    _PATTERNS: Tuple = (
        re.compile(r"^sk-[A-Za-z0-9_\-]{20,}"),
        re.compile(r"^sk-ant-[A-Za-z0-9_\-]{20,}"),
        re.compile(r"^gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"^AKIA[0-9A-Z]{16}"),
        re.compile(r"^AIza[0-9A-Za-z_\-]{35}"),
        re.compile(r"^xox[bp]-[A-Za-z0-9\-]{20,}"),
        re.compile(r"^hf_[A-Za-z0-9]{20,}"),
    )

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not isinstance(ctx.value, str):
            return []
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        # Don't shout at operators when the string is already
        # the argument of an env-var lookup (``os.environ[...]``).
        if ctx.in_runtime_attr:
            return []
        v = ctx.value.strip()
        for pat in self._PATTERNS:
            if pat.match(v):
                return [ViolationCandidate(
                    type=self.name,
                    content=truncate(repr(ctx.value)),
                    severity=self.default_severity,
                )]
        return []
