"""Security layer for the TorchaVerse framework (v0.3.0).

This package implements the **defence-in-depth** security pipeline
through four sequential gates:

1. **Input sanitisation** (:mod:`security.input_sanitizer`) -- NFC
   normalisation, control-character stripping, path-traversal detection,
   path whitelisting and prompt-injection detection.
2. **Execution sandbox** (:mod:`security.sandbox`) -- AST-based static
   analysis of untrusted Python code plus a restricted execution
   environment with timeout and memory limits.
3. **Output filtering** (:mod:`security.output_filter`) -- toxicity,
   NSFW and audio content screening of model outputs.
4. **Supply-chain & audit** (:mod:`security.audit`) -- dependency
   vulnerability scanning, CycloneDX SBOM generation, licence compliance
   and audit-log aggregation.

The entire package is **pure Python** -- it does *not* depend on
``torch`` -- so it can be imported in lightweight environments.  All
optional third-party backends (``RestrictedPython``, ``Detoxify``,
``NudeNet``, ``pip-audit``, ``safety``) are imported lazily with
``try/except`` guards.

Example:
    >>> from security import InputSanitizer, ASTAnalyzer
    >>> s = InputSanitizer()
    >>> s.sanitize_text("hello world")
    'hello world'
"""

from __future__ import annotations

from .audit import LicenseCheck, SecurityAudit, Vulnerability
from .input_sanitizer import InjectionResult, InputSanitizer
from .output_filter import FilterResult, OutputFilter
from .sandbox import (
    ASTAnalyzer,
    AnalysisResult,
    SandboxConfig,
    SandboxExecutor,
    SandboxTimeoutError,
    SandboxViolationError,
)

__all__ = [
    # Gate 1 -- input sanitisation
    "InputSanitizer",
    "InjectionResult",
    # Gate 2 -- execution sandbox
    "SandboxConfig",
    "ASTAnalyzer",
    "AnalysisResult",
    "SandboxExecutor",
    "SandboxTimeoutError",
    "SandboxViolationError",
    # Gate 3 -- output filtering
    "OutputFilter",
    "FilterResult",
    # Gate 4 -- supply chain & audit
    "SecurityAudit",
    "Vulnerability",
    "LicenseCheck",
]
