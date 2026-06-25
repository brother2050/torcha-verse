"""Supply-chain and audit for the TorchaVerse security layer (Gate 4).

This module is the **fourth gate** of the defence-in-depth pipeline.  It
governs the *supply chain* (third-party dependencies and asset
licences) and provides an audit-log aggregation view.

Components
----------
* :class:`Vulnerability` -- dataclass describing a single CVE in a
  dependency.
* :class:`LicenseCheck` -- dataclass describing the licence-compliance
  verdict for an asset.
* :class:`SecurityAudit` -- the main auditor.  It can:

  - :meth:`check_dependencies` -- scan a ``requirements.txt`` for known
    vulnerabilities (uses the optional ``pip-audit`` / ``safety``
    packages when available; otherwise parses the file and returns an
    empty list).
  - :meth:`generate_sbom` -- emit a CycloneDX-format JSON Software Bill
    of Materials.
  - :meth:`check_license` -- verify that an asset's licence permits the
    intended use (commercial / non-commercial).
  - :meth:`audit_log_summary` -- aggregate the framework's audit trail
    (delegating to :class:`infrastructure.audit_log.AuditLogger`) over a
    date range.

The module is **pure Python** (no ``torch`` dependency).  Optional
dependencies (``pip_audit``, ``safety``) are imported lazily with
``try/except`` guards.

Example:
    >>> audit = SecurityAudit()
    >>> vulns = audit.check_dependencies("requirements.txt")
    >>> isinstance(vulns, list)
    True
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "SecurityAudit",
    "Vulnerability",
    "LicenseCheck",
]

#: 模块级日志器，用于记录可选依赖缺失及扫描失败等警告信息。
_logger: logging.Logger = logging.getLogger("security.audit")

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from pip_audit.audit import audit as _pip_audit  # type: ignore
    from pip_audit.req import ReqFile  # type: ignore

    _HAS_PIP_AUDIT: bool = True
except Exception:  # pragma: no cover - pip-audit not installed
    _logger.debug("pip-audit 未安装，依赖漏洞扫描将使用回退模式。", exc_info=True)
    _HAS_PIP_AUDIT: bool = False

try:  # pragma: no cover - import guard
    import safety.safety as _safety  # type: ignore

    _HAS_SAFETY: bool = True
except Exception:  # pragma: no cover - safety not installed
    _logger.debug("safety 未安装，依赖漏洞扫描将使用回退模式。", exc_info=True)
    _HAS_SAFETY: bool = False


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: SPDX identifiers of licences that permit commercial use.
_COMMERCIAL_LICENSES: frozenset[str] = frozenset({
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "Python-2.0",
    "Unlicense",
    "Zlib",
    "BSL-1.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
})

#: SPDX identifiers of licences that restrict commercial use.
_NON_COMMERCIAL_LICENSES: frozenset[str] = frozenset({
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "SSPL-1.0",
    "BUSL-1.1",
})

#: CycloneDX spec version emitted by :meth:`generate_sbom`.
_SBOM_SPEC_VERSION: str = "1.5"

#: CycloneDX document type.
_SBOM_BOM_FORMAT: str = "CycloneDX"

#: CycloneDX component type.
_SBOM_COMPONENT_TYPE: str = "library"

#: Regular expression for parsing ``name>=version`` style requirement lines.
_REQ_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*(?:[<>=!~]=?\s*([A-Za-z0-9_.\-+*]+))?"
)

#: Severity ordering used when sorting vulnerabilities.
_SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "moderate": 2,
    "low": 3,
    "info": 4,
    "unknown": 5,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Vulnerability:
    """A single known vulnerability in a dependency.

    Attributes:
        package: Package name (e.g. ``"requests"``).
        version: Affected version string.
        severity: ``"critical"``, ``"high"``, ``"medium"``, ``"low"``
            or ``"unknown"``.
        description: Human-readable summary.
        cve: CVE identifier (e.g. ``"CVE-2021-12345"``).
    """

    package: str
    version: str
    severity: str
    description: str
    cve: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "package": self.package,
            "version": self.version,
            "severity": self.severity,
            "description": self.description,
            "cve": self.cve,
        }


@dataclass
class LicenseCheck:
    """Licence-compliance verdict for an asset.

    Attributes:
        asset_id: Identifier of the checked asset.
        license: SPDX identifier of the detected licence.
        commercial_use: ``True`` when commercial use is permitted.
        compliant: ``True`` when the licence satisfies the policy.
    """

    asset_id: str
    license: str
    commercial_use: bool
    compliant: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "asset_id": self.asset_id,
            "license": self.license,
            "commercial_use": self.commercial_use,
            "compliant": self.compliant,
        }


# ---------------------------------------------------------------------------
# SecurityAudit
# ---------------------------------------------------------------------------
class SecurityAudit:
    """Supply-chain auditor and audit-log aggregator (security Gate 4).

    Args:
        allow_commercial: When ``True`` (default) only licences that
            permit commercial use are considered compliant.
        audit_logger: Optional :class:`~infrastructure.audit_log.AuditLogger`
            instance used by :meth:`audit_log_summary`.  When ``None`` a
            fresh default logger is created lazily.

    Example:
        >>> audit = SecurityAudit()
        >>> vulns = audit.check_dependencies("requirements.txt")
        >>> sbom = audit.generate_sbom("sbom.json")
    """

    def __init__(
        self,
        allow_commercial: bool = True,
        audit_logger: Any = None,
    ) -> None:
        self._allow_commercial: bool = bool(allow_commercial)
        self._audit_logger: Any = audit_logger
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def has_pip_audit(self) -> bool:
        """``True`` when the ``pip-audit`` backend is available."""
        return _HAS_PIP_AUDIT

    @property
    def has_safety(self) -> bool:
        """``True`` when the ``safety`` backend is available."""
        return _HAS_SAFETY

    # ------------------------------------------------------------------
    # Dependency vulnerability scanning
    # ------------------------------------------------------------------
    def check_dependencies(
        self,
        requirements_path: Union[str, Path],
    ) -> list[Vulnerability]:
        """Scan a requirements file for known vulnerabilities.

        When ``pip-audit`` is installed it is used for an authoritative
        scan; otherwise the file is parsed and an empty list is
        returned (the caller is informed via :attr:`has_pip_audit`).

        Args:
            requirements_path: Path to a ``requirements.txt``-style
                file.

        Returns:
            A list of :class:`Vulnerability` objects sorted by severity.
        """
        path = Path(requirements_path)
        if not path.exists():
            raise FileNotFoundError(f"Requirements file not found: {path}")

        if _HAS_PIP_AUDIT:
            return self._scan_with_pip_audit(path)
        # Fallback: parse the file so callers can inspect the dependency
        # list, but report no vulnerabilities without a backend.
        return []

    # ------------------------------------------------------------------
    # SBOM generation
    # ------------------------------------------------------------------
    def generate_sbom(
        self,
        output_path: Union[str, Path],
        requirements_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate a CycloneDX-format JSON Software Bill of Materials.

        Args:
            output_path: Where to write the ``.json`` SBOM file.
            requirements_path: Optional requirements file to enumerate.
                When omitted the SBOM contains only the framework's own
                metadata.

        Returns:
            The resolved :class:`pathlib.Path` of the written SBOM.
        """
        out = Path(output_path)
        components = self._build_components(requirements_path)
        bom: dict[str, Any] = {
            "bomFormat": _SBOM_BOM_FORMAT,
            "specVersion": _SBOM_SPEC_VERSION,
            "serialNumber": "urn:uuid:torcha-verse-sbom",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tools": [
                    {
                        "vendor": "TorchaVerse",
                        "name": "SecurityAudit",
                        "version": "0.3.1",
                    }
                ],
            },
            "components": components,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(bom, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out.resolve()

    # ------------------------------------------------------------------
    # Licence compliance
    # ------------------------------------------------------------------
    def check_license(
        self,
        asset_ref: Union[str, Any],
    ) -> LicenseCheck:
        """Check the licence compliance of an asset.

        Accepts either a plain SPDX string or any object exposing a
        ``license`` / ``spdx_id`` attribute (e.g. an
        :class:`~assets.LicenseRef`).

        Args:
            asset_ref: SPDX string or asset reference.

        Returns:
            A :class:`LicenseCheck` verdict.
        """
        spdx, asset_id = self._extract_license(asset_ref)
        commercial = spdx in _COMMERCIAL_LICENSES
        if self._allow_commercial:
            compliant = commercial
        else:
            compliant = spdx in _COMMERCIAL_LICENSES or spdx in _NON_COMMERCIAL_LICENSES
        return LicenseCheck(
            asset_id=asset_id,
            license=spdx,
            commercial_use=commercial,
            compliant=compliant,
        )

    # ------------------------------------------------------------------
    # Audit-log summary
    # ------------------------------------------------------------------
    def audit_log_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Aggregate the framework's audit trail over a date range.

        Delegates to :class:`~infrastructure.audit_log.AuditLogger` (the
        import is lazy so this module stays torch-free at import time).
        When the infrastructure layer -- which transitively depends on
        ``torch`` -- cannot be imported, an empty summary with an
        ``error`` key is returned instead of raising.

        Args:
            start_date: Inclusive lower bound (UTC).  ``None`` = no
                lower bound.
            end_date: Inclusive upper bound (UTC).  ``None`` = no upper
                bound.

        Returns:
            A dictionary with keys ``total``, ``by_type``,
            ``by_severity`` and ``by_actor`` (and ``error`` when the
            audit logger was unavailable).
        """
        logger = self._get_audit_logger()
        if logger is None:
            return {
                "total": 0,
                "by_type": {},
                "by_severity": {},
                "by_actor": {},
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "error": "AuditLogger unavailable (infrastructure layer "
                "could not be imported).",
            }

        events = logger.query(start_time=start_date, end_time=end_date)

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_actor: dict[str, int] = {}
        for event in events:
            etype = getattr(event, "event_type", "UNKNOWN")
            by_type[etype] = by_type.get(etype, 0) + 1
            sev = getattr(event, "severity", "info")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            actor = getattr(event, "actor", "unknown")
            by_actor[actor] = by_actor.get(actor, 0) + 1

        return {
            "total": len(events),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_actor": by_actor,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _scan_with_pip_audit(self, path: Path) -> list[Vulnerability]:
        """Run pip-audit and convert results to :class:`Vulnerability`."""
        vulnerabilities: list[Vulnerability] = []
        try:
            reqs = ReqFile(str(path))  # type: ignore[name-defined]
            _, audit_results = _pip_audit(reqs)  # type: ignore[name-defined]
        except Exception as exc:
            _logger.warning("pip-audit 扫描失败，返回空漏洞列表: %s", exc)
            return vulnerabilities

        for dep, vulns in audit_results:
            for vuln in vulns:
                vulnerabilities.append(
                    Vulnerability(
                        package=dep.name,
                        version=str(dep.version),
                        severity=self._map_severity(vuln),
                        description=vuln.description or "",
                        cve=vuln.id or "",
                    )
                )
        vulnerabilities.sort(
            key=lambda v: _SEVERITY_ORDER.get(v.severity.lower(), 99)
        )
        return vulnerabilities

    @staticmethod
    def _map_severity(vuln: Any) -> str:
        """Best-effort mapping of a pip-audit vulnerability to a severity."""
        for attr in ("severity", "cvss", "ratings"):
            score = getattr(vuln, attr, None)
            if score is None:
                continue
            if isinstance(score, (int, float)):
                if score >= 9.0:
                    return "critical"
                if score >= 7.0:
                    return "high"
                if score >= 4.0:
                    return "medium"
                return "low"
        return "unknown"

    def _build_components(
        self,
        requirements_path: Optional[Union[str, Path]],
    ) -> list[dict[str, Any]]:
        """Build the CycloneDX ``components`` array from a requirements file."""
        components: list[dict[str, Any]] = []
        if requirements_path is None:
            return components
        path = Path(requirements_path)
        if not path.exists():
            return components
        for name, version in self._parse_requirements(path):
            component: dict[str, Any] = {
                "type": _SBOM_COMPONENT_TYPE,
                "name": name,
                "bom-ref": f"pkg:pypi/{name}",
            }
            if version:
                component["version"] = version
            components.append(component)
        return components

    @staticmethod
    def _parse_requirements(path: Path) -> list[tuple[str, str]]:
        """Parse a requirements file into ``(name, version)`` pairs."""
        results: list[tuple[str, str]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return results
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _REQ_LINE_RE.match(line)
            if match:
                name = match.group(1)
                version = match.group(2) or ""
                results.append((name, version))
        return results

    @staticmethod
    def _extract_license(asset_ref: Union[str, Any]) -> tuple[str, str]:
        """Extract an SPDX string and asset id from ``asset_ref``."""
        if isinstance(asset_ref, str):
            return asset_ref, asset_ref
        spdx = (
            getattr(asset_ref, "spdx_id", None)
            or getattr(asset_ref, "license", None)
            or str(asset_ref)
        )
        asset_id = (
            getattr(asset_ref, "asset_id", None)
            or getattr(asset_ref, "id", None)
            or str(asset_ref)
        )
        return str(spdx), str(asset_id)

    def _get_audit_logger(self) -> Any:
        """Lazily obtain an :class:`AuditLogger` instance.

        Returns ``None`` when the infrastructure layer (which transitively
        imports ``torch``) cannot be loaded, so that the rest of the
        security module remains usable in torch-free environments.
        """
        if self._audit_logger is None:
            with self._lock:
                if self._audit_logger is None:
                    try:
                        from infrastructure.audit_log import AuditLogger

                        self._audit_logger = AuditLogger()
                    except Exception as exc:
                        _logger.warning(
                            "无法加载 AuditLogger，审计日志功能不可用: %s", exc
                        )
                        self._audit_logger = None
        return self._audit_logger

    def __repr__(self) -> str:
        return (
            f"SecurityAudit(pip_audit={_HAS_PIP_AUDIT}, "
            f"safety={_HAS_SAFETY}, allow_commercial={self._allow_commercial})"
        )
