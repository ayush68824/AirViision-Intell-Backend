from __future__ import annotations

import re
from typing import Any, Dict, List

from app.models.schemas import SafetyReport

_DIAGNOSIS_PATTERNS = [
    r"\b(?:we|i)\s+diagnos(?:e|ed)\b",
    r"\byour diagnosis is\b",
    r"\bprescrib(?:e|ed)\s+(?:you|medication|drug)\b",
    r"\btake\s+\d+\s*mg\b",
    r"\byou are (?:suffering|sick)\b",
    r"\bi (?:diagnose|conclude) that you\b",
]

_UNSUPPORTED_CERTAINTY = [
    r"\bguaranteed\b",
    r"\bdefinitely safe\b",
    r"\bzero risk\b",
    r"\bwill not affect\b",
]


def run_safety_agent(answer: str, evidence: Dict[str, Any]) -> SafetyReport:
    flags: List[str] = []
    blocked: List[str] = []
    text = answer or ""

    for pat in _DIAGNOSIS_PATTERNS:
        if re.search(pat, text, re.I):
            blocked.append(f"possible_diagnosis_language:{pat}")
            flags.append("health_overclaim")

    for pat in _UNSUPPORTED_CERTAINTY:
        if re.search(pat, text, re.I):
            blocked.append(f"unsupported_certainty:{pat}")
            flags.append("overconfident_claim")

    forecast = evidence.get("forecast") or {}
    freshness = (forecast.get("data_freshness") or {}) if isinstance(forecast, dict) else {}
    data_freshness_ok = bool(freshness.get("quality_ok", True)) if freshness else True
    if freshness and not data_freshness_ok:
        flags.append("stale_or_low_quality_aqi")

    # Numerical consistency: if answer mentions PM2.5 numbers, ensure they appear in evidence
    nums = set(re.findall(r"(\d{1,3}(?:\.\d+)?)\s*µg", text))
    evidence_blob = str(evidence)
    inconsistent = [n for n in nums if n not in evidence_blob]
    numerical_ok = len(inconsistent) == 0
    if not numerical_ok:
        flags.append("numeric_not_in_evidence")

    citations = evidence.get("citations") or []
    if "asthma" in text.lower() and not citations:
        flags.append("health_claim_without_citation")

    # Vision hard rule
    vision = evidence.get("vision") or {}
    if vision.get("analysis") and re.search(r"pm\s*2\.?5\s*[:=]?\s*\d+", str(vision), re.I):
        flags.append("vision_numeric_pm_forbidden")
        blocked.append("vision_precise_pm25")

    passed = not blocked and "health_overclaim" not in flags
    return SafetyReport(
        passed=passed,
        flags=flags,
        blocked_claims=blocked,
        data_freshness_ok=data_freshness_ok,
        numerical_consistency_ok=numerical_ok,
    )


def apply_safety_rewrite(answer: str, report: SafetyReport) -> str:
    if report.passed and not report.flags:
        return answer
    disclaimer = (
        "\n\n[Safety review] Some statements were softened. "
        "AirVision provides environmental decision support only — not medical diagnosis or treatment. "
        "Numeric estimates come from models/tools and include uncertainty."
    )
    cleaned = answer
    cleaned = re.sub(r"(?i)you have asthma", "if you have asthma (as you indicated)", cleaned)
    cleaned = re.sub(r"(?i)definitely safe", "comparatively lower risk based on available data", cleaned)
    cleaned = re.sub(r"(?i)guaranteed", "estimated", cleaned)
    if not report.passed or report.flags:
        cleaned += disclaimer
    return cleaned
