from __future__ import annotations
from typing import Any, Dict, List, Optional

def build_verification_report(
    *,
    verification_id: str,
    timestamp: str,
    artifact_label: str,
    verdict: str,
    failures: List[Dict[str, Any]],
    verified_scope: List[str],
    reasoning_trace: Dict[str, Any],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "verification_id": verification_id,
        "timestamp": timestamp,
        "artifact_verified": artifact_label,
        "executive_verdict": verdict,
        "verified_scope": verified_scope,
        "reasoning_trace": reasoning_trace,
        "not_verified": [
            "market demand realism",
            "competitive dynamics",
            "management intent or strategy",
            "external macroeconomic conditions",
        ],
        "attestation": (
            "This report reflects a deterministic verification of internal logical consistency. "
            "A FAIL indicates unresolved reasoning conflicts and should block external approval until corrected or explicitly justified. "
            "InsightGuard does not replace executive judgment; it makes reasoning risk explicit."
        ),
    }

    if verdict == "FAIL":
        report["failure_summary"] = failures[:3]
        report["instruction"] = "External approval should not proceed until the listed issues are resolved or explicitly justified."
    else:
        report["confidence_summary"] = (
            "No internal reasoning conflicts were detected that would justify blocking external approval."
        )

    return report
