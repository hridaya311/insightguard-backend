from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

from app.product_contracts import RunVerifiedRequest, Failure

REQUIRED_METRICS = ["revenue", "gross_margin", "operating_margin"]

@dataclass
class VerificationResult:
    verdict: str  # "PASS" | "FAIL"
    failures: List[Failure]
    verified_scope: List[str]
    reasoning_trace: Dict[str, Any]

def _get_metric(block_metrics: Dict[str, float], key: str):
    return block_metrics.get(key, None)

def _pct_change(a: float, b: float):
    # change from a -> b, percent of a
    if a == 0:
        return None
    return (b - a) / a

def run_product_verification(req: RunVerifiedRequest) -> VerificationResult:
    """
    Deterministic verification for v1 product promise:
    - Fail closed.
    - No scoring.
    - Explicit reasons.
    """
    failures: List[Failure] = []
    scope = [
        "assumption-outcome consistency",
        "period-over-period reconciliation",
        "narrative-numeric alignment",
        "baseline constraint adherence",
        "verification integrity",
    ]

    # -------------------------
    # Check #7: Verification Integrity Failure (hard gate)
    # -------------------------
    for m in REQUIRED_METRICS:
        if m not in req.baseline.metrics or m not in req.projection.metrics:
            failures.append(Failure(
                code="VERIFICATION_INTEGRITY_FAILURE",
                message=f"Missing required metric '{m}' in baseline or projection."
            ))

    if req.baseline.period == req.projection.period:
        failures.append(Failure(
            code="VERIFICATION_INTEGRITY_FAILURE",
            message="Baseline period and projection period must be different."
        ))

    if not req.assumptions.narrative or not req.assumptions.narrative.strip():
        failures.append(Failure(
            code="VERIFICATION_INTEGRITY_FAILURE",
            message="Missing assumptions narrative. External approval cannot proceed without an explicit reasoning bridge."
        ))

    if failures:
        return VerificationResult(
            verdict="FAIL",
            failures=failures,
            verified_scope=scope,
            reasoning_trace={"note": "Stopped at integrity gate."}
        )

    narrative = req.assumptions.narrative.lower()
    cost_structure = req.assumptions.cost_structure or "mixed"

    # -------------------------
    # Derived deltas (used by multiple checks)
    # -------------------------
    r0 = float(_get_metric(req.baseline.metrics, "revenue"))
    r1 = float(_get_metric(req.projection.metrics, "revenue"))
    gm0 = float(_get_metric(req.baseline.metrics, "gross_margin"))
    gm1 = float(_get_metric(req.projection.metrics, "gross_margin"))
    om0 = float(_get_metric(req.baseline.metrics, "operating_margin"))
    om1 = float(_get_metric(req.projection.metrics, "operating_margin"))

    rev_growth = _pct_change(r0, r1)
    gm_delta = gm1 - gm0
    om_delta = om1 - om0

    # -------------------------
    # Check #2: Unreconciled Period-over-Period Deltas (v1 deterministic)
    # Rule: if deltas are material, narrative must explicitly acknowledge relevant driver area.
    # -------------------------
    REV_MATERIAL = 0.05      # 5%
    MARGIN_MATERIAL = 0.005  # 50 bps

    revenue_terms = ["revenue", "sales", "pricing", "volume", "units", "bookings", "demand"]
    gross_margin_terms = ["gross margin", "gm", "cogs", "pricing", "mix", "discount", "cost of goods"]
    op_margin_terms = ["operating margin", "margin", "opex", "cost", "expenses", "investment", "efficiency", "headcount"]

    if rev_growth is not None and abs(rev_growth) >= REV_MATERIAL:
        if not any(t in narrative for t in revenue_terms):
            failures.append(Failure(
                code="UNRECONCILED_DELTAS",
                message="Revenue changed materially versus baseline, but the assumptions narrative does not explicitly explain the revenue drivers."
            ))

    if abs(gm_delta) >= MARGIN_MATERIAL:
        if not any(t in narrative for t in gross_margin_terms):
            failures.append(Failure(
                code="UNRECONCILED_DELTAS",
                message="Gross margin changed materially versus baseline, but the assumptions narrative does not explicitly explain the gross margin drivers."
            ))

    if abs(om_delta) >= MARGIN_MATERIAL:
        if not any(t in narrative for t in op_margin_terms):
            failures.append(Failure(
                code="UNRECONCILED_DELTAS",
                message="Operating margin changed materially versus baseline, but the assumptions narrative does not explicitly explain the operating drivers (costs/opex/investments/efficiency)."
            ))

    # -------------------------
    # Check #1: Logical inconsistency between assumptions and outcomes (existing v1 rule)
    # -------------------------
    MATERIAL_REV_GROWTH = 0.10  # 10%
    FLAT_MARGIN_EPS = 0.002     # 20 bps

    if rev_growth is not None and rev_growth >= MATERIAL_REV_GROWTH and cost_structure in ("fixed", "mixed"):
        if abs(om_delta) <= FLAT_MARGIN_EPS:
            offset_terms = ["offset", "increased costs", "pricing pressure", "mix shift", "one-time", "investment"]
            if not any(t in narrative for t in offset_terms):
                failures.append(Failure(
                    code="LOGICAL_INCONSISTENCY",
                    message="Revenue growth under stated cost structure implies operating leverage, but operating margin remains flat without an explicit offset in the assumptions narrative."
                ))

    verdict = "FAIL" if failures else "PASS"
    trace = {
        "baseline": {"period": req.baseline.period, "metrics": req.baseline.metrics},
        "projection": {"period": req.projection.period, "metrics": req.projection.metrics},
        "assumptions": {
            "cost_structure": cost_structure,
            "revenue_drivers": req.assumptions.revenue_drivers,
            "efficiency_initiatives": req.assumptions.efficiency_initiatives,
            "incremental_investments": req.assumptions.incremental_investments,
        },
        "derived": {
            "rev_growth": rev_growth,
            "gross_margin_delta": gm_delta,
            "operating_margin_delta": om_delta,
        },
        "notes": "Deterministic checks only. PASS requires all checks to clear."
    }

    return VerificationResult(
        verdict=verdict,
        failures=failures,
        verified_scope=scope,
        reasoning_trace=trace
    )
