from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class FinancialBlock(BaseModel):
    period: str = Field(..., min_length=1)
    metrics: Dict[str, float] = Field(...)

class AssumptionsBlock(BaseModel):
    cost_structure: Optional[Literal["fixed", "variable", "mixed"]] = None
    revenue_drivers: Optional[List[str]] = None
    efficiency_initiatives: Optional[List[str]] = None
    incremental_investments: Optional[List[Dict[str, Any]]] = None
    narrative: str = Field(..., min_length=1)

class RunVerifiedRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    baseline: FinancialBlock
    projection: FinancialBlock
    assumptions: AssumptionsBlock

class Failure(BaseModel):
    code: str
    message: str

class RunVerifiedResponse(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    meta: Dict[str, Any]
    failures: List[Failure] = Field(default_factory=list)
    instruction: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
