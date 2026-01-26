from pathlib import Path
import re

MAIN = Path("app/main.py")
VERIFIER = Path("app/verifier.py")

# ---------------- verifier.py ----------------
VERIFIER.parent.mkdir(parents=True, exist_ok=True)
VERIFIER.write_text(
"""import json, re
from typing import Any, Dict, List, Tuple, Optional

REQUIRED_KEYS = ["executive_summary", "key_drivers", "risks", "recommendations"]
_num_re = re.compile(r"(?<![\\w/])(\\-?\\d+(?:\\.\\d+)?)")

def _parse_json_maybe(text_or_obj: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if isinstance(text_or_obj, dict):
        return text_or_obj, None
    if isinstance(text_or_obj, str):
        try:
            return json.loads(text_or_obj), None
        except Exception as e:
            return None, f"invalid_json:{e}"
    return None, "invalid_type"

def _collect_allowed_numbers(features: Dict[str, Any]) -> set:
    allowed = set()
    def add_num(x):
        if x is None or isinstance(x, bool):
            return
        try:
            if isinstance(x, (int, float)):
                fx = float(x)
                allowed.add(str(int(fx)) if fx.is_integer() else str(fx))
                allowed.add(f"{fx:.2f}".rstrip("0").rstrip("."))
            else:
                for m in _num_re.findall(str(x)):
                    allowed.add(m)
        except Exception:
            pass

    top_movers = features.get("top_movers", [])
    if isinstance(top_movers, list):
        for m in top_movers:
            if not isinstance(m, dict): 
                continue
            add_num(m.get("actual"))
            add_num(m.get("baseline"))
            add_num(m.get("budget"))
            add_num(m.get("variance"))
            add_num(m.get("variance_pct"))

    anoms = features.get("anomaly_candidates", [])
    if isinstance(anoms, list):
        for a in anoms:
            if not isinstance(a, dict):
                continue
            add_num(a.get("variance"))
            add_num(a.get("variance_pct"))

    return allowed

def verify_insights_against_facts(
    output_text_or_obj: Any,
    *,
    table_categories: List[str],
    features: Dict[str, Any],
    require_json: bool = True,
    require_keys: bool = True,
    forbid_new_categories: bool = True,
    forbid_unsupported_numbers: bool = True,
) -> Dict[str, Any]:
    report = {"pass": False, "errors": [], "warnings": []}

    out, err = _parse_json_maybe(output_text_or_obj)
    if err:
        if require_json:
            report["errors"].append(err)
            return report
        report["warnings"].append(err)
        out = {}

    if require_keys:
        missing = [k for k in REQUIRED_KEYS if k not in out]
        if missing:
            report["errors"].append(f"missing_keys:{missing}")

    allowed_cats = set(map(str, table_categories or []))
    used_cats = set()

    kd = out.get("key_drivers", [])
    if isinstance(kd, list):
        for item in kd:
            if isinstance(item, dict) and item.get("category"):
                used_cats.add(str(item["category"]))

    if forbid_new_categories and allowed_cats:
        extra = sorted([c for c in used_cats if c not in allowed_cats])
        if extra:
            report["errors"].append(f"extraneous_categories:{extra}")

    if forbid_unsupported_numbers and isinstance(features, dict) and features:
        allowed_nums = _collect_allowed_numbers(features)

        def scan_numbers(obj):
            found = []
            if isinstance(obj, str):
                found += _num_re.findall(obj)
            elif isinstance(obj, dict):
                for v in obj.values(): found += scan_numbers(v)
            elif isinstance(obj, list):
                for v in obj: found += scan_numbers(v)
            return found

        found_nums = scan_numbers(out)
        unsupported = [n for n in found_nums if n not in allowed_nums]

        def pct_equiv(n: str) -> bool:
            try:
                x = float(n)
                cand = x/100.0
                return (f"{cand:.4f}".rstrip("0").rstrip(".")) in allowed_nums or str(cand) in allowed_nums
            except Exception:
                return False

        unsupported2 = [n for n in unsupported if not pct_equiv(n)]
        if unsupported2:
            report["errors"].append(f"unsupported_numbers:{unsupported2[:20]}{'...(more)' if len(unsupported2)>20 else ''}")

    report["pass"] = (len(report["errors"]) == 0)
    report["extracted"] = out
    return report
""",
    encoding="utf-8"
)

t = MAIN.read_text(encoding="utf-8")

# ---------------- ensure import ----------------
if "from app.verifier import verify_insights_against_facts" not in t:
    t = re.sub(r"(from fastapi[^\n]*\n)", r"\1from app.verifier import verify_insights_against_facts\n", t, count=1)

# ---------------- add VerifyRequest model ----------------
if "class VerifyRequest(BaseModel):" not in t:
    ins = "\nclass VerifyRequest(BaseModel):\n    extracted_s3_key: str\n    features_s3_key: str\n    output_json: dict\n\n"
    m = re.search(r"(class InsightsRequest\(BaseModel\):[\s\S]*?\n)\n", t)
    if m:
        t = t[:m.end()] + ins + t[m.end():]
    else:
        t += ins

# ---------------- add /verify endpoint ----------------
if '@app.post("/verify")' not in t:
    t += """
@app.post("/verify")
def verify(req: VerifyRequest) -> Dict[str, Any]:
    bucket = get_bucket()
    s3 = get_s3()

    try:
        obj = s3.get_object(Bucket=bucket, Key=req.extracted_s3_key)
        extracted = json.loads(obj["Body"].read().decode("utf-8"))
        rows = extracted.get("rows") or extracted.get("table") or extracted.get("preview") or []
        cats = sorted(set(str(r.get("category")) for r in rows if isinstance(r, dict) and r.get("category")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load extracted: {e}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=req.features_s3_key)
        features_payload = json.loads(obj["Body"].read().decode("utf-8"))
        feats = features_payload.get("features") if isinstance(features_payload.get("features"), dict) else features_payload
        if not isinstance(feats, dict):
            feats = {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load features: {e}")

    return verify_insights_against_facts(
        req.output_json,
        table_categories=cats,
        features=feats,
        require_json=True,
        require_keys=True,
        forbid_new_categories=True,
        forbid_unsupported_numbers=True,
    )
"""

# ---------------- patch /insights verify+fallback ----------------
# We patch by inserting before a line that starts with exactly 4 spaces + insights_key =
m_block = re.search(
    r'@app\.post\("/insights"\)\s*\n(def\s+insights\([^\)]*\)\s*->\s*Dict\[str,\s*Any\]:\s*\n)([\s\S]*?)(?=\n@app\.post\(|\n@app\.get\(|\n@app\.put\(|\n@app\.delete\(|\nif __name__|\Z)',
    t
)
if not m_block:
    raise SystemExit("ERROR: Could not locate /insights block.")
block = t[m_block.start():m_block.end()]

if "BRUTAL_ACCURACY_LOCKED" not in block:
    needle = "\n    insights_key ="
    idx = block.find(needle)
    if idx == -1:
        raise SystemExit("ERROR: Could not find '\\n    insights_key =' inside /insights (expected 4-space indent).")

    ins = """
    # BRUTAL_ACCURACY_LOCKED: verify + fallback before persisting insights
    table_cats = []
    try:
        obj = s3.get_object(Bucket=bucket, Key=req.extracted_s3_key)
        extracted_payload = json.loads(obj["Body"].read().decode("utf-8"))
        rows = extracted_payload.get("rows") or extracted_payload.get("table") or extracted_payload.get("preview") or []
        if isinstance(rows, list):
            table_cats = sorted(set(str(r.get("category")) for r in rows if isinstance(r, dict) and r.get("category")))
    except Exception:
        table_cats = []

    features_for_verify = locals().get("feats")
    if not isinstance(features_for_verify, dict):
        features_for_verify = {}

    ver_report = verify_insights_against_facts(
        out,
        table_categories=table_cats,
        features=features_for_verify,
        require_json=True,
        require_keys=True,
        forbid_new_categories=True,
        forbid_unsupported_numbers=True,
    )
    out.setdefault("meta", {})
    if not ver_report.get("pass"):
        out["executive_summary"] = "Variance commentary generated with guardrails. Review top movers and anomaly candidates for verified details."
        out["meta"]["verification"] = ver_report
    else:
        out["meta"]["verification"] = {"pass": True}

"""
    block2 = block[:idx] + ins + block[idx:]
    t = t[:m_block.start()] + block2 + t[m_block.end():]

MAIN.write_text(t, encoding="utf-8")
print("OK: brutal accuracy patch applied (v2)")
