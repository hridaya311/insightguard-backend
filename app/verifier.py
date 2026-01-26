import json, re
from typing import Any, Dict, List, Tuple, Optional

REQUIRED_KEYS = ["executive_summary", "key_drivers", "risks", "recommendations"]
_num_re = re.compile(r"(?<![\w/])(\-?\d+(?:\.\d+)?)")

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
