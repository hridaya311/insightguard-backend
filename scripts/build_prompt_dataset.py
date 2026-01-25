import json, argparse, os
from datetime import datetime, timezone

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def make_prompt(ex):
    # Structured prompt: table + computed features
    table = ex["table"]
    movers = ex["features"]["top_movers"]
    anoms = ex["features"]["anomaly_candidates"]

    # Keep prompt compact but structured
    lines = []
    lines.append("You are an FP&A analyst. Write executive-ready variance commentary.")
    lines.append("Return STRICT JSON with keys: executive_summary, key_drivers, risks, recommendations.")
    lines.append("")
    lines.append("Variance table (category, actual, budget):")
    for r in table:
        lines.append(f"- {r['category']}: actual={r['actual']}, budget={r['budget']}")
    lines.append("")
    lines.append("Computed signals:")
    lines.append("Top movers:")
    for m in movers:
        vp = m.get("variance_pct")
        vp_s = f"{vp*100:.1f}%" if isinstance(vp, (int,float)) and vp is not None else "n/a"
        lines.append(f"- {m['category']}: variance={m['variance']} ({vp_s}), reason_hint={m.get('shock_reason')}")
    if anoms:
        lines.append("Anomaly candidates:")
        for a in anoms:
            vp = a.get("variance_pct")
            vp_s = f"{vp*100:.1f}%" if isinstance(vp, (int,float)) and vp is not None else "n/a"
            lines.append(f"- {a['category']}: variance={a['variance']} ({vp_s}), reason_hint={a.get('shock_reason')}")
    else:
        lines.append("Anomaly candidates: none")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Do not mention categories not present in the table.")
    lines.append("- Be concise and consistent.")
    lines.append("- Recommendations must be actionable.")
    return "\n".join(lines)

def make_target(ex):
    # Convert our existing gold summary into full JSON target (still deterministic, consistent)
    movers = ex["features"]["top_movers"][:3]
    anoms = ex["features"]["anomaly_candidates"][:3]

    def fmt_pct(vp):
        if vp is None: return None
        return round(vp, 4)

    key_drivers = []
    for m in movers:
        key_drivers.append({
            "category": m["category"],
            "actual": m["actual"],
            "budget": m["budget"],
            "variance": m["variance"],
            "variance_pct": fmt_pct(m.get("variance_pct")),
            "reason_hint": m.get("shock_reason")
        })

    risks = []
    if anoms:
        for a in anoms:
            risks.append({
                "category": a["category"],
                "variance": a["variance"],
                "variance_pct": fmt_pct(a.get("variance_pct")),
                "reason_hint": a.get("shock_reason"),
                "note": "Outsized variance; confirm driver and whether recurring."
            })
    else:
        risks.append({"note": "No anomaly candidates triggered by threshold."})

    recs = [
        "Validate flagged variances with source transactions and confirm one-time vs recurring drivers.",
        "Attach owner + 1–2 bullet root-cause notes for top variances to standardize commentary.",
        "Track these categories next period to confirm whether trend is improving or worsening."
    ]

    target = {
        "executive_summary": ex["gold"]["executive_summary"],
        "key_drivers": key_drivers,
        "risks": risks,
        "recommendations": recs
    }
    return json.dumps(target, ensure_ascii=False)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    rows = []
    for ex in iter_jsonl(args.in_path):
        rows.append({
            "id": ex["id"],
            "prompt": make_prompt(ex),
            "target": make_target(ex),
            "meta": {"style": ex["gold"].get("style"), "period": ex.get("period")}
        })

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    write_jsonl(args.out_path, rows)
    print("Wrote prompt dataset:", args.out_path, "rows=", len(rows))

if __name__ == "__main__":
    main()
