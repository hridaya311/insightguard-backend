import os, json, math, random, argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import boto3

# -------------------------
# Synthetic finance dataset
# -------------------------

DEFAULT_CATEGORIES = [
    "Sales", "COGS", "Marketing", "R&D", "G&A", "Cloud Hosting", "Payroll",
    "Professional Services", "Travel", "Facilities", "IT Software", "Support",
    "Logistics", "Customer Success", "Security", "Recruiting"
]

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def money(x: float) -> float:
    # keep to 2 decimals for realism
    return float(f"{x:.2f}")

def choose_categories(rng: random.Random, n: int) -> List[str]:
    cats = DEFAULT_CATEGORIES[:]
    rng.shuffle(cats)
    if n <= len(cats):
        return cats[:n]
    # if n larger, extend with BU tags
    extra = [f"Other_{i+1}" for i in range(n - len(cats))]
    return cats + extra

def generate_row(rng: random.Random, category: str, scale: float) -> Dict[str, Any]:
    # base budget: lognormal-ish
    base = np.random.lognormal(mean=math.log(scale), sigma=0.45)
    # add category shape
    if category in ("Sales",):
        base *= rng.uniform(2.0, 6.0)
    elif category in ("COGS", "Payroll"):
        base *= rng.uniform(1.2, 3.5)

    # actual = budget * (1 + noise + occasional shock)
    noise = rng.uniform(-0.08, 0.08)

    shock = 0.0
    shock_reason = None
    # 18% chance of meaningful shock
    if rng.random() < 0.18:
        shock = rng.choice([-1, 1]) * rng.uniform(0.12, 0.60)
        shock_reason = rng.choice([
            "timing_shift", "volume_change", "price_rate_change",
            "one_time_expense", "vendor_increase", "staffing_change",
            "project_overrun", "campaign_spend"
        ])

    actual = base * (1.0 + noise + shock)

    # forecast (optional realism)
    forecast = base * (1.0 + rng.uniform(-0.05, 0.05))

    # clamp non-negative for expense lines, allow Sales as revenue (positive)
    base = max(base, 0.0)
    actual = max(actual, 0.0)
    forecast = max(forecast, 0.0)

    row = {
        "category": category,
        "budget": money(base),
        "actual": money(actual),
        "forecast": money(forecast),
        "shock_reason": shock_reason
    }
    return row

def make_report(rng: random.Random, report_id: str, period: str, n_rows: int, scale: float) -> Dict[str, Any]:
    cats = choose_categories(rng, n_rows)
    rows = [generate_row(rng, c, scale) for c in cats]

    # compute numeric features for gold labeling
    feats = []
    for r in rows:
        var = r["actual"] - r["budget"]
        var_pct = (var / r["budget"]) if r["budget"] not in (0.0, None) else None
        feats.append({**r, "variance": money(var), "variance_pct": var_pct})

    # sort movers
    movers = sorted(feats, key=lambda x: abs(x["variance"]), reverse=True)
    top = movers[:5]

    # gold narrative (deterministic, consistent)
    drivers = []
    for t in top[:3]:
        sign = "+" if t["variance"] >= 0 else ""
        pct = f"{t['variance_pct']*100:.1f}%" if t["variance_pct"] is not None else "n/a"
        drivers.append(f"{t['category']} ({sign}{money(t['variance'])}, {pct})")

    anoms = [t for t in movers if (t["variance_pct"] is not None and abs(t["variance_pct"]) >= 0.20)]
    summary = "Budget vs Actual highlights: " + (", ".join(drivers) if drivers else "No major variances.") + "."
    if anoms:
        summary += f" {len(anoms)} item(s) flagged as anomaly candidates."

    # pack example
    example = {
        "id": report_id,
        "period": period,
        "table": [{ "category": r["category"], "actual": r["actual"], "budget": r["budget"], "forecast": r["forecast"] } for r in rows],
        "features": {
            "top_movers": [
                {
                    "category": t["category"],
                    "actual": t["actual"],
                    "budget": t["budget"],
                    "variance": t["variance"],
                    "variance_pct": t["variance_pct"],
                    "shock_reason": t["shock_reason"]
                } for t in movers[:5]
            ],
            "anomaly_candidates": [
                {
                    "category": t["category"],
                    "variance": t["variance"],
                    "variance_pct": t["variance_pct"],
                    "shock_reason": t["shock_reason"]
                } for t in anoms[:5]
            ]
        },
        "gold": {
            "executive_summary": summary,
            "style": "finance_variance_commentary_v1"
        }
    }
    return example

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def upload_to_s3(local_path: str, bucket: str, key: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read(), ContentType="application/jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--min_rows", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=25)
    ap.add_argument("--scale_min", type=float, default=5_000.0)
    ap.add_argument("--scale_max", type=float, default=250_000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data_out/synth_v1")
    ap.add_argument("--upload_s3", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    n_valid = int(args.n * args.valid_ratio)
    n_train = args.n - n_valid

    def gen_many(count: int, split: str) -> List[Dict[str, Any]]:
        items = []
        for i in range(count):
            rid = f"{split}_{i:06d}"
            period = f"2026-{rng.randint(1,12):02d}"
            n_rows = rng.randint(args.min_rows, args.max_rows)
            scale = rng.uniform(args.scale_min, args.scale_max)
            items.append(make_report(rng, rid, period, n_rows, scale))
        return items

    train = gen_many(n_train, "train")
    valid = gen_many(n_valid, "valid")

    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    print("Wrote:", train_path, "rows=", len(train))
    print("Wrote:", valid_path, "rows=", len(valid))

    if args.upload_s3:
        bucket = os.environ["S3_BUCKET"]
        region = os.environ["AWS_REGION"]
        prefix = f"datasets/synth_v1/{utc_ts()}"
        upload_to_s3(train_path, bucket, f"{prefix}/train.jsonl", region)
        upload_to_s3(valid_path, bucket, f"{prefix}/valid.jsonl", region)
        print("Uploaded to s3://%s/%s/" % (bucket, prefix))

if __name__ == "__main__":
    main()
