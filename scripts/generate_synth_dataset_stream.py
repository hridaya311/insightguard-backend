import os, json, math, random, argparse, gzip
from datetime import datetime, timezone
import numpy as np
import boto3

DEFAULT_CATEGORIES = [
    "Sales", "COGS", "Marketing", "R&D", "G&A", "Cloud Hosting", "Payroll",
    "Professional Services", "Travel", "Facilities", "IT Software", "Support",
    "Logistics", "Customer Success", "Security", "Recruiting"
]

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def money(x: float) -> float:
    return float(f"{x:.2f}")

def choose_categories(rng: random.Random, n: int):
    cats = DEFAULT_CATEGORIES[:]
    rng.shuffle(cats)
    if n <= len(cats):
        return cats[:n]
    extra = [f"Other_{i+1}" for i in range(n - len(cats))]
    return cats + extra

def generate_row(rng: random.Random, category: str, scale: float):
    base = np.random.lognormal(mean=math.log(scale), sigma=0.45)
    if category == "Sales":
        base *= rng.uniform(2.0, 6.0)
    elif category in ("COGS", "Payroll"):
        base *= rng.uniform(1.2, 3.5)

    noise = rng.uniform(-0.08, 0.08)
    shock = 0.0
    shock_reason = None
    if rng.random() < 0.18:
        shock = rng.choice([-1, 1]) * rng.uniform(0.12, 0.60)
        shock_reason = rng.choice([
            "timing_shift", "volume_change", "price_rate_change",
            "one_time_expense", "vendor_increase", "staffing_change",
            "project_overrun", "campaign_spend"
        ])

    actual = base * (1.0 + noise + shock)
    forecast = base * (1.0 + rng.uniform(-0.05, 0.05))

    base = max(base, 0.0)
    actual = max(actual, 0.0)
    forecast = max(forecast, 0.0)

    return {
        "category": category,
        "budget": money(base),
        "actual": money(actual),
        "forecast": money(forecast),
        "shock_reason": shock_reason
    }

def make_report(rng: random.Random, report_id: str, period: str, n_rows: int, scale: float):
    cats = choose_categories(rng, n_rows)
    rows = [generate_row(rng, c, scale) for c in cats]

    feats = []
    for r in rows:
        var = r["actual"] - r["budget"]
        var_pct = (var / r["budget"]) if r["budget"] not in (0.0, None) else None
        feats.append({**r, "variance": money(var), "variance_pct": var_pct})

    movers = sorted(feats, key=lambda x: abs(x["variance"]), reverse=True)
    top = movers[:5]

    drivers = []
    for t in top[:3]:
        sign = "+" if t["variance"] >= 0 else ""
        pct = f"{t['variance_pct']*100:.1f}%" if t["variance_pct"] is not None else "n/a"
        drivers.append(f"{t['category']} ({sign}{money(t['variance'])}, {pct})")

    anoms = [t for t in movers if (t["variance_pct"] is not None and abs(t["variance_pct"]) >= 0.20)]
    summary = "Budget vs Actual highlights: " + (", ".join(drivers) if drivers else "No major variances.") + "."
    if anoms:
        summary += f" {len(anoms)} item(s) flagged as anomaly candidates."

    return {
        "id": report_id,
        "period": period,
        "table": [{"category": r["category"], "actual": r["actual"], "budget": r["budget"], "forecast": r["forecast"]} for r in rows],
        "features": {
            "top_movers": [{
                "category": t["category"],
                "actual": t["actual"],
                "budget": t["budget"],
                "variance": t["variance"],
                "variance_pct": t["variance_pct"],
                "shock_reason": t["shock_reason"]
            } for t in movers[:5]],
            "anomaly_candidates": [{
                "category": t["category"],
                "variance": t["variance"],
                "variance_pct": t["variance_pct"],
                "shock_reason": t["shock_reason"]
            } for t in anoms[:5]],
        },
        "gold": {"executive_summary": summary, "style": "finance_variance_commentary_v1"}
    }

def open_writer(path: str, gzip_level: int):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=gzip_level)
    return open(path, "w", encoding="utf-8")

def s3_upload(local_path: str, bucket: str, key: str, region: str):
    s3 = boto3.client("s3", region_name=region)
    extra = {"ContentType": "application/jsonl"}
    if local_path.endswith(".gz"):
        extra["ContentEncoding"] = "gzip"
        extra["ContentType"] = "application/gzip"
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra)

def gen_split(out_path: str, count: int, split: str, rng: random.Random, args):
    with open_writer(out_path, args.gzip_level) as f:
        for i in range(count):
            rid = f"{split}_{i:07d}"
            period = f"2026-{rng.randint(1,12):02d}"
            n_rows = rng.randint(args.min_rows, args.max_rows)
            scale = rng.uniform(args.scale_min, args.scale_max)
            ex = make_report(rng, rid, period, n_rows, scale)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if (i+1) % args.log_every == 0:
                print(f"{split}: wrote {i+1}/{count}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--min_rows", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=25)
    ap.add_argument("--scale_min", type=float, default=5_000.0)
    ap.add_argument("--scale_max", type=float, default=250_000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data_out/synth_v2_big")
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--gzip_level", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=20000)
    ap.add_argument("--upload_s3", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    n_valid = int(args.n * args.valid_ratio)
    n_train = args.n - n_valid

    ext = ".jsonl.gz" if args.gzip else ".jsonl"
    train_path = os.path.join(args.out_dir, f"train{ext}")
    valid_path = os.path.join(args.out_dir, f"valid{ext}")

    print("Generating train:", n_train, "->", train_path)
    gen_split(train_path, n_train, "train", rng, args)
    print("Generating valid:", n_valid, "->", valid_path)
    gen_split(valid_path, n_valid, "valid", rng, args)

    if args.upload_s3:
        bucket = os.environ["S3_BUCKET"]
        region = os.environ["AWS_REGION"]
        prefix = f"datasets/synth_v2_big/{utc_ts()}"
        s3_upload(train_path, bucket, f"{prefix}/{os.path.basename(train_path)}", region)
        s3_upload(valid_path, bucket, f"{prefix}/{os.path.basename(valid_path)}", region)
        print(f"Uploaded to s3://{bucket}/{prefix}/")

if __name__ == "__main__":
    main()
