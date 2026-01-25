import os, json, math, random, argparse, gzip
from datetime import datetime, timezone
from typing import Optional
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

class MultipartS3Writer:
    """
    File-like object that buffers bytes and uploads to S3 via multipart upload.
    Works with gzip.GzipFile(fileobj=writer, mode='wb').
    """
    def __init__(self, s3, bucket: str, key: str, content_type: str, content_encoding: Optional[str],
                 part_size: int = 8 * 1024 * 1024):
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        self.part_size = part_size
        self.buf = bytearray()
        self.parts = []
        extra = {"ContentType": content_type}
        if content_encoding:
            extra["ContentEncoding"] = content_encoding

        resp = self.s3.create_multipart_upload(Bucket=bucket, Key=key, **extra)
        self.upload_id = resp["UploadId"]
        self.part_number = 1

    def write(self, b: bytes):
        if not b:
            return 0
        self.buf.extend(b)
        while len(self.buf) >= self.part_size:
            self._flush_part(self.part_size)
        return len(b)

    def _flush_part(self, n: int):
        chunk = bytes(self.buf[:n])
        del self.buf[:n]
        resp = self.s3.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            PartNumber=self.part_number,
            Body=chunk,
        )
        self.parts.append({"PartNumber": self.part_number, "ETag": resp["ETag"]})
        self.part_number += 1

    def close(self):
        # flush remaining buffer
        if len(self.buf) > 0:
            self._flush_part(len(self.buf))
        # complete upload
        self.s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            MultipartUpload={"Parts": self.parts},
        )

    def abort(self):
        try:
            self.s3.abort_multipart_upload(Bucket=self.bucket, Key=self.key, UploadId=self.upload_id)
        except Exception:
            pass

def generate_split_to_s3(s3, bucket: str, key: str, count: int, split: str,
                         rng: random.Random, args):
    writer = MultipartS3Writer(
        s3=s3,
        bucket=bucket,
        key=key,
        content_type="application/jsonl",
        content_encoding="gzip",
        part_size=args.part_size_mb * 1024 * 1024,
    )
    try:
        gz = gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=args.gzip_level)
        for i in range(count):
            rid = f"{split}_{i:07d}"
            period = f"2026-{rng.randint(1,12):02d}"
            n_rows = rng.randint(args.min_rows, args.max_rows)
            scale = rng.uniform(args.scale_min, args.scale_max)
            ex = make_report(rng, rid, period, n_rows, scale)
            line = (json.dumps(ex, ensure_ascii=False) + "\n").encode("utf-8")
            gz.write(line)

            if (i + 1) % args.log_every == 0:
                print(f"{split}: generated {i+1}/{count}")

        gz.close()
        writer.close()
        print(f"Uploaded split: s3://{bucket}/{key}")
    except Exception as e:
        writer.abort()
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--min_rows", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=25)
    ap.add_argument("--scale_min", type=float, default=5_000.0)
    ap.add_argument("--scale_max", type=float, default=250_000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="", help="S3 prefix to write under (optional). If blank, uses datasets/synth_v2_big/<timestamp>/")
    ap.add_argument("--gzip_level", type=int, default=6)
    ap.add_argument("--part_size_mb", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50000)
    args = ap.parse_args()

    bucket = os.environ["S3_BUCKET"]
    region = os.environ["AWS_REGION"]
    s3 = boto3.client("s3", region_name=region)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_valid = int(args.n * args.valid_ratio)
    n_train = args.n - n_valid

    prefix = args.prefix.strip() or f"datasets/synth_v2_big/{utc_ts()}"
    train_key = f"{prefix}/train.jsonl.gz"
    valid_key = f"{prefix}/valid.jsonl.gz"

    print("Target prefix:", f"s3://{bucket}/{prefix}/")
    print("Train:", n_train, "->", train_key)
    generate_split_to_s3(s3, bucket, train_key, n_train, "train", rng, args)

    print("Valid:", n_valid, "->", valid_key)
    generate_split_to_s3(s3, bucket, valid_key, n_valid, "valid", rng, args)

    manifest = {
        "dataset": "synth_v2_big",
        "created_at_utc": utc_ts(),
        "n_total": args.n,
        "n_train": n_train,
        "n_valid": n_valid,
        "files": [train_key, valid_key],
        "compression": "gzip",
        "generator": "scripts/generate_synth_dataset_to_s3.py",
        "notes": "Generated directly to S3 via multipart streaming (no local files)."
    }
    mkey = f"{prefix}/manifest.json"
    s3.put_object(Bucket=bucket, Key=mkey, Body=json.dumps(manifest, indent=2).encode("utf-8"),
                  ContentType="application/json")
    print("Manifest:", f"s3://{bucket}/{mkey}")

if __name__ == "__main__":
    main()
