#!/usr/bin/env python3
import argparse, os, json, time, gzip, io, csv, random, urllib.request, urllib.error
from datetime import datetime, timezone
import boto3

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def s3_client(region: str):
    return boto3.client("s3", region_name=region)

def read_gz_jsonl_sample(s3, bucket: str, key: str, scan_lines: int, sample_n: int, seed: int = 42):
    """
    Cost-conscious sampling: reads only the FIRST `scan_lines` lines of the gz JSONL,
    then randomly samples `sample_n` from those.
    (If you want a truer global sample later, we can do S3 Select or multi-range sampling.)
    """
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    gz = gzip.GzipFile(fileobj=body)

    items = []
    for i, raw in enumerate(gz):
        if i >= scan_lines:
            break
        raw = raw.decode("utf-8").strip()
        if not raw:
            continue
        try:
            items.append(json.loads(raw))
        except Exception:
            # skip malformed lines
            continue

    if not items:
        raise RuntimeError("No JSON rows found in scanned portion of dataset.")

    rng = random.Random(seed)
    if sample_n >= len(items):
        return items
    return rng.sample(items, sample_n)

def infer_rows(example: dict):
    """
    Tries to find a list-of-dicts table in common keys.
    """
    for k in ["table", "rows", "data", "variance_table"]:
        v = example.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    # Some generators store a dict with 'rows' key
    if isinstance(example.get("extracted"), dict):
        v = example["extracted"].get("rows")
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise RuntimeError(f"Could not infer table rows from example keys: {list(example.keys())[:20]}")

def infer_mapping(rows: list[dict]):
    """
    We assume finance variance tables typically have:
      - category/name/line_item
      - actual
      - budget/baseline/forecast
    """
    cols = set()
    for r in rows:
        cols.update(r.keys())

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    category_col = pick(["category", "Category", "line_item", "lineItem", "account", "Account", "name", "Name"])
    actual_col = pick(["actual", "Actual", "value", "Value"])
    baseline_col = pick(["budget", "Budget", "baseline", "Baseline", "forecast", "Forecast"])

    if not (category_col and actual_col and baseline_col):
        raise RuntimeError(f"Could not infer mapping. Found cols={sorted(cols)[:30]}")

    return category_col, actual_col, baseline_col

def rows_to_csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")

def http_post_json(url: str, payload: dict, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:
        return 0, json.dumps({"error": str(e)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_s3_key", required=True, help="S3 key of gz JSONL dataset (e.g. datasets/.../train.jsonl.gz)")
    ap.add_argument("--sample_n", type=int, default=200, help="How many examples to test")
    ap.add_argument("--scan_lines", type=int, default=5000, help="How many lines to scan from start of gz before sampling")
    ap.add_argument("--api_base", default="http://127.0.0.1:8000", help="FastAPI base URL")
    ap.add_argument("--out_prefix", default=None, help="S3 prefix for eval outputs (default: evals/run_verified/<ts>/)")
    ap.add_argument("--upload_prefix", default=None, help="S3 prefix for temp csv uploads (default: eval_tmp/<ts>/)")
    ap.add_argument("--cleanup_uploads", action="store_true", help="Delete temp uploaded CSVs after eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bucket = os.environ["S3_BUCKET"]
    region = os.environ["AWS_REGION"]
    s3 = s3_client(region)

    run_id = utc_ts()
    out_prefix = args.out_prefix or f"evals/run_verified/{run_id}/"
    upload_prefix = args.upload_prefix or f"eval_tmp/{run_id}/"

    dataset_key = args.dataset_s3_key

    print("Bucket:", bucket)
    print("Dataset:", dataset_key)
    print("Sample N:", args.sample_n, "Scan lines:", args.scan_lines)
    print("API:", args.api_base)
    print("Out prefix:", out_prefix)
    print("Upload prefix:", upload_prefix)
    print("Cleanup:", args.cleanup_uploads)

    # 1) sample examples (reads only first scan_lines)
    t0 = time.time()
    examples = read_gz_jsonl_sample(
        s3=s3, bucket=bucket, key=dataset_key,
        scan_lines=args.scan_lines, sample_n=args.sample_n, seed=args.seed
    )
    print(f"Sampled {len(examples)} examples in {time.time()-t0:.2f}s")

    results = []
    uploaded_keys = []

    # 2) run eval
    for idx, ex in enumerate(examples, start=1):
        ex_id = ex.get("id") or ex.get("example_id") or f"ex_{idx:06d}"
        started = time.time()
        item = {
            "idx": idx,
            "id": ex_id,
            "upload_s3_key": None,
            "http_status": None,
            "verification": None,
            "error": None,
            "latency_ms": None,
        }

        try:
            rows = infer_rows(ex)
            category_col, actual_col, baseline_col = infer_mapping(rows)

            # stable field order: category, actual, baseline, then rest sorted
            cols = list(rows[0].keys())
            base = [category_col, actual_col, baseline_col]
            rest = [c for c in cols if c not in base]
            fieldnames = base + sorted(rest)

            csv_bytes = rows_to_csv_bytes(rows, fieldnames)
            upload_key = f"{upload_prefix}{ex_id}.csv"
            s3.put_object(Bucket=bucket, Key=upload_key, Body=csv_bytes, ContentType="text/csv")
            uploaded_keys.append(upload_key)
            item["upload_s3_key"] = upload_key

            payload = {
                "s3_key": upload_key,
                "category_col": category_col,
                "actual_col": actual_col,
                "baseline_col": baseline_col,
                "top_n": 5,
                "max_drivers": 5,
                "fail_on_unverified": True,
            }
            status, body = http_post_json(args.api_base.rstrip("/") + "/run-verified", payload, timeout=120)
            item["http_status"] = status

            if status == 200:
                d = json.loads(body)
                item["verification"] = (d.get("meta") or {}).get("verification")
            else:
                # keep a compact error
                item["error"] = body[:800]

        except Exception as e:
            item["error"] = f"exception:{type(e).__name__}:{str(e)[:300]}"
            item["http_status"] = item["http_status"] or 0

        item["latency_ms"] = int((time.time() - started) * 1000)
        results.append(item)

        if idx % 25 == 0 or idx == len(examples):
            ok = sum(1 for r in results if r.get("http_status") == 200)
            print(f"Progress {idx}/{len(examples)}  ok={ok}  last_latency_ms={item['latency_ms']}")

    # 3) summarize
    total = len(results)
    ok = [r for r in results if r.get("http_status") == 200]
    pass_ver = [r for r in ok if (r.get("verification") or {}).get("pass") is True]
    fail_ver = [r for r in ok if (r.get("verification") or {}).get("pass") is False]
    non200 = [r for r in results if r.get("http_status") != 200]

    def pct(a, b): 
        return round((a / b) * 100, 2) if b else 0.0

    latency = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    latency.sort()
    def pctl(p):
        if not latency:
            return None
        k = int(round((p/100) * (len(latency)-1)))
        return latency[k]

    summary = {
        "run_id": run_id,
        "dataset_s3_key": dataset_key,
        "sample_n": args.sample_n,
        "scan_lines": args.scan_lines,
        "total": total,
        "http_200": len(ok),
        "http_200_pct": pct(len(ok), total),
        "verification_pass": len(pass_ver),
        "verification_pass_pct_of_total": pct(len(pass_ver), total),
        "verification_fail": len(fail_ver),
        "http_non_200": len(non200),
        "latency_ms_p50": pctl(50),
        "latency_ms_p90": pctl(90),
        "latency_ms_p99": pctl(99),
    }

    # error buckets
    buckets = {}
    for r in non200:
        code = str(r.get("http_status"))
        buckets[code] = buckets.get(code, 0) + 1
    summary["non_200_buckets"] = buckets

    report = {
        "summary": summary,
        "results": results[:2000],  # safety: don't blow up report size
        "notes": [
            "Sampling reads only the first scan_lines of the gz file, then random-samples from that subset.",
            "If you want unbiased sampling across the full 1M file, we can add S3 Select or multi-prefix shard sampling next."
        ],
    }

    # 4) upload report to S3
    report_key = f"{out_prefix}report.json"
    s3.put_object(
        Bucket=bucket,
        Key=report_key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nReport written to s3://%s/%s" % (bucket, report_key))

    # 5) cleanup temp uploads if requested
    if args.cleanup_uploads and uploaded_keys:
        # delete in chunks of 1000
        for i in range(0, len(uploaded_keys), 1000):
            chunk = uploaded_keys[i:i+1000]
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]}
            )
        print(f"Cleaned up {len(uploaded_keys)} uploaded temp CSVs under {upload_prefix}")

if __name__ == "__main__":
    main()
