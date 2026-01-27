import argparse, gzip, io, json, os, random, time
from typing import Any, Dict, List, Tuple

import boto3
import requests

DEFAULT_API_URL = "http://127.0.0.1:8000/run-verified"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_s3_key", required=True)
    p.add_argument("--sample_n", type=int, default=200)
    p.add_argument("--scan_lines", type=int, default=5000)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--cleanup_uploads", action="store_true", help="Delete temp uploaded CSVs at end")
    return p.parse_args()

def _pct(a: int, b: int) -> float:
    return round(100.0 * a / max(1, b), 2)

def _pctl(sorted_vals: List[int], p: float):
    if not sorted_vals:
        return None
    i = int(round((len(sorted_vals) - 1) * p))
    i = max(0, min(len(sorted_vals) - 1, i))
    return sorted_vals[i]

def _read_gz_jsonl_from_s3(s3, bucket: str, key: str, max_lines: int) -> List[Dict[str, Any]]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    rows: List[Dict[str, Any]] = []
    with gzip.GzipFile(fileobj=obj["Body"]) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _infer_table_and_mapping(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    table = None
    if isinstance(row.get("table"), list):
        table = row["table"]
    elif isinstance(row.get("rows"), list):
        table = row["rows"]

    if not isinstance(table, list) or not table or not isinstance(table[0], dict):
        raise KeyError("Row does not contain a usable table/rows structure (expected list of dicts).")

    category_col = row.get("category_col") or "category"
    actual_col = row.get("actual_col") or "actual"
    baseline_col = row.get("baseline_col") or row.get("budget_col") or "budget"

    mapping = {"category_col": category_col, "actual_col": actual_col, "baseline_col": baseline_col}
    return table, mapping

def _table_to_csv_bytes(table: List[Dict[str, Any]], mapping: Dict[str, str]) -> bytes:
    ccol = mapping["category_col"]
    acol = mapping["actual_col"]
    bcol = mapping["baseline_col"]

    out = io.StringIO()
    out.write(f"{ccol},{acol},{bcol}\n")
    for r in table:
        cat = r.get(ccol, r.get("category"))
        act = r.get(acol, r.get("actual"))
        base = r.get(bcol, r.get("budget", r.get("baseline")))
        out.write(f"{cat},{act},{base}\n")
    return out.getvalue().encode("utf-8")

def _upload_csv_to_s3(s3, bucket: str, key: str, csv_bytes: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes, ContentType="text/csv")

def _preflight(api_url: str, timeout: int) -> None:
    # /run-verified might not have a GET, so check /health on same host
    base = api_url.split("/run-verified")[0].rstrip("/")
    health = base + "/health"
    try:
        r = requests.get(health, timeout=min(5, timeout))
        if r.status_code != 200:
            raise RuntimeError(f"Health check not OK: {health} status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        raise SystemExit(
            f"\nPRECHECK FAILED: cannot reach API.\n"
            f"API_URL={api_url}\n"
            f"Tried health={health}\n"
            f"Error={type(e).__name__}: {e}\n\n"
            f"Fix: start uvicorn, or set API_URL correctly.\n"
        )

def main():
    args = parse_args()
    api_url = os.getenv("API_URL", DEFAULT_API_URL)

    bucket = os.environ["S3_BUCKET"]
    region = os.environ["AWS_REGION"]
    s3 = boto3.client("s3", region_name=region)

    _preflight(api_url, args.timeout)

    run_id = time.strftime("%Y%m%dT%H%M%SZ")
    out_prefix = f"evals/run_verified/{run_id}/"
    report_key = out_prefix + "report.json"
    tmp_prefix = out_prefix + "tmp_uploads/"

    rows = _read_gz_jsonl_from_s3(s3, bucket, args.dataset_s3_key, args.scan_lines)
    if not rows:
        raise SystemExit("No rows loaded from dataset. Increase --scan_lines or verify dataset key.")

    sample = random.sample(rows, min(args.sample_n, len(rows)))

    stats = {
        "total_attempted": 0,   # attempts to call API
        "http_200": 0,
        "verification_pass": 0,
        "verification_fail": 0,
        "http_non_200": 0,
        "latencies": [],
        "non_200_buckets": {},
        "schema_fail": 0,
        "request_exception_examples": [],
    }

    uploaded_keys: List[str] = []

    for idx, row in enumerate(sample):
        try:
            if "s3_key" in row and isinstance(row["s3_key"], str) and row["s3_key"]:
                s3_key = row["s3_key"]
                mapping = {
                    "category_col": row.get("category_col", "category"),
                    "actual_col": row.get("actual_col", "actual"),
                    "baseline_col": row.get("baseline_col", "budget"),
                }
            else:
                table, mapping = _infer_table_and_mapping(row)
                csv_bytes = _table_to_csv_bytes(table, mapping)
                s3_key = f"{tmp_prefix}ex_{idx:06d}.csv"
                _upload_csv_to_s3(s3, bucket, s3_key, csv_bytes)
                uploaded_keys.append(s3_key)
        except Exception as e:
            stats["schema_fail"] += 1
            continue

        payload = {
            "s3_key": s3_key,
            "category_col": mapping["category_col"],
            "actual_col": mapping["actual_col"],
            "baseline_col": mapping["baseline_col"],
            "top_n": 5,
            "max_drivers": 5,
            "fail_on_unverified": True,
        }

        stats["total_attempted"] += 1
        t0 = time.time()
        try:
            r = requests.post(api_url, json=payload, timeout=args.timeout)
        except Exception as e:
            stats["http_non_200"] += 1
            stats["non_200_buckets"]["request_exception"] = stats["non_200_buckets"].get("request_exception", 0) + 1
            if len(stats["request_exception_examples"]) < 5:
                stats["request_exception_examples"].append(f"{type(e).__name__}: {e}")
            continue
        dt = int((time.time() - t0) * 1000)
        stats["latencies"].append(dt)

        if r.status_code == 200:
            stats["http_200"] += 1
            meta = (r.json().get("meta") or {})
            if (meta.get("verification") or {}).get("pass") is True:
                stats["verification_pass"] += 1
            else:
                stats["verification_fail"] += 1
        else:
            stats["http_non_200"] += 1
            stats["non_200_buckets"][str(r.status_code)] = stats["non_200_buckets"].get(str(r.status_code), 0) + 1

    lat = sorted(stats["latencies"])

    summary = {
        "run_id": run_id,
        "api_url": api_url,
        "dataset_s3_key": args.dataset_s3_key,
        "sample_n_requested": args.sample_n,
        "scan_lines": args.scan_lines,
        "rows_loaded": len(rows),

        "total_attempted": stats["total_attempted"],
        "http_200": stats["http_200"],
        "http_200_pct": _pct(stats["http_200"], stats["total_attempted"]),
        "verification_pass": stats["verification_pass"],
        "verification_pass_pct_of_total": _pct(stats["verification_pass"], stats["total_attempted"]),
        "verification_fail": stats["verification_fail"],

        "http_non_200": stats["http_non_200"],
        "schema_fail": stats["schema_fail"],
        "latency_ms_p50": _pctl(lat, 0.50),
        "latency_ms_p90": _pctl(lat, 0.90),
        "latency_ms_p99": _pctl(lat, 0.99),
        "non_200_buckets": stats["non_200_buckets"],
        "request_exception_examples": stats["request_exception_examples"],

        "tmp_uploads_prefix": tmp_prefix,
        "tmp_uploads_count": len(uploaded_keys),
        "cleanup_uploads": bool(args.cleanup_uploads),
    }

    s3.put_object(
        Bucket=bucket,
        Key=report_key,
        Body=json.dumps({"summary": summary}, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    if args.cleanup_uploads and uploaded_keys:
        for i in range(0, len(uploaded_keys), 1000):
            chunk = uploaded_keys[i:i+1000]
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )

    print("EVAL COMPLETE")
    print("REPORT_KEY=", report_key)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
