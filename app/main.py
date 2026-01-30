import os
from botocore.exceptions import ClientError
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import boto3
from botocore.config import Config
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.verifier import verify_insights_against_facts
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "InsightGuard backend running", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

def get_bucket() -> str:
    b = os.getenv("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET env var not set")
    return b

def get_region() -> str:
    r = os.getenv("AWS_REGION")
    if not r:
        raise RuntimeError("AWS_REGION env var not set")
    return r

def get_s3():
    return boto3.client("s3", region_name=get_region(), config=Config(signature_version="s3v4"))
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def safe_name(name: str) -> str:
    return (name or "upload").replace("/", "_")

def _job_key(job_id: str) -> str:
    return f"jobs/{job_id}.json"

def _write_job(s3, bucket: str, job_id: str, job: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=_job_key(job_id),
        Body=json.dumps(job).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

def _read_job(s3, bucket: str, job_id: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=_job_key(job_id))
    return json.loads(obj["Body"].read().decode("utf-8"))


def _s3_read_json(s3, bucket: str, key: str):
    """
    Read JSON from S3 and normalize common shapes.
    Some earlier extract steps may store a list instead of an envelope dict.
    Returns dict or list depending on content, but attempts to wrap lists into a dict when it looks like tabular data.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    # Normalize: if data is a list, try to wrap/unwrap to expected dict shapes
    if isinstance(data, list):
        # Case A: list with a single dict envelope
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        # Case B: list of dict rows (tabular)
        if len(data) > 0 and all(isinstance(x, dict) for x in data[:5]):
            return {"preview": data}
        # Otherwise return as-is
        return data

    return data


def _s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def verify_insights_v1(*, extracted: dict, features: dict, insights: dict) -> dict:
    """
    Deterministic verification gate (bulletproof):
    - Never throws.
    - Validates types first.
    - Checks required keys in insights (if dict).
    - Checks driver categories exist in extracted categories (best effort).
    - Soft check: exec summary mentions at least one mover.
    """
    # ---- type guards ----
    if not isinstance(extracted, dict):
        return {"pass": False, "reasons": [f"bad_type:extracted:{type(extracted).__name__}"]}
    if not isinstance(features, dict):
        return {"pass": False, "reasons": [f"bad_type:features:{type(features).__name__}"]}
    if not isinstance(insights, dict):
        return {"pass": False, "reasons": [f"bad_type:insights:{type(insights).__name__}"]}

    reasons = []
    passed = True

    # required keys
    for k in ["executive_summary", "key_drivers", "risks", "recommendations"]:
        if k not in insights:
            passed = False
            reasons.append(f"missing_key:{k}")

    # extracted categories from preview (best effort)
    cats = set()
    preview = extracted.get("preview") or []
    if isinstance(preview, list):
        for row in preview:
            if isinstance(row, dict) and "category" in row:
                cats.add(str(row["category"]))

    # movers
    movers = []
    tm = features.get("top_movers") or []
    if isinstance(tm, list):
        for item in tm[:5]:
            if isinstance(item, dict) and item.get("category") is not None:
                movers.append(str(item.get("category")))

    # parse key_drivers into categories (supports list[str] OR list[dict])
    kd = insights.get("key_drivers") or []
    kd_cats = []
    if isinstance(kd, list):
        for item in kd[:10]:
            if isinstance(item, dict) and item.get("category") is not None:
                kd_cats.append(str(item.get("category")))
            elif isinstance(item, str):
                s = item.strip().lstrip("-").strip()
                # common: "Sales: ..." -> "Sales"
                cat = s.split(":")[0].strip()
                if cat:
                    kd_cats.append(cat)

    # hard check (only if we have extracted cats)
    if cats and kd_cats:
        bad = [c for c in kd_cats if c not in cats]
        if bad:
            passed = False
            reasons.append("key_driver_category_not_in_extracted:" + ",".join(bad[:5]))

    # soft check: exec summary mentions at least one mover (optional)
    exec_sum = insights.get("executive_summary")
    exec_sum = str(exec_sum) if exec_sum is not None else ""
    if movers and exec_sum and not any(c in exec_sum for c in movers[:3]):
        reasons.append("exec_summary_missing_top_mover_names")

    return {"pass": passed, "reasons": reasons}


def presign_download(key: str, expires_in_seconds: int = 900) -> dict:
    """
    Presigned GET for any object in this bucket (restricted by IAM policy).
    Use this for UI downloads without proxying through the API.
    """
    bucket = get_bucket()
    s3 = get_s3()
    if not _s3_exists(s3, bucket, key):
        raise HTTPException(status_code=404, detail=f"No such key: {key}")
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in_seconds,
    )
    return {"s3_bucket": bucket, "s3_key": key, "download_url": url, "expires_in_seconds": expires_in_seconds}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    key = f"uploads/{utc_ts()}_{safe_name(file.filename)}"

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    return {"s3_bucket": bucket, "s3_key": key, "bytes": len(data)}

class ExtractRequest(BaseModel):
    s3_key: str
    sheet_name: Optional[str] = None  # for xlsx: None -> first sheet

def read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def read_xlsx_from_bytes(data: bytes, sheet_name: Optional[str]) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)

def df_to_json_records(df: pd.DataFrame) -> str:
    clean = df.where(pd.notnull(df), None)
    return clean.to_json(orient="records")

@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    key = req.s3_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="s3_key is required")

    # Download
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e}")

    lower = key.lower()
    try:
        if lower.endswith(".csv"):
            df = read_csv_from_bytes(data)
            source_type = "csv"
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            df = read_xlsx_from_bytes(data, req.sheet_name)
            source_type = "xlsx"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for extract. Use .csv or .xlsx for now.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    if df is None:
        raise HTTPException(status_code=400, detail="No data found in file")

    extracted_key = f"extracted/{utc_ts()}_{safe_name(os.path.basename(key))}.json"

    try:
        json_body = df_to_json_records(df)
        s3.put_object(
            Bucket=bucket,
            Key=extracted_key,
            Body=json_body.encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write extracted JSON to S3: {e}")

    preview_rows = min(5, len(df))
    preview = df.head(preview_rows).where(pd.notnull(df.head(preview_rows)), None).to_dict(orient="records")

    return {
        "source_type": source_type,
        "source_s3_key": key,
        "extracted_s3_key": extracted_key,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns.tolist()],
        "preview": preview,
    }

class FeaturesRequest(BaseModel):
    extracted_s3_key: str
    category_col: str = "category"
    actual_col: str = "actual"
    baseline_col: str = "budget"  # could be forecast, prior_period, etc.
    top_n: int = 5

def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

@app.post("/features")
def features(req: FeaturesRequest) -> Dict[str, Any]:
    """
    Reads extracted JSON (records) from S3 and computes variance signals:
    variance = actual - baseline, variance_pct = variance / baseline (when baseline != 0)
    Returns top movers and writes feature output to S3 under features/.
    """
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    key = req.extracted_s3_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="extracted_s3_key is required")

    # Download extracted json
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        records = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download/parse extracted JSON: {e}")

    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=400, detail="Extracted JSON is empty or not a list of records")

    df = pd.DataFrame(records)

    missing = [c for c in [req.category_col, req.actual_col, req.baseline_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns in extracted data: {missing}")

    # Compute signals
    df["_actual"] = df[req.actual_col].map(_to_float)
    df["_base"] = df[req.baseline_col].map(_to_float)
    df["variance"] = df["_actual"] - df["_base"]
    df["variance_pct"] = df.apply(
        lambda r: (r["variance"] / r["_base"]) if (r["_base"] not in (None, 0)) else None,
        axis=1
    )

    # Simple anomaly flag: abs(variance_pct) >= 0.2 OR abs(variance) >= 10000 (fallback)
    # (In real finance packs we'd tune thresholds per scale; for MVP keep deterministic.)
    def is_anom(row):
        vp = row["variance_pct"]
        v = row["variance"]
        if vp is not None and abs(vp) >= 0.20:
            return True
        if v is not None and abs(v) >= 10000:
            return True
        return False

    df["anomaly_candidate"] = df.apply(is_anom, axis=1)

    # Top movers by absolute variance
    df_sorted_abs = df.sort_values(by="variance", key=lambda s: s.abs(), ascending=False)
    top_movers = df_sorted_abs.head(req.top_n)

    # Top positive and negative variances
    top_pos = df.sort_values(by="variance", ascending=False).head(req.top_n)
    top_neg = df.sort_values(by="variance", ascending=True).head(req.top_n)

    def pack(rows: pd.DataFrame) -> List[Dict[str, Any]]:
        out = []
        for _, r in rows.iterrows():
            out.append({
                req.category_col: r.get(req.category_col),
                "actual": r.get("_actual"),
                "baseline": r.get("_base"),
                "variance": r.get("variance"),
                "variance_pct": r.get("variance_pct"),
                "anomaly_candidate": bool(r.get("anomaly_candidate")),
            })
        return out

    result = {
        "source_extracted_s3_key": key,
        "category_col": req.category_col,
        "actual_col": req.actual_col,
        "baseline_col": req.baseline_col,
        "top_n": req.top_n,
        "top_movers": pack(top_movers),
        "top_positive": pack(top_pos),
        "top_negative": pack(top_neg),
        "anomaly_candidates": pack(df[df["anomaly_candidate"] == True].head(req.top_n)),
    }

    features_key = f"features/{utc_ts()}_{safe_name(os.path.basename(key))}.features.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=features_key,
            Body=json.dumps(result, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write features JSON to S3: {e}")

    result["features_s3_key"] = features_key
    return result

class InsightsRequest(BaseModel):
    features_s3_key: str
    max_drivers: int = 5


class VerifyRequest(BaseModel):
    extracted_s3_key: str
    features_s3_key: str
    output_json: dict

def fmt_pct(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    return f"{x*100:.1f}%"

def fmt_num(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    # simple formatting
    if abs(x) >= 1000000:
        return f"{x/1000000:.2f}M"
    if abs(x) >= 1000:
        return f"{x/1000:.2f}K"
    # keep small numbers clean
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.2f}"

@app.post("/insights")
def insights(req: InsightsRequest) -> Dict[str, Any]:
    """
    Deterministic insight generator (cheap MVP):
    Reads features JSON from S3, produces consistent executive narrative,
    writes to S3 under insights/.
    """
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    key = req.features_s3_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="features_s3_key is required")

    # Load features json
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download/parse features JSON: {e}")

    top_pos = payload.get("top_positive", [])[: req.max_drivers]
    top_neg = payload.get("top_negative", [])[: req.max_drivers]
    anoms = payload.get("anomaly_candidates", [])[: req.max_drivers]

    # Build consistent narrative
    # Executive summary: mention directionally the biggest movers + count anomalies
    drivers_bits = []
    for d in top_pos[:3]:
        name = d.get(payload.get("category_col", "category"), d.get("category"))
        v = d.get("variance")
        sign = "+" if (v is not None and v >= 0) else ""
        drivers_bits.append(f"{name} ({sign}{fmt_num(v)}, {fmt_pct(d.get('variance_pct'))})")

    summary = "Budget vs Actual highlights: " + (", ".join(drivers_bits) if drivers_bits else "No major variances detected.") + "."
    if anoms:
        summary += f" {len(anoms)} item(s) flagged as anomaly candidates based on variance thresholds."

    # Key drivers section
    key_drivers = []
    for d in top_pos:
        name = d.get(payload.get("category_col", "category"), d.get("category"))
        key_drivers.append(
            f"{name}: Actual {fmt_num(d.get('actual'))} vs Baseline {fmt_num(d.get('baseline'))} "
            f"→ Variance +{fmt_num(d.get('variance'))} ({fmt_pct(d.get('variance_pct'))})."
        )

    # Risks section: anomaly candidates
    risks = []
    if anoms:
        for d in anoms:
            name = d.get(payload.get("category_col", "category"), d.get("category"))
            risks.append(
                f"{name} shows an outsized variance of +{fmt_num(d.get('variance'))} ({fmt_pct(d.get('variance_pct'))}); "
                f"verify drivers and confirm if recurring."
            )
    else:
        risks.append("No anomaly candidates triggered by current thresholds.")

    # Recommendations (actionable, generic but structured)
    recs = []
    if anoms:
        recs.append("Validate flagged variances with source transactions and confirm one-time vs recurring drivers.")
    recs.append("Add owner + root-cause note for top 3 variances to standardize month-end commentary.")
    recs.append("Track the same categories next period to confirm whether trend is improving or worsening.")

    out = {
        "source_features_s3_key": key,
        "executive_summary": summary,
        "key_drivers": key_drivers,
        "risks": risks,
        "recommendations": recs,
        "meta": {
            "generated_at_utc": utc_ts(),
            "engine": "template_v1"
        }
    }

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


    insights_key = f"insights/{utc_ts()}_{safe_name(os.path.basename(key))}.insights.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=insights_key,
            Body=json.dumps(out, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write insights JSON to S3: {e}")

    out["insights_s3_key"] = insights_key
    return out

class CompareRequest(BaseModel):
    extracted_s3_key_a: str  # "previous" / baseline period
    extracted_s3_key_b: str  # "current" period
    category_col: str = "category"
    actual_col: str = "actual"
    baseline_col: Optional[str] = "budget"
    top_n: int = 5

def _load_extracted_df(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    records = json.loads(obj["Body"].read().decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError("Extracted JSON is not a list")
    return pd.DataFrame(records)

@app.post("/compare")
def compare(req: CompareRequest) -> Dict[str, Any]:
    """
    Compare two extracted datasets (period A vs period B).
    Produces 'what changed' deltas by category for Actual (and Variance if baseline exists).
    """
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    ka = req.extracted_s3_key_a.strip()
    kb = req.extracted_s3_key_b.strip()
    if not ka or not kb:
        raise HTTPException(status_code=400, detail="Both extracted_s3_key_a and extracted_s3_key_b are required")

    try:
        dfa = _load_extracted_df(s3, bucket, ka)
        dfb = _load_extracted_df(s3, bucket, kb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load extracted datasets: {e}")

    for c in [req.category_col, req.actual_col]:
        if c not in dfa.columns or c not in dfb.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column '{c}' in one of the datasets")

    # Reduce to key columns
    cols_a = [req.category_col, req.actual_col]
    cols_b = [req.category_col, req.actual_col]
    has_base = req.baseline_col and (req.baseline_col in dfa.columns) and (req.baseline_col in dfb.columns)
    if has_base:
        cols_a.append(req.baseline_col)
        cols_b.append(req.baseline_col)

    dfa2 = dfa[cols_a].copy()
    dfb2 = dfb[cols_b].copy()

    # numeric coercion
    dfa2["_actual_a"] = dfa2[req.actual_col].map(_to_float)
    dfb2["_actual_b"] = dfb2[req.actual_col].map(_to_float)
    if has_base:
        dfa2["_base_a"] = dfa2[req.baseline_col].map(_to_float)
        dfb2["_base_b"] = dfb2[req.baseline_col].map(_to_float)

    merged = dfa2[[req.category_col, "_actual_a"] + (["_base_a"] if has_base else [])].merge(
        dfb2[[req.category_col, "_actual_b"] + (["_base_b"] if has_base else [])],
        on=req.category_col,
        how="outer",
        indicator=True
    )

    # Identify new/missing categories
    new_cats = merged[merged["_merge"] == "right_only"][req.category_col].dropna().tolist()
    missing_cats = merged[merged["_merge"] == "left_only"][req.category_col].dropna().tolist()

    # Compute deltas
    merged["actual_delta"] = merged["_actual_b"] - merged["_actual_a"]

    if has_base:
        merged["var_a"] = merged["_actual_a"] - merged["_base_a"]
        merged["var_b"] = merged["_actual_b"] - merged["_base_b"]
        merged["variance_delta"] = merged["var_b"] - merged["var_a"]
    else:
        merged["var_a"] = None
        merged["var_b"] = None
        merged["variance_delta"] = None

    # Top changes
    top_actual_increase = merged.sort_values(by="actual_delta", ascending=False).head(req.top_n)
    top_actual_decrease = merged.sort_values(by="actual_delta", ascending=True).head(req.top_n)

    def pack_change(rows: pd.DataFrame) -> List[Dict[str, Any]]:
        out = []
        for _, r in rows.iterrows():
            out.append({
                req.category_col: r.get(req.category_col),
                "actual_a": r.get("_actual_a"),
                "actual_b": r.get("_actual_b"),
                "actual_delta": r.get("actual_delta"),
                "variance_a": r.get("var_a"),
                "variance_b": r.get("var_b"),
                "variance_delta": r.get("variance_delta"),
                "status": r.get("_merge"),
            })
        return out

    result = {
        "extracted_a": ka,
        "extracted_b": kb,
        "category_col": req.category_col,
        "actual_col": req.actual_col,
        "baseline_col": req.baseline_col if has_base else None,
        "top_n": req.top_n,
        "new_categories": new_cats,
        "missing_categories": missing_cats,
        "top_actual_increase": pack_change(top_actual_increase),
        "top_actual_decrease": pack_change(top_actual_decrease),
    }

    if has_base:
        top_var_change = merged.sort_values(by="variance_delta", key=lambda s: s.abs(), ascending=False).head(req.top_n)
        result["top_variance_change"] = pack_change(top_var_change)

    compare_key = f"comparisons/{utc_ts()}_{safe_name(os.path.basename(ka))}_VS_{safe_name(os.path.basename(kb))}.compare.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=compare_key,
            Body=json.dumps(result, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write comparison JSON to S3: {e}")

    result["comparison_s3_key"] = compare_key
    return result

class CompareInsightsRequest(BaseModel):
    comparison_s3_key: str
    top_n: int = 5

@app.post("/compare-insights")
def compare_insights(req: CompareInsightsRequest) -> Dict[str, Any]:
    """
    Deterministic narrative for comparison results (cheap MVP).
    Reads comparison JSON from S3 and produces 'what changed / why it matters / what to do'.
    """
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    key = req.comparison_s3_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="comparison_s3_key is required")

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        comp = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download/parse comparison JSON: {e}")

    inc = comp.get("top_actual_increase", [])[: req.top_n]
    dec = comp.get("top_actual_decrease", [])[: req.top_n]
    varchg = comp.get("top_variance_change", [])[: req.top_n]
    new_cats = comp.get("new_categories", [])
    missing_cats = comp.get("missing_categories", [])

    def line(item):
        cat = item.get(comp.get("category_col", "category"), item.get("category"))
        return (
            f"{cat}: Actual Δ {fmt_num(item.get('actual_delta'))} "
            f"(A {fmt_num(item.get('actual_a'))} → B {fmt_num(item.get('actual_b'))}); "
            + (
                f"Variance Δ {fmt_num(item.get('variance_delta'))} "
                f"(A {fmt_num(item.get('variance_a'))} → B {fmt_num(item.get('variance_b'))})."
                if comp.get("baseline_col") else ""
            )
        )

    # Executive summary: best 1-2 sentences
    headline_bits = []
    if inc:
        top = inc[0]
        headline_bits.append(f"Biggest increase: {top.get('category')} ({fmt_num(top.get('actual_delta'))}).")
    if dec:
        top = dec[0]
        headline_bits.append(f"Biggest decrease: {top.get('category')} ({fmt_num(top.get('actual_delta'))}).")
    if varchg:
        top = varchg[0]
        headline_bits.append(f"Largest variance shift: {top.get('category')} ({fmt_num(top.get('variance_delta'))}).")

    summary = "Period-over-period changes: " + (" ".join(headline_bits) if headline_bits else "No significant changes detected.")

    changes = {
        "top_actual_increases": [line(x) for x in inc],
        "top_actual_decreases": [line(x) for x in dec],
        "top_variance_changes": [line(x) for x in varchg] if varchg else [],
        "new_categories": new_cats,
        "missing_categories": missing_cats,
    }

    # Risks + actions
    risks = []
    actions = []

    if varchg:
        for x in varchg[:3]:
            cat = x.get("category")
            vd = x.get("variance_delta")
            if vd is not None and vd > 0:
                risks.append(f"{cat} variance worsened by {fmt_num(vd)} vs last period; confirm drivers and mitigation.")
    if new_cats:
        risks.append(f"New categories appeared: {', '.join(map(str, new_cats))}. Confirm mapping/classification.")
    if missing_cats:
        risks.append(f"Categories missing vs prior: {', '.join(map(str, missing_cats))}. Confirm reporting completeness.")

    if not risks:
        risks.append("No major risks flagged from period-over-period changes.")

    actions.append("Review top variance shifts and attach owner + root-cause notes (1–2 bullets each).")
    actions.append("Validate whether changes are timing, volume, price, or classification-driven.")
    actions.append("If variance worsened, propose corrective actions and track next period for reversion/recurrence.")

    out = {
        "source_comparison_s3_key": key,
        "executive_summary": summary,
        "key_changes": changes,
        "risks": risks,
        "recommended_actions": actions,
        "meta": {
            "generated_at_utc": utc_ts(),
            "engine": "compare_template_v1"
        }
    }

    out_key = f"compare_insights/{utc_ts()}_{safe_name(os.path.basename(key))}.compare_insights.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=out_key,
            Body=json.dumps(out, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write compare insights JSON to S3: {e}")

    out["compare_insights_s3_key"] = out_key
    return out

class PresignUploadRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    expires_in_seconds: int = 900  # 15 min


class RunRequest(BaseModel):
    # Either provide a raw uploaded file s3_key OR an extracted_s3_key
    s3_key: str | None = None
    extracted_s3_key: str | None = None

    # Mapping for tabular variance-style data
    category_col: str = "category"
    actual_col: str = "actual"
    baseline_col: str = "budget"  # budget/forecast/prior

    # Optional for Excel
    sheet_name: str | None = None

    # Controls
    top_n: int = 5
    max_drivers: int = 5
    verify: bool = False
    fail_on_unverified: bool = False

@app.post("/presign-upload")
def presign_upload(req: PresignUploadRequest) -> Dict[str, Any]:
    """
    SigV4 presigned PUT URL for direct-to-S3 uploads.
    Avoids API size limits (413) and works for large files.
    """
    bucket = get_bucket()
    s3 = get_s3()
    key = f"uploads/{utc_ts()}_{safe_name(req.filename)}"

    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": req.content_type,
                            },
            ExpiresIn=req.expires_in_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {e}")

    return {
        "s3_bucket": bucket,
        "s3_key": key,
        "upload_method": "PUT",
        "upload_url": url,
        "expires_in_seconds": req.expires_in_seconds,
        "next_step": {"call_extract_with": {"s3_key": key}},
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """Fetch job record from S3 (robust)."""
    s3 = get_s3()
    bucket = get_bucket()
    key = f"jobs/{job_id}.json"
    try:
        data = _s3_read_json(s3, bucket, key)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {e}")

    # Normalize: if stored as [ { ... } ] return the dict
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]

    # If still list, wrap (so callers always get a dict-ish response)
    if isinstance(data, list):
        return {"job_id": job_id, "raw": data}

    return data



@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    """
    Orchestrates: extract -> features -> insights
    Writes an auditable job record to S3 under jobs/<job_id>.json
    Returns job_id + keys + executive summary.
    """
    try:
        bucket = get_bucket()
        s3 = get_s3()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "started",
        "created_at_utc": utc_ts(),
        "input": {
            "s3_key": req.s3_key,
            "extracted_s3_key": req.extracted_s3_key,
            "mapping": {
                "category_col": req.category_col,
                "actual_col": req.actual_col,
                "baseline_col": req.baseline_col,
            },
            "sheet_name": req.sheet_name,
            "top_n": req.top_n,
            "max_drivers": req.max_drivers,
        },
        "outputs": {},
        "errors": [],
        "updated_at_utc": utc_ts(),
    }

    _write_job(s3, bucket, job_id, job)

    try:
        # 1) Extract (if needed)
        extracted_key = req.extracted_s3_key
        extract_resp = None

        if not extracted_key:
            if not req.s3_key:
                raise HTTPException(status_code=400, detail="Provide either s3_key or extracted_s3_key")

            extract_resp = extract(ExtractRequest(s3_key=req.s3_key, sheet_name=req.sheet_name))
            extracted_key = extract_resp["extracted_s3_key"]

        job["outputs"]["extracted_s3_key"] = extracted_key
        job["status"] = "extracted"
        job["updated_at_utc"] = utc_ts()
        _write_job(s3, bucket, job_id, job)

        # 2) Features
        feat_resp = features(FeaturesRequest(
            extracted_s3_key=extracted_key,
            category_col=req.category_col,
            actual_col=req.actual_col,
            baseline_col=req.baseline_col,
            top_n=req.top_n
        ))

        job["outputs"]["features_s3_key"] = feat_resp["features_s3_key"]
        job["status"] = "featured"
        job["updated_at_utc"] = utc_ts()
        _write_job(s3, bucket, job_id, job)

        # 3) Insights
        ins_resp = insights(InsightsRequest(
            features_s3_key=feat_resp["features_s3_key"],
            max_drivers=req.max_drivers
        ))

        job["outputs"]["insights_s3_key"] = ins_resp["insights_s3_key"]
        job["outputs"]["executive_summary"] = ins_resp["executive_summary"]
        if extract_resp:
            job["outputs"]["preview"] = extract_resp.get("preview")

        job["status"] = "completed"
        job["updated_at_utc"] = utc_ts()
        _write_job(s3, bucket, job_id, job)


        # ---- verification gate ----

        if getattr(req, "verify", True):

            try:

                extracted_obj = _s3_read_json(s3, bucket, job["outputs"]["extracted_s3_key"])

                features_obj  = _s3_read_json(s3, bucket, job["outputs"]["features_s3_key"])

                insights_obj  = _s3_read_json(s3, bucket, job["outputs"]["insights_s3_key"])

                verification = verify_insights_v1(extracted=extracted_obj, features=features_obj, insights=insights_obj)

            except Exception as e:

                verification = {"pass": False, "reasons": [f"verification_exception:{e.__class__.__name__}"]}


            job.setdefault("meta", {})

            job["meta"]["verification"] = verification

            _write_job(s3, bucket, job_id, job)


            if getattr(req, "fail_on_unverified", True) and not verification.get("pass", False):

                raise HTTPException(status_code=422, detail={"verification": verification, "job_id": job_id})

        return {
            "job_id": job_id,
            "job_s3_key": _job_key(job_id),
            "input": job["input"],
            "outputs": job["outputs"],
        }

    except HTTPException as e:
        job["status"] = "failed"
        job["errors"].append({"type": "http", "detail": str(e.detail)})
        job["updated_at_utc"] = utc_ts()
        _write_job(s3, bucket, job_id, job)
        raise

    except Exception as e:
        job["status"] = "failed"
        job["errors"].append({"type": "exception", "detail": str(e)})
        job["updated_at_utc"] = utc_ts()
        _write_job(s3, bucket, job_id, job)
        raise HTTPException(status_code=500, detail=f"Run failed: {e}")

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


@app.post("/run2")
def run2(req: RunRequest) -> Dict[str, Any]:
    """
    Safe wrapper: runs the pipeline and returns the stored job JSON from S3.
    This ensures the response includes meta.verification (and any other job metadata).
    """
    out = run(req)  # existing endpoint returns {"job_id":..., ...} or job-like dict
    job_id = out.get("job_id") if isinstance(out, dict) else None
    if not job_id:
        return out  # fallback

    s3 = get_s3()
    bucket = get_bucket()
    job_key = f"jobs/{job_id}.json"
    try:
        return _s3_read_json(s3, bucket, job_key)
    except Exception:
        return out


from fastapi import Response


from fastapi import Response

@app.post("/run-verified")
def run_verified(req: RunRequest, response: Response) -> Dict[str, Any]:
    """
    Enterprise-safe run:
    - forces verification
    - optionally fails closed (422) when verification fails
    - returns the job json (including meta.verification)
    """
    # Build a new RunRequest-like dict so we don't mutate Pydantic object
    payload = req.model_dump()
    payload["verify"] = True  # force verification
    # Ensure fail_on_unverified default is honored
    payload["fail_on_unverified"] = bool(payload.get("fail_on_unverified", False))

    # Call run2 using a proper model instance (safe)
    req2 = RunRequest(**payload)
    out = run2(req2)

    ver = (out.get("meta") or {}).get("verification") if isinstance(out, dict) else None
    if req2.fail_on_unverified:
        if not (ver and ver.get("pass") is True):
            response.status_code = 422
            return {"detail": {"verification": ver or {"pass": False, "reasons": ["missing_verification"]},
                               "job_id": out.get("job_id") if isinstance(out, dict) else None}}
    return out

# =========================
# Product v1 (Locked) API
# =========================

from app.product_contracts import RunVerifiedRequest, RunVerifiedResponse, Failure
from app.product_verify import run_product_verification
from app.product_report import build_verification_report
from app.product_storage import tenant_job_key, tenant_report_key, s3_put_json, s3_get_json

class _RunV1Out(BaseModel):
    # kept for compatibility if you want to inspect response shapes easily in docs
    pass

@app.post("/run-verified-v1", response_model=RunVerifiedResponse)
def run_verified_v1(req: RunVerifiedRequest):
    """
    Locked v1 product path:
    - input = baseline + projection + assumptions narrative
    - output = PASS/FAIL + reasons + report artifact
    - fail-closed
    """
    bucket = get_bucket()
    s3 = get_s3()

    job_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    result = run_product_verification(req)

    verification_meta = {
        "pass": (result.verdict == "PASS"),
        "fail_count": len(result.failures),
        "timestamp": ts,
        "run_id": run_id,
    }
    response_meta = {
        "verification": verification_meta,
        "tenant_id": req.tenant_id,
        "job_id": job_id,
    }

    # Build report JSON artifact (core product artifact)
    report_obj = build_verification_report(
        verification_id=f"IG-{run_id}",
        timestamp=ts,
        artifact_label=f"{req.projection.period} projection vs {req.baseline.period} baseline",
        verdict=result.verdict,
        failures=[f.model_dump() for f in result.failures],
        verified_scope=result.verified_scope,
        reasoning_trace=result.reasoning_trace,
    )

    report_key = tenant_report_key(req.tenant_id, run_id)
    s3_put_json(s3, bucket, report_key, report_obj)

    # Persist job record (tenant-isolated)
    job = {
        "job_id": job_id,
        "run_id": run_id,
        "tenant_id": req.tenant_id,
        "timestamp": ts,
        "verdict": result.verdict,
        "meta": response_meta,
        "failures": [f.model_dump() for f in result.failures],
        "report": {"format": "signoff", "location": report_key},
    }
    s3_put_json(s3, bucket, tenant_job_key(req.tenant_id, job_id), job)

    # Fail-closed: non-200 on FAIL
    if result.verdict == "FAIL":
        return JSONResponse(
            status_code=422,
            content={
                "verdict": "FAIL",
                "meta": response_meta,
                "failures": [f.model_dump() for f in result.failures],
                "instruction": "External approval should not proceed until issues are resolved or explicitly justified.",
                "report": {"format": "signoff", "location": report_key},
            },
        )



    return RunVerifiedResponse(
        verdict="PASS",
        meta=response_meta,
        failures=[],
        instruction=None,
        report={"format": "signoff", "location": report_key},
    )


@app.get("/jobs-v1/{tenant_id}/{job_id}")
def get_job_v1(tenant_id: str, job_id: str):
    """
    Tenant-isolated job retrieval for product v1.
    """
    bucket = get_bucket()
    s3 = get_s3()
    key = tenant_job_key(tenant_id, job_id)
    try:
        return s3_get_json(s3, bucket, key)
    except ClientError:
        raise HTTPException(status_code=404, detail={"error": "job_not_found", "job_id": job_id})
