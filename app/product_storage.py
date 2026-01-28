from __future__ import annotations
import json
from typing import Dict, Any

def tenant_job_key(tenant_id: str, job_id: str) -> str:
    return f"tenants/{tenant_id}/jobs/{job_id}.json"

def tenant_report_key(tenant_id: str, run_id: str) -> str:
    return f"tenants/{tenant_id}/runs/{run_id}/verification_report.json"

def s3_put_json(s3, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

def s3_get_json(s3, bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))
