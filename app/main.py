import os
from datetime import datetime, timezone

import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException

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

def get_s3():
    region = os.getenv("AWS_REGION")
    if not region:
        raise RuntimeError("AWS_REGION env var not set")
    return boto3.client("s3", region_name=region)

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

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = (file.filename or "upload").replace("/", "_")
    key = f"uploads/{ts}_{safe_name}"

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
