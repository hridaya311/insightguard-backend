import os, argparse, boto3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_prefix", required=True, help="e.g. datasets/synth_v2_big/")
    ap.add_argument("--keep_prefix", required=True, help="exact prefix to keep, e.g. datasets/synth_v2_big/20260125T123000Z")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    bucket = os.environ["S3_BUCKET"]
    region = os.environ["AWS_REGION"]
    s3 = boto3.client("s3", region_name=region)

    # list "folders" under base_prefix
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = set()

    # find top-level prefixes: base_prefix/<ts>/
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=args.base_prefix, Delimiter="/")
    for cp in resp.get("CommonPrefixes", []):
        prefixes.add(cp["Prefix"].rstrip("/"))

    keep = args.keep_prefix.rstrip("/")
    delete_prefixes = sorted([p for p in prefixes if p != keep])

    print("Bucket:", bucket)
    print("Keep:", keep)
    print("Will delete:", delete_prefixes if delete_prefixes else "(none)")

    if args.dry_run:
        print("Dry run only. No deletions performed.")
        return

    # delete objects under each delete prefix
    for pref in delete_prefixes:
        print("Deleting prefix:", pref)
        cont = True
        token = None
        while cont:
            kwargs = dict(Bucket=bucket, Prefix=pref + "/")
            if token:
                kwargs["ContinuationToken"] = token
            out = s3.list_objects_v2(**kwargs)
            objs = [{"Key": o["Key"]} for o in out.get("Contents", [])]
            if objs:
                # delete in batches of 1000
                for i in range(0, len(objs), 1000):
                    batch = objs[i:i+1000]
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            cont = out.get("IsTruncated", False)
            token = out.get("NextContinuationToken")

    print("Done.")
