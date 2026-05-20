"""Download every object under an S3 prefix into a local folder.

Loads credentials from `automl/.env` (ENDPOINT_URL, ACCESS_KEY, SECRET_KEY,
BUCKET_NAME). Skips files whose local size already matches the S3 object,
unless `--overwrite` is passed.

Example:
    python automl/scripts/s3_download_folder.py \\
        --prefix archive/docked_data_for_train \\
        --dest SMILES_generative_models/docked_data_for_train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

DEFAULT_ENV = Path(__file__).resolve().parent.parent / ".env"


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=(os.getenv("ENDPOINT_URL") or "").strip(),
        aws_access_key_id=(os.getenv("ACCESS_KEY") or "").strip(),
        aws_secret_access_key=(os.getenv("SECRET_KEY") or "").strip(),
        config=Config(signature_version="s3v4"),
    )


def fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", required=True, help="S3 key prefix (e.g. 'archive/docked_data_for_train').")
    parser.add_argument("--dest", required=True, help="Local destination directory.")
    parser.add_argument("--bucket", default=None, help="Bucket name (default: BUCKET_NAME from .env).")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if local size matches.")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help=f"Path to .env. Default: {DEFAULT_ENV}")
    args = parser.parse_args()

    load_dotenv(args.env)

    bucket = args.bucket or (os.getenv("BUCKET_NAME") or "").strip()
    if not bucket:
        print("ERROR: bucket not specified and BUCKET_NAME missing from .env", file=sys.stderr)
        return 2

    prefix = args.prefix.replace("\\", "/").strip("/") + "/"
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    client = make_client()

    paginator = client.get_paginator("list_objects_v2")
    objects: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            objects.append((item["Key"], int(item["Size"])))

    if not objects:
        print(f"No objects under s3://{bucket}/{prefix}")
        return 1

    total_bytes = sum(size for _, size in objects)
    print(f"Downloading {len(objects)} files ({fmt_bytes(total_bytes)}) into {dest}/")

    downloaded = 0
    skipped = 0
    failed = 0
    for key, size in objects:
        rel = key[len(prefix):] if key.startswith(prefix) else key
        if not rel or rel.endswith("/"):
            continue
        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.overwrite and local_path.exists() and local_path.stat().st_size == size:
            print(f"  SKIP  {rel} (already present, {fmt_bytes(size)})")
            skipped += 1
            continue

        print(f"  DL    {rel} ({fmt_bytes(size)}) ...", end=" ", flush=True)
        try:
            client.download_file(bucket, key, str(local_path))
            print("ok")
            downloaded += 1
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"Done. downloaded={downloaded} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
