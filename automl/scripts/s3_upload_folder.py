"""Upload all files from a local folder to S3 under a given prefix.

Loads credentials from `automl/.env` (ENDPOINT_URL, ACCESS_KEY, SECRET_KEY,
BUCKET_NAME). Skips files that already exist with the same size, unless
`--overwrite` is passed.

Example:
    python automl/scripts/s3_upload_folder.py \\
        SMILES_generative_models/docked_data_for_train \\
        --prefix archive/docked_data_for_train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
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


def head_size(client, bucket: str, key: str) -> int | None:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        return int(head["ContentLength"])
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Local directory whose files to upload.")
    parser.add_argument("--bucket", default=None, help="Bucket name (default: BUCKET_NAME from .env).")
    parser.add_argument("--prefix", required=True, help="S3 key prefix (e.g. 'archive/docked_data_for_train').")
    parser.add_argument("--recursive", action="store_true", help="Walk subdirectories too.")
    parser.add_argument("--overwrite", action="store_true", help="Re-upload even if size matches.")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help=f"Path to .env. Default: {DEFAULT_ENV}")
    args = parser.parse_args()

    load_dotenv(args.env)

    bucket = args.bucket or (os.getenv("BUCKET_NAME") or "").strip()
    if not bucket:
        print("ERROR: bucket not specified and BUCKET_NAME missing from .env", file=sys.stderr)
        return 2

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        return 2

    files = sorted(folder.rglob("*") if args.recursive else folder.iterdir())
    files = [p for p in files if p.is_file()]
    if not files:
        print(f"No files found in {folder}")
        return 0

    prefix = args.prefix.replace("\\", "/").strip("/")
    client = make_client()

    total_bytes = sum(p.stat().st_size for p in files)
    print(f"Uploading {len(files)} files ({fmt_bytes(total_bytes)}) to s3://{bucket}/{prefix}/")

    uploaded = 0
    skipped = 0
    failed = 0
    for path in files:
        rel = path.relative_to(folder).as_posix()
        key = f"{prefix}/{rel}"
        size = path.stat().st_size

        if not args.overwrite:
            existing = head_size(client, bucket, key)
            if existing == size:
                print(f"  SKIP  {key} (already present, {fmt_bytes(size)})")
                skipped += 1
                continue

        print(f"  UP    {key} ({fmt_bytes(size)}) ...", end=" ", flush=True)
        try:
            client.upload_file(str(path), bucket, key)
            print("ok")
            uploaded += 1
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"Done. uploaded={uploaded} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
