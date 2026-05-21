import os
from pathlib import Path

from train_data.utils.s3_utils import S3BucketService, s3_service as default_s3_service

# Hardcoded contract: SHARED between automl-mcp and GenerativeModelsMCP.
# Both servers read/write the same key so state.json is one source of truth.
# Do NOT make this configurable — see comment in base_state.py.
STATE_S3_KEY = "state/state.json"


def build_s3_service(
    endpoint: str = None,
    access_key: str = None,
    secret_key: str = None,
    bucket_name: str = None,
) -> S3BucketService:
    endpoint = endpoint or os.getenv("ENDPOINT_URL") or default_s3_service.endpoint
    access_key = access_key or os.getenv("ACCESS_KEY") or default_s3_service.access_key
    secret_key = secret_key or os.getenv("SECRET_KEY") or default_s3_service.secret_key
    bucket_name = bucket_name or os.getenv("BUCKET_NAME") or default_s3_service.bucket_name

    if not endpoint:
        raise ValueError("S3 endpoint is empty. Set ENDPOINT_URL.")
    if not access_key:
        raise ValueError("S3 access key is empty. Set ACCESS_KEY.")
    if not secret_key:
        raise ValueError("S3 secret key is empty. Set SECRET_KEY.")
    if not bucket_name:
        raise ValueError("S3 bucket name is empty. Set BUCKET_NAME.")

    return S3BucketService(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
    )


def _split_s3_key(s3_key: str):
    normalized = s3_key.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("S3 key is empty.")
    if "/" not in normalized:
        return "", normalized
    prefix, source_file_name = normalized.rsplit("/", 1)
    return prefix, source_file_name


def download_state_file(
    local_path: str,
    s3_bucket_service: S3BucketService = None,
) -> str:
    local_state_path = Path(local_path)
    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    service = s3_bucket_service or build_s3_service()
    service.download_image_from_s3(s3_key=STATE_S3_KEY, local_path=str(local_state_path))
    return STATE_S3_KEY


def upload_state_file(
    local_path: str,
    s3_bucket_service: S3BucketService = None,
) -> str:
    local_state_path = Path(local_path)
    if not local_state_path.is_file():
        raise FileNotFoundError(f"State file not found: {local_state_path}")

    service = s3_bucket_service or build_s3_service()
    prefix, source_file_name = _split_s3_key(STATE_S3_KEY)
    service.upload_file_object(
        prefix=prefix,
        source_file_name=source_file_name,
        file_path=str(local_state_path),
    )
    return STATE_S3_KEY
