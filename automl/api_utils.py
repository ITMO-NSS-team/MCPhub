import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import Body
from pydantic import BaseModel

from utils.automl_main import (
    BASE_DIR as AUTOML_BASE_DIR,
    _resolve_existing_path,
    run_predict_automl_from_list,
    run_train_automl,
)
from utils.base_state import TrainState

try:
    from s3_utils import S3BucketService, s3_service as default_s3_service
except ModuleNotFoundError:
    from .s3_utils import S3BucketService, s3_service as default_s3_service


_TRAIN_DATA_SIZE_LIMITS: Dict[str, Optional[int]] = {
    "TEST": 200,
    "LIMITED_DEV": 10_000,
    "UNLIMITED": None,
}
DEFAULT_TRAIN_DATA_LIMIT_MODE = "LIMITED_DEV"


def _resolve_train_data_size_limit() -> Optional[int]:
    raw = (os.getenv("TRAIN_DATA_LIMIT") or DEFAULT_TRAIN_DATA_LIMIT_MODE).strip().upper()
    if raw not in _TRAIN_DATA_SIZE_LIMITS:
        raise ValueError(
            f"Unknown TRAIN_DATA_LIMIT={raw!r}. "
            f"Expected one of: {sorted(_TRAIN_DATA_SIZE_LIMITS)}"
        )
    return _TRAIN_DATA_SIZE_LIMITS[raw]


def _apply_train_data_size_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Cap row count of a training DataFrame per `TRAIN_DATA_LIMIT` env var.

    Modes: `TEST` (200), `LIMITED_DEV` (10 000), `UNLIMITED` (no cap).
    Default: `LIMITED_DEV`. Guards against OOM on low-memory hosts.
    """
    limit = _resolve_train_data_size_limit()
    if limit is None:
        print(f"TRAIN_DATA_LIMIT=UNLIMITED: keeping all {len(df)} rows")
        return df
    if len(df) <= limit:
        print(f"TRAIN_DATA_LIMIT cap={limit}: dataset has {len(df)} rows, keeping all")
        return df
    print(f"TRAIN_DATA_LIMIT cap={limit}: truncating dataset from {len(df)} to {limit} rows")
    return df.head(limit)

class MLData(BaseModel):
        """
        Represents a container for machine learning data, handling loading, processing, and storage.
        
                 Class Attributes:
                 - data: The loaded dataset.
                 - case: Specifies whether the data represents a regression or classification task.
                 - data_path: Path to the dataset file.
                 - target_column: Name of the target variable column.
                 - smiles_list: List of SMILES strings if applicable to the dataset.
                 - timeout: Timeout value for data loading operations.
                 - feature_column: Name of the feature column.
                 - path_to_save: Path to save the processed data.
                 - description: A textual description of the dataset.
                 - regression_props: List of regression properties.
                 - classification_props: List of classification properties.
        """

        data:dict = None
        case:str = None
        data_path:str = None
        target_column:list = None
        smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:list = ['Smiles']
        path_to_save:str = 'train_model_data/trained_data'
        description:str = 'Unknown case.'
        regression_props:list= None
        classification_props:list = None
        save_trained_data_to_sync_server:bool = False
        endpoint_url:str = os.getenv("ENDPOINT_URL")
        access_key:str = os.getenv("ACCESS_KEY")
        secret_key:str = os.getenv("SECRET_KEY")
        bucket_name:str = os.getenv("BUCKET_NAME")
        s3_key:str = None
        # Full HTTP/HTTPS URL of the training CSV (e.g. an S3 presigned URL).
        # When set, takes precedence over `s3_key`: the CSV is fetched via
        # `requests.get(...)` and no S3 credentials are required to read it.
        data_url:str = None
        # Backward compatibility with previous payload names.
        s3_bucket:str = None
        s3_endpoint_url:str = None


def _build_s3_service(data: MLData) -> S3BucketService:
    endpoint_url = data.endpoint_url or data.s3_endpoint_url or os.getenv("ENDPOINT_URL") or default_s3_service.endpoint
    access_key = data.access_key or os.getenv("ACCESS_KEY") or default_s3_service.access_key
    secret_key = data.secret_key or os.getenv("SECRET_KEY") or default_s3_service.secret_key
    bucket_name = data.bucket_name or data.s3_bucket or os.getenv("BUCKET_NAME") or default_s3_service.bucket_name

    if not endpoint_url:
        raise ValueError("S3 endpoint is empty. Set `endpoint_url` or env `ENDPOINT_URL`.")
    if not access_key:
        raise ValueError("S3 access key is empty. Set `access_key` or env `ACCESS_KEY`.")
    if not secret_key:
        raise ValueError("S3 secret key is empty. Set `secret_key` or env `SECRET_KEY`.")
    if not bucket_name:
        raise ValueError("S3 bucket is empty. Set `bucket_name` or env `BUCKET_NAME`.")

    return S3BucketService(
        endpoint=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
    )


def _resolve_train_s3_key(data: MLData) -> str:
    """Resolve S3 object key for training CSV.

    Priority:
        1. `data.s3_key` if explicitly provided (supports plain key or `s3://bucket/key` URI).
        2. Default fallback: `train/{case}.csv`.
    """
    raw_key = (data.s3_key or "").strip()
    if raw_key:
        if raw_key.lower().startswith("s3://"):
            # s3://bucket/key/path.csv -> "key/path.csv"
            without_scheme = raw_key[len("s3://"):]
            _, _, key_part = without_scheme.partition("/")
            normalized = key_part
        else:
            normalized = raw_key
        normalized = normalized.replace("\\", "/").lstrip("/")
        if not normalized:
            raise ValueError(f"`s3_key` is empty after normalization: {data.s3_key!r}")
        return normalized
    if not data.case:
        raise ValueError("Either `s3_key` or `case` must be provided.")
    return f"train/{data.case}.csv"


def _download_dataset_from_s3(data: MLData) -> str:
    """Fetch the training CSV and return the normalized local path.

    Source resolution (priority order):
        1. `data.data_url` — HTTP(S) URL (e.g. an S3 presigned URL) is
           downloaded via `requests.get(...)`. No S3 credentials are read,
           so the URL may point at any reachable endpoint / account.
        2. `data.s3_key` (plain key or `s3://bucket/key` URI) — downloaded
           via the configured boto3 client.
        3. Default `train/{case}.csv` (when neither URL nor explicit key is
           given but `case` is set) — downloaded via boto3.
    """
    if not data.case and not data.s3_key and not data.data_url:
        raise ValueError("`case`, `s3_key`, or `data_url` is required.")

    if not data.data_path:
        case_dir = data.case or "from_s3_key"
        data.data_path = f"data/{case_dir}/data.csv"
    data.data_path = data.data_path.replace("\\", "/")
    local_dir = os.path.dirname(data.data_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    data_url = (data.data_url or "").strip()
    if data_url:
        # HTTP/HTTPS path — works for S3 presigned URLs and any public URL.
        timeout_s = int(os.getenv("TRAIN_DATA_HTTP_TIMEOUT_S", "1800"))
        print(f"Downloading training CSV via HTTP -> {data.data_path}")
        with requests.get(data_url, stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            with open(data.data_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
        print(f"Downloaded {os.path.getsize(data.data_path)} bytes from URL")
        return data.data_path

    # S3 path — boto3 client with configured credentials.
    s3_service = _build_s3_service(data)
    s3_key = _resolve_train_s3_key(data)
    s3_service.download_image_from_s3(s3_key=s3_key, local_path=data.data_path)
    return data.data_path


def train_ml_with_data(data:MLData=Body()):
    """
    Trains a machine learning model using provided data.
    
    This method prepares data for model training by saving it to a CSV file and 
    configuring the training environment with details like data location, 
    feature columns, and target columns. It then initiates the AutoML training process. 
    This allows the system to learn from specific datasets provided by the user.
    
    Args:
        data (MLData): The input data containing case details, data (as a pandas DataFrame), 
                       feature columns, target columns, and prediction properties.
    
    Returns:
        None
    """
    state = TrainState()
    state.add_new_case(case_name=data.case,
                        rewrite=True,
                        description=data.description)
    if data.case:
            _download_dataset_from_s3(data)
            df = pd.read_csv(data.data_path).dropna()
            if data.feature_column[0] not in df.columns:
                raise ValueError(
                    f"Feature column '{data.feature_column[0]}' not found in downloaded file. "
                    f"Available columns: {df.columns.tolist()}"
                )
            df = _apply_train_data_size_limit(df)
            df.to_csv(data.data_path, index=False)
    elif data.data is not None:
            df = pd.DataFrame(data.data)
            data.data_path = f"data/{data.case}/data.csv"
            os.makedirs(os.path.dirname(data.data_path), exist_ok=True)
            data.data_path = f"data/{data.case}/data.csv"
            os.makedirs(os.path.dirname(data.data_path), exist_ok=True)
            df = df.dropna()
            df = _apply_train_data_size_limit(df)
            df.to_csv(data.data_path, index=False)
                    
    state.ml_model_upd_data(case=data.case,
                            data_path=data.data_path,
                            feature_column=data.feature_column,
                            target_column=data.target_column,
                                predictable_properties={"regression":data.regression_props, "classification":data.classification_props})
    run_train_automl(case=data.case,
                        path_to_save=data.path_to_save,
                        timeout=data.timeout,
                        save_trained_data_to_sync_server=data.save_trained_data_to_sync_server)

def inference_ml(data:MLData=Body()):
    """
    Runs a prediction using an automated machine learning model.

    Args:
        data: An MLData object containing the case identifier and a list of SMILES strings representing chemical structures.

    Returns:
        The prediction results generated by the AutoML model, based on the provided chemical structures and case.
    """
    resutls = run_predict_automl_from_list(data.case,data=data.smiles_list)
    return resutls


def upload_predictions_csv_to_s3(
    data: MLData,
    local_csv_path: str,
    s3_key: str,
    presigned_expiration: int = 3600,
) -> dict:
    """Upload a local predictions CSV to S3 and return its presigned URL.

    Args:
        data: MLData used to resolve S3 credentials (bucket / endpoint / keys).
        local_csv_path: path to the CSV file with predictions on the local filesystem.
        s3_key: full S3 object key (e.g. `predictions/Alzheimer/<uuid>.csv`).
        presigned_expiration: lifetime of returned presigned URL in seconds.

    Returns:
        Dict with `bucket_name`, `s3_key`, `presigned_url`, `expires_in`.
    """
    s3_service = _build_s3_service(data)
    normalized_key = s3_key.replace("\\", "/").lstrip("/")
    if "/" in normalized_key:
        prefix, source_file_name = normalized_key.rsplit("/", 1)
    else:
        prefix, source_file_name = "", normalized_key
    s3_service.upload_file_object(
        prefix=prefix,
        source_file_name=source_file_name,
        file_path=local_csv_path,
    )
    presigned_url = s3_service.generate_presigned_url(
        s3_key=normalized_key, expiration=presigned_expiration
    )
    return {
        "bucket_name": s3_service.bucket_name,
        "s3_key": normalized_key,
        "presigned_url": presigned_url,
        "expires_in": presigned_expiration,
    }


def download_smiles_csv_from_s3(
    data: MLData,
    s3_key_or_uri: str,
    local_csv_path: str,
) -> str:
    """Download a CSV file from S3 to a local path.

    Args:
        data: MLData used to resolve S3 credentials.
        s3_key_or_uri: S3 object key (e.g. `predictions/foo.csv`) or `s3://bucket/key`.
        local_csv_path: target local path.

    Returns:
        The local path that was written.
    """
    raw = (s3_key_or_uri or "").strip()
    if not raw:
        raise ValueError("S3 key is empty.")
    if raw.lower().startswith("s3://"):
        without_scheme = raw[len("s3://"):]
        _, _, key_part = without_scheme.partition("/")
        normalized = key_part
    else:
        normalized = raw
    normalized = normalized.replace("\\", "/").lstrip("/")
    if not normalized:
        raise ValueError(f"S3 key is empty after normalization: {s3_key_or_uri!r}")

    local_dir = os.path.dirname(local_csv_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    s3_service = _build_s3_service(data)
    s3_service.download_image_from_s3(s3_key=normalized, local_path=local_csv_path)
    return local_csv_path


def download_weights_folder_from_s3(
    data: MLData,
    s3_prefix: str,
    local_target_dir: str,
) -> int:
    """Download every object under `s3_prefix` into `local_target_dir`.

    Returns:
        Number of files downloaded. Zero means the prefix exists but is empty
        (or does not exist at all).
    """
    s3_service = _build_s3_service(data)
    normalized_prefix = s3_prefix.replace("\\", "/").lstrip("/")
    if not normalized_prefix.endswith("/"):
        normalized_prefix = normalized_prefix + "/"

    keys = s3_service.list_objects(prefix=normalized_prefix)
    if not keys:
        return 0

    target = Path(local_target_dir)
    target.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for key in keys:
        if key.startswith(normalized_prefix):
            relative = key[len(normalized_prefix):]
        else:
            relative = key
        relative = relative.lstrip("/")
        if not relative:
            continue
        local_file_path = target / relative
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_service.download_image_from_s3(s3_key=key, local_path=str(local_file_path))
        downloaded += 1

    return downloaded


def _resolve_weights_local_dir(weights_path: str) -> Path:
    """Resolve a weights path stored in state.json to a local Path.

    Mirrors `_resolve_existing_path` semantics but always returns a Path,
    so callers can mkdir/download into the chosen location even if it
    does not exist yet.
    """
    try:
        resolved = _resolve_existing_path(weights_path)
    except Exception:
        resolved = None

    if resolved:
        return Path(resolved)

    raw = Path(weights_path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "automl":
        raw = Path(*raw.parts[1:])
    return Path(AUTOML_BASE_DIR) / raw


def ensure_ml_weights_available(data: MLData) -> Dict[str, Any]:
    """Ensure trained model weights for `data.case` are available locally.

    Checks each predictable problem registered in state.json for the case.
    For every missing local artifact, attempts to download files from
    `s3://{bucket}/ml_weights/{case}/{folder_name}/...` (the prefix used by
    `_upload_folder_to_s3` during training).

    Returns:
        Dict with keys:
        - `status`: one of
            * `ok`           — every needed artifact is present locally
              (either cached or just downloaded).
            * `case_not_found` — case is not in state.json.
            * `no_predictable_properties` — case has no problems registered.
            * `weights_not_found` — at least one problem has no local files
              and no S3 backup.
        - `case`, `message`, `problems_checked`, `problems_downloaded`,
          `problems_missing`, `details`.
    """
    state = TrainState()
    case_state = state(data.case, "ml")
    if case_state is None:
        return {
            "status": "case_not_found",
            "case": data.case,
            "message": (
                f"Case '{data.case}' is not registered in state.json. "
                "Train the model first via `train_ml`."
            ),
            "problems_checked": [],
            "problems_downloaded": [],
            "problems_missing": [],
            "details": {},
        }

    properties = case_state.get("Predictable properties") or {}
    weights_paths = case_state.get("weights_path") or {}

    if not properties:
        return {
            "status": "no_predictable_properties",
            "case": data.case,
            "message": (
                f"Case '{data.case}' has no predictable properties registered. "
                "Re-run `train_ml` with `regression_props` and/or `classification_props`."
            ),
            "problems_checked": [],
            "problems_downloaded": [],
            "problems_missing": [],
            "details": {},
        }

    problems_checked: List[str] = []
    problems_downloaded: List[str] = []
    problems_missing: List[str] = []
    details: Dict[str, Any] = {}

    for problem in properties:
        problems_checked.append(problem)
        weights_path = weights_paths.get(problem)
        if not weights_path:
            problems_missing.append(problem)
            details[problem] = {
                "status": "missing",
                "reason": "No `weights_path` recorded in state for this problem.",
            }
            continue

        local_dir = _resolve_weights_local_dir(weights_path)
        if local_dir.is_dir() and any(local_dir.iterdir()):
            details[problem] = {
                "status": "local",
                "local_path": str(local_dir),
            }
            continue

        s3_prefix = f"ml_weights/{data.case}/{Path(weights_path).name}/"
        try:
            count = download_weights_folder_from_s3(
                data=data,
                s3_prefix=s3_prefix,
                local_target_dir=str(local_dir),
            )
        except Exception as exc:
            problems_missing.append(problem)
            details[problem] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "s3_prefix": s3_prefix,
                "local_path": str(local_dir),
            }
            continue

        if count > 0:
            problems_downloaded.append(problem)
            details[problem] = {
                "status": "downloaded",
                "local_path": str(local_dir),
                "s3_prefix": s3_prefix,
                "files_downloaded": count,
            }
        else:
            problems_missing.append(problem)
            details[problem] = {
                "status": "missing",
                "reason": "No objects under expected S3 prefix.",
                "s3_prefix": s3_prefix,
                "local_path": str(local_dir),
            }

    if problems_missing:
        return {
            "status": "weights_not_found",
            "case": data.case,
            "message": (
                f"Trained weights for case '{data.case}' are not available "
                f"(missing problems: {problems_missing}). Train via `train_ml` "
                f"with `save_trained_data_to_sync_server=True`, or upload "
                f"existing weights under `ml_weights/{data.case}/...` in the bucket."
            ),
            "problems_checked": problems_checked,
            "problems_downloaded": problems_downloaded,
            "problems_missing": problems_missing,
            "details": details,
        }

    return {
        "status": "ok",
        "case": data.case,
        "problems_checked": problems_checked,
        "problems_downloaded": problems_downloaded,
        "problems_missing": [],
        "details": details,
    }


