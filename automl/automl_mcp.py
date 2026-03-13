import os
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

from fastmcp import FastMCP

from api_utils import MLData, inference_ml, train_ml_with_data
from utils.base_state import TrainState

try:
    from utils.state_s3 import download_state_file
except ModuleNotFoundError:
    from .utils.state_s3 import download_state_file

try:
    from s3_utils import S3BucketService, s3_service as default_s3_service
except ModuleNotFoundError:
    from .s3_utils import S3BucketService, s3_service as default_s3_service

load_dotenv()

IMPORT_PATH = Path(__file__).resolve().parent
STATE_FILE = "state.json"

mcp = FastMCP("automl-mcp")
_TRAIN_JOBS: Dict[str, Process] = {}
_TRAIN_JOB_META: Dict[str, Dict[str, Any]] = {}


def _normalize_s3_prefix(prefix: str) -> str:
    normalized = (prefix or "").replace("\\", "/").strip("/")
    return f"{normalized}/" if normalized else ""


def _normalize_s3_extension(extension: str) -> str:
    if extension is None:
        return ""
    normalized = extension.strip()
    if not normalized:
        return ""
    return normalized if normalized.startswith(".") else f".{normalized}"


def _build_s3_service(
    *,
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> S3BucketService:
    endpoint = endpoint_url or os.getenv("ENDPOINT_URL") or default_s3_service.endpoint
    access = access_key or os.getenv("ACCESS_KEY") or default_s3_service.access_key
    secret = secret_key or os.getenv("SECRET_KEY") or default_s3_service.secret_key
    bucket = bucket_name or os.getenv("BUCKET_NAME") or default_s3_service.bucket_name

    if not endpoint:
        raise ValueError("S3 endpoint is empty. Set `ENDPOINT_URL` or pass `endpoint_url`.")
    if not access:
        raise ValueError("S3 access key is empty. Set `ACCESS_KEY` or pass `access_key`.")
    if not secret:
        raise ValueError("S3 secret key is empty. Set `SECRET_KEY` or pass `secret_key`.")
    if not bucket:
        raise ValueError("S3 bucket name is empty. Set `BUCKET_NAME` or pass `bucket_name`.")

    return S3BucketService(
        endpoint=endpoint,
        access_key=access,
        secret_key=secret,
        bucket_name=bucket,
    )


def _extract_case_name_from_s3_key(s3_key: str, *, prefix: str, extension: str) -> Optional[str]:
    normalized_key = (s3_key or "").replace("\\", "/").strip("/")
    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension).lower()

    if normalized_prefix and not normalized_key.startswith(normalized_prefix):
        return None

    relative_key = normalized_key[len(normalized_prefix) :] if normalized_prefix else normalized_key
    if not relative_key or relative_key.endswith("/"):
        return None

    # Expected format for training datasets: train/{case_name}.csv
    if "/" in relative_key:
        return None

    if normalized_extension:
        if not relative_key.lower().endswith(normalized_extension):
            return None
        return relative_key[: -len(normalized_extension)]

    return relative_key


def _sync_state_from_s3() -> str:
    """Download the latest shared `state.json` from S3 storage.

    Returns:
        Absolute path to downloaded file in local `automl` directory.
    """
    state_path = IMPORT_PATH / STATE_FILE
    download_state_file(local_path=str(state_path))
    return str(state_path)


def _ml_data_to_dict(payload: MLData) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)
    return payload.dict(exclude_none=True)


def _train_ml_worker(payload_data: dict[str, Any]) -> None:
    """Background worker that syncs state and runs training."""
    payload = MLData(**payload_data)
    _sync_state_from_s3()
    train_ml_with_data(payload)


@mcp.tool()
def list_s3_train_cases(
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Lists S3 objects and resolves dataset names (`case_name`) for MCP training workflows.

    Main purpose:
        Find training dataset files for the MCP server of generative and predictive molecular models.
        By default, this tool searches inside `train/` because `start_generative_model_training`
        expects datasets at `train/{case_name}.csv`.

    Prefix behavior:
        - Default (`prefix="train/"`): standard mode for training dataset discovery.
        - Global search (`prefix=""`): lists all objects in the bucket.
          This can be useful to inspect bucket structure (for example, understand available
          folders first), and then run a narrower search for train files.

    Args:
        prefix:
            S3 prefix (folder) where train datasets are stored.
            Default: `train/`.
        extension:
            File extension for train datasets.
            Default: `.csv`.

    Returns:
        Dict[str, Any]:
            - `bucket_name`: resolved bucket name.
            - `prefix`: normalized prefix used for listing.
            - `extension`: normalized extension filter.
            - `total_train_files`: count of files matching expected train format.
            - `s3_keys`: matching S3 object keys.
            - `case_names`: normalized case names (without prefix and extension) for `case_name`.
    """
    s3_service = _build_s3_service()

    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension)
    keys = s3_service.list_objects(prefix=normalized_prefix)

    filtered_keys: List[str] = []
    case_names: List[str] = []
    seen_cases: Set[str] = set()

    for key in keys:
        case_name = _extract_case_name_from_s3_key(
            key,
            prefix=normalized_prefix,
            extension=normalized_extension,
        )
        if not case_name:
            continue

        normalized_key = key.replace("\\", "/").strip("/")
        filtered_keys.append(normalized_key)
        if case_name not in seen_cases:
            seen_cases.add(case_name)
            case_names.append(case_name)

    filtered_keys.sort()
    case_names.sort()

    return {
        "bucket_name": s3_service.bucket_name,
        "prefix": normalized_prefix,
        "total_train_files": len(filtered_keys),
        "s3_keys": filtered_keys,
        "case_names": case_names
       
    }


@mcp.tool()
def get_s3_train_case_columns(
    case_name: str,
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Download train dataset by case from S3 and return CSV column names.

    Useful when dataset file is known but target/feature columns are unknown.
    """
    normalized_case = (case_name or "").strip()
    if not normalized_case:
        raise ValueError("`case_name` must not be empty.")
    if "/" in normalized_case or "\\" in normalized_case:
        raise ValueError("`case_name` must not contain path separators.")

    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension) or ".csv"
    s3_key = f"{normalized_prefix}{normalized_case}{normalized_extension}".replace("//", "/")

    s3_service = _build_s3_service()
    preview_dir = IMPORT_PATH / "data" / "s3_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    local_path = preview_dir / f"{normalized_case}{normalized_extension}"

    try:
        s3_service.download_image_from_s3(s3_key=s3_key, local_path=str(local_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset from S3 key '{s3_key}': {exc}") from exc

    try:
        df = pd.read_csv(local_path, nrows=0)
    except Exception:
        # Fallback for uncommon delimiters.
        df = pd.read_csv(local_path, nrows=0, sep=None, engine="python")

    columns = [str(col) for col in df.columns.tolist()]
    return {
        "case_name": normalized_case,
        "s3_key": s3_key,
        "column_count": len(columns),
        "columns": columns,
    }


@mcp.tool()
def health_check() -> dict[str, str]:
    """Liveness probe for the AutoML MCP server.

    Returns:
        Dictionary with fixed key/value: `{\"health_check\": \"ok\"}`.
    """
    return {"health_check": "ok"}


@mcp.tool()
def check_state() -> dict[str, Any]:
    """Get current training registry and available calculable properties.

    State is read from local `state.json`.

    Returns:
        Dictionary with fields:
        - `state`: all registered cases and their metadata, excluding the internal
          `Calculateble properties` section.
        - `calc_propreties`: list of property names that are computed directly and
          therefore do not require ML model training.
    """
    state = TrainState(state_path=str(IMPORT_PATH / STATE_FILE))
    calc_properties = state.show_calculateble_propreties()
    current_state = state().copy()
    current_state.pop("Calculateble properties", None)
    return {"state": current_state, "calc_propreties": list(calc_properties)}


@mcp.tool()
def train_ml(
    case: str,
    target_column: Optional[List[str]] = None,
    feature_column: Optional[List[str]] = None,
    description: str = "Unknown case.",
    regression_props: Optional[List[str]] = None,
    classification_props: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Train AutoML pipelines for molecule property prediction by specific case.

    Training may take from minutes to hours depending on dataset size and resources.
    The tool starts training in a background process and returns immediately with `job_id`.

    Args:
        case: Case identifier. Should be unique for each training dataset and will be used to reference the trained model for inference. and it is similar to trian dataset name.
        target_column: Target column names for prediction.
        feature_column: Feature column names, default is `['Smiles']`.
        description: Case description.
        regression_props: Regression targets.
        classification_props: Classification targets.


    Returns:
        Dictionary with async start metadata:
        - `status`: `accepted`
        - `case`: case name
        - `job_id`: background training job id
        - `pid`: OS process id
        - `started_at`: UTC timestamp
    """
    payload_data_raw: dict[str, Any] = {
        "case": case,
        "description": description,
        
    }
    if target_column is not None:
        payload_data_raw["target_column"] = target_column
    if feature_column is not None:
        payload_data_raw["feature_column"] = feature_column
    if regression_props is not None:
        payload_data_raw["regression_props"] = regression_props
    if classification_props is not None:
        payload_data_raw["classification_props"] = classification_props

    payload = MLData(**payload_data_raw)
    payload_data = _ml_data_to_dict(payload)

    job_id = uuid4().hex
    process = Process(target=_train_ml_worker, args=(payload_data,), daemon=False)
    process.start()

    started_at = datetime.now(timezone.utc).isoformat()
    _TRAIN_JOBS[job_id] = process
    _TRAIN_JOB_META[job_id] = {
        "case": payload.case,
        "pid": process.pid,
        "started_at": started_at,
    }

    return {
        "status": "accepted",
        "case": payload.case,
        "job_id": job_id,
        "pid": process.pid,
        "started_at": started_at,
    }


@mcp.tool()
def train_ml_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a background training job started by `train_ml`.

    For model-level status/metrics use `check_state` with the corresponding case.
    """
    meta = _TRAIN_JOB_META.get(job_id)
    process = _TRAIN_JOBS.get(job_id)

    if meta is None or process is None:
        return {"job_id": job_id, "status": "not_found"}

    alive = process.is_alive()
    exitcode = process.exitcode
    if alive:
        status = "running"
    elif exitcode == 0:
        status = "finished"
    elif exitcode is None:
        status = "unknown"
    else:
        status = "failed"

    return {
        "job_id": job_id,
        "status": status,
        "case": meta["case"],
        "pid": meta["pid"],
        "started_at": meta["started_at"],
        "exitcode": exitcode,
    }


@mcp.tool()
def predict_ml(
    case: str,
    smiles_list: List[str],
    timeout: int = 30,
) -> Any:
    """Run AutoML inference for SMILES list in an existing case.

    Runs inference for a list of SMILES strings in a selected case. Before
    inference, refreshes shared `state.json` from S3.

    Args:
        case: Trained case name.
        smiles_list: Molecules in SMILES format.
        timeout: Optional timeout in minutes.

    Returns:
        Inference result from `inference_ml(...)`. The structure depends on the
        trained case and available predictors.

    Raises:
        pydantic.ValidationError: if payload format does not match `MLData`.
        Any exception propagated by inference or S3 sync logic.
    """
    payload = MLData(case=case, smiles_list=smiles_list, timeout=timeout)
    _sync_state_from_s3()
    return inference_ml(payload)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MOLS_ML_MCP_PORT", "8777"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
