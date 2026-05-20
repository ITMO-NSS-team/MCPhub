import os
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

from fastmcp import FastMCP

from api_utils import (
    MLData,
    download_smiles_csv_from_s3,
    ensure_ml_weights_available,
    inference_ml,
    train_ml_with_data,
    upload_predictions_csv_to_s3,
)
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


def _normalize_s3_uri_to_key(s3_uri_or_key: Optional[str]) -> Optional[str]:
    """Normalize either `s3://bucket/key` or plain key to a bucket-relative key."""
    if not s3_uri_or_key:
        return None
    raw = s3_uri_or_key.strip()
    if not raw:
        return None
    if raw.lower().startswith("s3://"):
        without_scheme = raw[len("s3://"):]
        _, _, key_part = without_scheme.partition("/")
        normalized = key_part
    else:
        normalized = raw
    normalized = normalized.replace("\\", "/").lstrip("/")
    return normalized or None


def _read_smiles_from_local_csv(local_csv_path: str, smiles_column: str) -> List[str]:
    """Read SMILES strings from a column in a downloaded CSV file."""
    try:
        df = pd.read_csv(local_csv_path)
    except Exception:
        # Fallback for uncommon delimiters.
        df = pd.read_csv(local_csv_path, sep=None, engine="python")
    if smiles_column not in df.columns:
        raise ValueError(
            f"Column '{smiles_column}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )
    series = df[smiles_column].dropna().astype(str).str.strip()
    series = series[series != ""]
    return series.tolist()


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
def list_automl_train_cases(
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Lists S3 objects and resolves dataset names (`case_name`) for AutoML training.

    This is the AutoML (`automl-mcp`) server's view of the shared bucket. The
    Generative server exposes a functionally equivalent
    `list_generative_train_cases` against the same bucket — pick whichever
    server you are already talking to; the result is identical.

    Main purpose:
        Find training dataset files for the MCP server of generative and predictive molecular models.
        By default, this tool searches inside `train/` because `start_generative_model_training`
        and `train_ml` expect datasets at `train/{case_name}.csv`.

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
    train_data_url: str,
    target_column: Optional[List[str]] = None,
    feature_column: Optional[List[str]] = None,
    description: str = "Unknown case.",
    regression_props: Optional[List[str]] = None,
    classification_props: Optional[List[str]] = None,
    save_trained_data_to_sync_server: bool = True,
) -> dict[str, Any]:
    """Train AutoML pipelines for molecule property prediction by specific case.

    The training CSV is always supplied as an HTTP(S) URL (e.g. an S3 presigned
    URL), which the training backend fetches via plain `requests.get(...)`.
    No S3 credentials are required for the read, so the URL may point at any
    reachable endpoint / account.

    Trained model artifacts are uploaded to S3 under
    `ml_weights/{case}/trained_data_{case}_{problem}/...` by default
    (`save_trained_data_to_sync_server=True`). This makes it possible to spin
    up a fresh inference container and have `predict_ml` automatically
    download the weights on demand. Set the flag to `False` to keep weights
    only on the local filesystem of the training container.

    Training runs in a background process — the tool returns immediately with
    `job_id`. Use `train_ml_job_status` to poll status.

    Required: at least ONE of `regression_props` / `classification_props` must
    be supplied — these drive what is actually fitted. If both are omitted the
    job completes with `exitcode 0` and the case is marked `Trained`, but no
    Fedot pipeline is produced and subsequent `predict_ml` calls return
    `status: "no_predictable_properties"`. `target_column` alone is NOT enough
    — it is stored as metadata only and does not trigger training.

    Calculable properties are silently filtered out. Any property name that
    appears in the bundled calculable list (e.g. `LogP`, `QED`,
    `Synthetic Accessibility`, `PAINS`, `Brenk`, `Glaxo`, `SureChEMBL`,
    `Validity`, `Polar Surface Area`, `H-bond Donors`, `H-bond Acceptors`,
    `Rotatable Bonds`, `Aromatic Rings`) is removed from
    `regression_props` / `classification_props` before training, because
    these are computed directly from SMILES via RDKit at inference time and
    do not need a learned model. Pass only experimental / non-calculable
    targets (`docking_score`, `IC50`, `Ki`, `Minimum Energy`, custom assay
    columns, etc.). Use `check_state()` → `calc_propreties` to see the full
    bundled list.

    Args:
        case: Case identifier. Should be unique for each training dataset and
            will be used to reference the trained model for inference.
        train_data_url: Required HTTP(S) URL of the training CSV — typically
            an S3 presigned URL. The training server downloads it directly;
            the agent does not stream raw data through itself.
        target_column: Names of target columns from the CSV. Stored in state
            as metadata; does NOT by itself launch a training job. To fit a
            model, also pass the same names into `regression_props` or
            `classification_props`.
        feature_column: Feature column names, default is `['Smiles']`.
            Note: most curated datasets use `canonical_smiles`, not `Smiles`
            — verify against the dataset (e.g. `get_s3_train_case_columns`)
            and pass the exact column name if it differs.
        description: Free-text case description. Saved into state.json under
            the case entry. Has no effect on training itself; the agent
            should use it to record the task context (dataset origin,
            intended use, target rationale, dataset version) so future MCP
            calls can recall what this case was trained for.
        regression_props: Regression target columns. See "Required" /
            "Calculable properties" notes above.
        classification_props: Classification target columns. Same rules apply.
        save_trained_data_to_sync_server: If True (default), trained model
            artifacts are uploaded to S3 under
            `ml_weights/{case}/trained_data_{case}_{problem}/...` after
            training finishes.

    Returns:
        Dictionary with async start metadata:
        - `status`: `accepted`
        - `case`: case name
        - `job_id`: background training job id
        - `pid`: OS process id
        - `started_at`: UTC timestamp
        - `data_url`: training CSV URL submitted to the backend
        - `weights_s3_root`: S3 prefix where trained artifacts will be
          uploaded after training (always present, but actually written
          only when `save_trained_data_to_sync_server=True`).
        - `weights_s3_prefixes`: per-problem S3 prefixes that `predict_ml`
          looks at when downloading weights into a fresh container.
    """
    resolved_url = (train_data_url or "").strip()
    if not resolved_url:
        raise ValueError("train_data_url must not be empty")
    lowered = resolved_url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            "train_data_url must be an HTTP(S) URL (e.g. an S3 presigned URL); "
            f"got: {train_data_url!r}"
        )

    payload_data_raw: dict[str, Any] = {
        "case": case,
        "description": description,
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
        "data_url": resolved_url,
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

    expected_problems: List[str] = []
    if regression_props:
        expected_problems.append("regression")
    if classification_props:
        expected_problems.append("classification")

    weights_s3_root = f"ml_weights/{payload.case}/"
    weights_s3_prefixes = {
        problem: f"{weights_s3_root}trained_data_{payload.case}_{problem}/"
        for problem in expected_problems
    }

    result: Dict[str, Any] = {
        "status": "accepted",
        "case": payload.case,
        "job_id": job_id,
        "pid": process.pid,
        "started_at": started_at,
        "data_url": resolved_url,
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
        "weights_s3_root": weights_s3_root,
        "weights_s3_prefixes": weights_s3_prefixes,
    }
    return result


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
    smiles_list: Optional[List[str]] = None,
    input_s3_key: Optional[str] = None,
    smiles_column: str = "Smiles",
    upload_predictions_to_s3: bool = True,
    output_s3_prefix: str = "predictions",
    return_inline_predictions: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Run AutoML model inference for a given case to predict molecular properties.

    Input options (provide exactly one):
        - `smiles_list`: list of SMILES molecules.
        

    Output:
        Predictions are saved to S3 as a CSV under
        `{output_s3_prefix}/{case}/{uuid}.csv` and a presigned URL is returned
        in the response. The raw inline dict is omitted by default to keep
        the agent response small; set `return_inline_predictions=True` to also
        include the full predictions inline.

    Args:
        case: Trained case name.
        smiles_list: Optional inline SMILES list.
        smiles_column: Column name with SMILES when reading `input_s3_key`.
            Default `"Smiles"`.
        upload_predictions_to_s3: When True (default), upload the predictions
            CSV to S3 and return a presigned URL. When False, only inline
            predictions are returned (`return_inline_predictions` is forced
            to True in that case).
        output_s3_prefix: S3 prefix for uploaded predictions CSV.
            Default `"predictions"`. Final key:
            `{output_s3_prefix}/{case}/{uuid}.csv`.
        return_inline_predictions: When True, include the raw predictions
            dictionary alongside the S3 link.
        timeout: Optional timeout in minutes (passed through to MLData).

    Weights resolution:
        Before running inference the tool checks that trained pipelines for
        every predictable problem of the case are available locally. If they
        are missing it tries to download them from
        `s3://{bucket}/ml_weights/{case}/...` (where training puts them when
        called with `save_trained_data_to_sync_server=True`). If neither
        local cache nor S3 has the weights the tool returns a structured
        error response (`status="weights_not_found"` /
        `status="case_not_found"`) instead of raising.

    Returns:
        Dict with:
            - `status`: `ok` / `case_not_found` / `weights_not_found` /
              `no_predictable_properties` / `weights_load_failed` /
              `inference_failed`. The agent should branch on this.
            - `case`: trained case name.
            - `input_smiles_count`: number of input SMILES received.
            - On success (`status="ok"`):
                - `predicted_row_count`: rows in the predictions CSV.
                - `property_columns`: predicted property column names.
                - `weights_downloaded_from_s3`: list of problems whose weights
                  were freshly fetched from S3 during this call.
                - `predictions_s3_key`, `predictions_presigned_url`,
                  `expires_in`, `bucket_name`: present when
                  `upload_predictions_to_s3=True`.
                - `predictions`: raw dict, only when
                  `return_inline_predictions=True`.
            - On failure: `message`, `problems_missing`,
              `problems_downloaded`, `problems_checked`, `weights_details`.

    Raises:
        ValueError: if neither or both input forms are given, or if the SMILES
            column is missing in the input CSV.
    """
    if (smiles_list is None or len(smiles_list) == 0) and not input_s3_key:
        raise ValueError("Provide either `smiles_list` or `input_s3_key`.")
    if smiles_list and input_s3_key:
        raise ValueError("Provide only one of `smiles_list` or `input_s3_key`.")

    payload = MLData(case=case, timeout=timeout)

    if input_s3_key:
        normalized_in_key = _normalize_s3_uri_to_key(input_s3_key)
        if not normalized_in_key:
            raise ValueError(f"`input_s3_key` is invalid: {input_s3_key!r}")
        download_dir = IMPORT_PATH / "data" / "s3_inputs"
        download_dir.mkdir(parents=True, exist_ok=True)
        local_input = download_dir / f"{uuid4().hex}.csv"
        download_smiles_csv_from_s3(
            data=payload,
            s3_key_or_uri=normalized_in_key,
            local_csv_path=str(local_input),
        )
        resolved_smiles = _read_smiles_from_local_csv(
            local_csv_path=str(local_input),
            smiles_column=smiles_column,
        )
    else:
        resolved_smiles = [str(s).strip() for s in (smiles_list or []) if str(s).strip()]

    if not resolved_smiles:
        raise ValueError("No SMILES strings resolved from inputs.")

    payload.smiles_list = resolved_smiles
    _sync_state_from_s3()

    weights_status = ensure_ml_weights_available(payload)
    if weights_status.get("status") != "ok":
        return {
            "case": case,
            "status": weights_status.get("status", "weights_check_failed"),
            "message": weights_status.get(
                "message",
                f"Trained weights for case '{case}' are not available.",
            ),
            "input_smiles_count": len(resolved_smiles),
            "problems_checked": weights_status.get("problems_checked", []),
            "problems_downloaded": weights_status.get("problems_downloaded", []),
            "problems_missing": weights_status.get("problems_missing", []),
            "weights_details": weights_status.get("details", {}),
        }

    try:
        predictions = inference_ml(payload)
    except FileNotFoundError as exc:
        return {
            "case": case,
            "status": "weights_load_failed",
            "message": (
                "Pipeline files were located but could not be loaded. "
                "The cached/downloaded weights folder may be incomplete or corrupt. "
                f"Error: {exc}"
            ),
            "input_smiles_count": len(resolved_smiles),
            "weights_details": weights_status.get("details", {}),
        }
    except Exception as exc:
        return {
            "case": case,
            "status": "inference_failed",
            "message": f"Inference failed: {type(exc).__name__}: {exc}",
            "input_smiles_count": len(resolved_smiles),
            "weights_details": weights_status.get("details", {}),
        }

    if not isinstance(predictions, dict):
        # Defensive: keep the wrapping consistent even if downstream changes.
        predictions = {"value": list(predictions) if hasattr(predictions, "__iter__") else [predictions]}

    property_columns = list(predictions.keys())
    predicted_row_count = max((len(v) if hasattr(v, "__len__") else 0 for v in predictions.values()), default=0)

    result: Dict[str, Any] = {
        "case": case,
        "status": "ok",
        "input_smiles_count": len(resolved_smiles),
        "predicted_row_count": predicted_row_count,
        "property_columns": property_columns,
        "weights_downloaded_from_s3": weights_status.get("problems_downloaded", []),
    }

    effective_inline = return_inline_predictions or not upload_predictions_to_s3
    if effective_inline:
        result["predictions"] = predictions

    if upload_predictions_to_s3:
        normalized_prefix = (output_s3_prefix or "predictions").replace("\\", "/").strip("/")
        case_slug = (case or "case").replace("/", "_").replace("\\", "_").strip() or "case"
        filename = f"{uuid4().hex}.csv"
        s3_key = f"{normalized_prefix}/{case_slug}/{filename}"

        output_dir = IMPORT_PATH / "data" / "s3_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        local_output = output_dir / filename

        normalized_predictions = {col: list(vals) for col, vals in predictions.items()}
        if predicted_row_count > 0:
            for col, vals in normalized_predictions.items():
                if len(vals) < predicted_row_count:
                    vals.extend([None] * (predicted_row_count - len(vals)))
                elif len(vals) > predicted_row_count:
                    normalized_predictions[col] = vals[:predicted_row_count]
        df = pd.DataFrame(normalized_predictions)
        df.to_csv(local_output, index=False)

        upload_info = upload_predictions_csv_to_s3(
            data=payload,
            local_csv_path=str(local_output),
            s3_key=s3_key,
        )
        result.update({
            "bucket_name": upload_info["bucket_name"],
            "predictions_s3_key": upload_info["s3_key"],
            "predictions_presigned_url": upload_info["presigned_url"],
            "expires_in": upload_info["expires_in"],
        })

    return result


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MOLS_ML_MCP_PORT", "8777"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
