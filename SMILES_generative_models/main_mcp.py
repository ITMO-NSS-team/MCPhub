from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

import pandas as pd
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
mcp = FastMCP(name="GenerativeModelsMCP")


def _normalize_base_url(base: str) -> str:
    base = base.strip().rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    return base


def _build_base_url(base_env: str, default_port: str) -> str:
    explicit = os.getenv(base_env)
    if explicit:
        return _normalize_base_url(explicit)
    # Keep localhost default for MCP-to-local FastAPI communication.
    return f"http://localhost:{default_port}"


DEFAULT_API_PORT = str((os.getenv("GEN_MOLS_MODEL_APP_PORT") or "8000")).strip()
PRED_BASE_URL = _build_base_url(base_env="ML_TOOLS_BASE_URL", default_port=DEFAULT_API_PORT)
GEN_BASE_URL = _build_base_url(base_env="DL_TOOLS_BASE_URL", default_port=DEFAULT_API_PORT)

# Properties returned by /gan_case_generator (gan_auto_generator).
GAN_GENERATED_PROPERTIES = [
    "Smiles",
    "Brenk",
    "QED",
    "Synthetic Accessibility",
    "LogP",
    "Polar Surface Area",
    "H-bond Donors",
    "H-bond Acceptors",
    "Rotatable Bonds",
    "Aromatic Rings",
    "Glaxo",
    "SureChEMBL",
    "PAINS",
    "Validity",
    "Duplicates",
]

# Base properties returned by disease-specific generation endpoints (case_generator).
CASE_GENERATED_PROPERTIES = [
    "Molecules",
    "QED",
    "Synthetic Accessibility",
    "PAINS",
    "SureChEMBL",
    "Glaxo",
    "Brenk",
    "BBB",
    "IC50",
    "KI",
]

CASE_ENDPOINTS: Dict[str, str] = {
    "skleroz": "search_Skleroz",
    "parkinson": "search_Parkinson",
    "cancer": "search_Canser",
    "dyslipidemia": "search_Dyslipidemia",
    "drug_resist": "search_Drug_resist",
    "alzheimer": "search_Alzheimer",
}


def _resolve_base_url(selector: str) -> str:
    key = (selector or "").strip().lower()
    if key in {"pred", "ml", "prediction"}:
        return PRED_BASE_URL
    if key in {"gen", "dl", "generative"}:
        return GEN_BASE_URL
    return _normalize_base_url(selector)


def _parse_response_json(resp: requests.Response) -> Any:
    data = resp.json()
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def _request_json(
    method: str,
    url: str,
    *,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    try:
        resp = requests.request(method, url, timeout=timeout_s, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    return _parse_response_json(resp)


# def _predict_http_timeout(timeout_minutes: int) -> int:
#     return max(30, int(timeout_minutes) * 60 + 30)


def _gan_http_timeout() -> int:
    return int(os.getenv("GAN_HTTP_TIMEOUT_S", "1160"))


def _upload_dict_as_csv_to_s3(
    data_dict: Dict[str, Any],
    s3_key: str,
    expiration: int = 3600,
) -> Dict[str, Any]:
    """Save `data_dict` (column -> list) as a CSV and upload to S3.

    Returns metadata: bucket_name, s3_key, presigned_url, expires_in, rows.
    """
    s3_service = _build_s3_service()

    normalized_key = s3_key.replace("\\", "/").lstrip("/")
    if "/" in normalized_key:
        prefix, source_file_name = normalized_key.rsplit("/", 1)
    else:
        prefix, source_file_name = "", normalized_key

    columns = list(data_dict.keys())
    max_len = 0
    for col in columns:
        values = data_dict[col]
        if hasattr(values, "__len__"):
            max_len = max(max_len, len(values))

    normalized = {}
    for col in columns:
        values = list(data_dict[col]) if hasattr(data_dict[col], "__iter__") and not isinstance(data_dict[col], (str, bytes)) else [data_dict[col]]
        if len(values) < max_len:
            values = values + [None] * (max_len - len(values))
        elif len(values) > max_len:
            values = values[:max_len]
        normalized[col] = values

    df = pd.DataFrame(normalized)

    output_dir = Path(__file__).resolve().parent / "data" / "s3_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / source_file_name
    df.to_csv(local_path, index=False)

    s3_service.upload_file_object(
        prefix=prefix,
        source_file_name=source_file_name,
        file_path=str(local_path),
    )
    presigned_url = s3_service.generate_presigned_url(
        s3_key=normalized_key, expiration=expiration
    )
    return {
        "bucket_name": s3_service.bucket_name,
        "s3_key": normalized_key,
        "presigned_url": presigned_url,
        "expires_in": expiration,
        "rows": max_len,
        "columns": columns,
    }


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
):
    try:
        from train_data.utils.s3_utils import S3BucketService, s3_service as default_s3_service
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "S3 dependencies are missing. Install boto3 (for example via requirements.txt / requirements.mcp.txt)."
        ) from exc

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

    # `start_generative_model_training` expects key format: train/{case_name}.csv
    if "/" in relative_key:
        return None

    if normalized_extension:
        if not relative_key.lower().endswith(normalized_extension):
            return None
        return relative_key[: -len(normalized_extension)]

    return relative_key


def _validate_numb_mol(numb_mol: int) -> int:
    if int(numb_mol) < 1:
        raise ValueError("numb_mol must be >= 1")
    return int(numb_mol)


def _post_generation_request(
    endpoint: str,
    *,
    numb_mol: int,
    case: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"numb_mol": _validate_numb_mol(numb_mol)}
    if case is not None:
        payload["case_"] = case

    data = _request_json(
        "POST",
        f"{GEN_BASE_URL}/{endpoint.lstrip('/')}",
        json=payload,
        timeout_s=_gan_http_timeout(),
    )
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response format from {endpoint}: {type(data).__name__}")
    return data


def _summarize_generation_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a small dict (counts / property column names) without inline arrays."""
    columns = list(raw_result.keys()) if isinstance(raw_result, dict) else []
    smiles_key = None
    for candidate in ("Smiles", "Molecules"):
        if candidate in columns:
            smiles_key = candidate
            break
    generated_count = 0
    if smiles_key is not None and hasattr(raw_result.get(smiles_key), "__len__"):
        generated_count = len(raw_result[smiles_key])
    else:
        for col in columns:
            value = raw_result.get(col)
            if hasattr(value, "__len__"):
                generated_count = max(generated_count, len(value))
    return {"generated_count": generated_count, "columns": columns}


_GAN_ERROR_STATUSES = {
    "case_not_found",
    "case_not_trained",
    "weights_not_found",
    "weights_load_failed",
}


def _maybe_upload_generation_result(
    *,
    raw_result: Dict[str, Any],
    case: str,
    requested_count: int,
    upload_results_to_s3: bool,
    output_s3_prefix: str,
    return_inline_results: bool,
) -> Dict[str, Any]:
    """Wrap a raw generation dict into an MCP response, optionally uploading CSV to S3.

    If the FastAPI side reports a structured error (case not trained, weights
    missing in S3, weights file corrupt) — propagate it as-is so the agent can
    branch on `status` instead of treating noise as molecules.
    """
    case_slug = (case or "gan_default").replace("/", "_").replace("\\", "_").strip() or "gan_default"

    if isinstance(raw_result, dict) and raw_result.get("status") in _GAN_ERROR_STATUSES:
        return {
            "case": case_slug,
            "requested_count": int(requested_count),
            **raw_result,
        }

    summary = _summarize_generation_result(raw_result)

    response: Dict[str, Any] = {
        "case": case_slug,
        "status": "ok",
        "requested_count": int(requested_count),
        **summary,
    }

    effective_inline = return_inline_results or not upload_results_to_s3
    if effective_inline:
        response["results"] = raw_result

    if upload_results_to_s3 and isinstance(raw_result, dict) and raw_result:
        normalized_prefix = (output_s3_prefix or "generated").replace("\\", "/").strip("/")
        filename = f"{uuid4().hex}.csv"
        s3_key = f"{normalized_prefix}/{case_slug}/{filename}"
        try:
            upload_info = _upload_dict_as_csv_to_s3(data_dict=raw_result, s3_key=s3_key)
            response.update({
                "bucket_name": upload_info["bucket_name"],
                "results_s3_key": upload_info["s3_key"],
                "results_presigned_url": upload_info["presigned_url"],
                "expires_in": upload_info["expires_in"],
            })
        except Exception as exc:
            response["s3_upload_error"] = f"{type(exc).__name__}: {exc}"
            if not effective_inline:
                # Fall back to inline so the agent does not lose the data.
                response["results"] = raw_result

    return response


@mcp.tool()
def list_generative_train_cases(
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Lists S3 objects and resolves dataset names (`case_name`) for GAN training.

    This is the Generative (`GenerativeModelsMCP`) server's view of the shared
    bucket. The AutoML server exposes a functionally equivalent
    `list_automl_train_cases` against the same bucket — pick whichever server
    you are already talking to; the result is identical.

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
def get_state_from_server(url: str = "gen", case: Optional[str] = None) -> Union[dict, str]:
    """
    Returns model registry state from prediction or generative API server.
    This state discribes available cases, their training status, and metadata. It is used by MCP clients to discover cases and track training progress.

    Args:
        url:
            - "pred" -> use prediction base URL.
            - "gen" -> use generative base URL.
            - any other value is treated as explicit base URL.
        case:
            Optional case key. If provided, returns only this case entry.

    Returns:
        Union[dict, str]:
            - Full state dictionary when `case` is None.
            - Single case dictionary when `case` exists.
            - Error string for unavailable server or unknown case.

    Notes:
        Calls `GET /check_state` on the selected server and parses both JSON
        and JSON-in-string responses.
    """
    base_url = _resolve_base_url(url)
    resp = requests.get(f"{base_url}/check_state", timeout=30)
    if resp.status_code == 500:
        return "Server error"
    resp.raise_for_status()

    data = _parse_response_json(resp)
    if not isinstance(data, dict):
        return data

    state = data.get("state", data)
    if case:
        return state.get(case, f"Case: {case} not found")
    return state



@mcp.tool()
def start_generative_model_training(
    case_name: str,
    train_data_url: str,
    feature_column: List[str] = ["Smiles"],
    epochs: int = 10,
    fine_tune: bool = True,
    save_trained_data_to_sync_server: bool = True,
) -> Dict[str, Any]:
    """
    Starts GAN generative training. The training CSV is always supplied as an
    HTTP(S) URL (e.g. an S3 presigned URL), which the training backend fetches
    via plain `requests.get(...)`. No S3 credentials are required for the read,
    so the URL may point at any reachable endpoint / account.

    BLOCKING call. Unlike `train_ml` (which spawns a background process and
    returns a `job_id` immediately), this tool issues a synchronous HTTP
    request to the FastAPI backend and waits for training to finish — up to
    `GAN_HTTP_TIMEOUT_S` seconds (default ~19 min). There is no `job_id` and
    nothing to poll; success/failure is reported in the response. Plan
    surrounding tool calls accordingly.

    Trained weights are uploaded to S3 under
    `gan_weights/{case_name}/train_GAN_{case_name}/...` by default
    (`save_trained_data_to_sync_server=True`). A fresh inference container
    will then auto-download them on the first `generate_mols(case=case_name)`.

    Args:
        case_name:
            Unique case identifier used to name and track the training run.
            Use `list_s3_train_cases` to discover available datasets.
        train_data_url:
            Required HTTP(S) URL of the training CSV — typically an S3
            presigned URL. The training server downloads it directly; the
            agent does not stream raw data through itself.
        feature_column:
            Name of the SMILES column in the downloaded CSV.
            Example: `["canonical_smiles"]`.
        epochs:
            Number of GAN training steps/epochs.
        fine_tune:
            Whether to fine-tune existing GAN weights.
        save_trained_data_to_sync_server:
            If True (default), trained GAN weights are uploaded to S3 under
            `gan_weights/{case_name}/train_GAN_{case_name}/...` after
            training finishes.

    Returns:
        Dict[str, Any]:
            Metadata about the submitted training request — `case_name`,
            `data_url`, `weights_s3_prefix`, plus the backend response.

    Raises:
        ValueError: If required fields are empty / malformed.
        RuntimeError: If the request to the training service fails.
    """
    case_name = (case_name or "").strip()
    if not case_name:
        raise ValueError("case_name must not be empty")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")

    resolved_url = (train_data_url or "").strip()
    if not resolved_url:
        raise ValueError("train_data_url must not be empty")
    lowered = resolved_url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            "train_data_url must be an HTTP(S) URL (e.g. an S3 presigned URL); "
            f"got: {train_data_url!r}"
        )

    payload: Dict[str, Any] = {
        "case": case_name,
        "data_url": resolved_url,
        "feature_column": feature_column,
        "epochs": int(epochs),
        "fine_tune": bool(fine_tune),
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
    }

    response = _request_json(
        "POST",
        f"{GEN_BASE_URL}/train_gan",
        json=payload,
        timeout_s=_gan_http_timeout(),
    )

    weights_s3_prefix = (
        f"gan_weights/{case_name}/train_GAN_{case_name}/"
        if save_trained_data_to_sync_server
        else None
    )

    result: Dict[str, Any] = {
        "case_name": case_name,
        "status": "training_started",
        "message": f"GAN training request sent for case '{case_name}'.",
        "data_url": resolved_url,
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
        "weights_s3_prefix": weights_s3_prefix,
        "endpoint": f"{GEN_BASE_URL}/train_gan",
    }
    if response is not None:
        result["server_response"] = response
    return result


@mcp.tool()
def generate_mols(
    num: int = 10,
    case: Optional[str] = None,
    upload_results_to_s3: bool = True,
    output_s3_prefix: str = "generated",
    return_inline_results: bool = False,
) -> Dict[str, Any]:
    """
    Generic, FAST GAN molecule generator. Default tool for "give me N drug-like
    molecules" requests when no disease-specific tuning is required. Its generate good molecules fast, when training is timeconsuming or unavailable — for example, when the agent is exploring a new case and has not trained a GAN yet or has no train data.

    Model: GAN-LSTM (`gan_auto_generator` on the FastAPI side). One forward
    pass per batch — no iterative novelty/property filtering loop, no docking,
    no IC50 evaluation. Returns the SMILES plus a fixed set of RDKit-computed
    properties (`QED`, `LogP`, `Synthetic Accessibility`, `Brenk`, `PAINS`,
    `Glaxo`, `SureChEMBL`, `Polar Surface Area`, H-bond/Rotatable/Aromatic
    counters, `Validity`, `Duplicates`).

    Two operating modes:

    1. Generic mode (`case` omitted) — uses the bundled fallback GAN weights
       shipped with the image (`GAN/gan_lstm_refactoring/weights/
       v4_gan_mol_124_0.0003_8k.pkl`, pulled from HuggingFace at build time).
       No S3 lookup, no case state required. This is the right tool for any
       case that does NOT already have a fine-tuned GAN.

    2. Case-specific mode (`case` provided) — STRICT: weights MUST be the
       case-specific ones produced by `start_generative_model_training`. If
       missing locally, the FastAPI side tries to download
       `gan_weights/{case}/train_GAN_{case}/...` from S3. If neither local
       cache nor S3 has them, the response carries
       `status="case_not_trained" / "weights_not_found" / "weights_load_failed"`
       — there is NO silent fallback to the generic GAN.

    Typical agent workflow: `generate_mols(num=...)` → pipe SMILES into
    `predict_ml(case=..., smiles_list=...)` for property prediction. If the
    results are insufficient, request more molecules — no retraining needed.

    For the 6 hard-coded disease cases (alzheimer, parkinson, cancer,
    skleroz, dyslipidemia, drug_resist) prefer `generate_case_mols` — it
    runs a different, slower model (multi-property CVAE) tuned for those
    targets with property constraints. Do NOT use those names here unless
    you have separately trained a GAN under that case name.

    Output contract:
        By default the full generated table (SMILES + calculated properties)
        is saved to S3 as a CSV under
        `{output_s3_prefix}/{case_or_gan_default}/{uuid}.csv` and a presigned
        URL is returned. The raw inline arrays are omitted to keep the agent
        response small; set `return_inline_results=True` to also include them.

    Args:
        num: Number of molecules requested. Default 10. No hard cap.
        case: Optional trained-GAN case name (see "Case-specific mode" above).
            Omit for the generic GAN.
        upload_results_to_s3: When True (default), upload the CSV with
            generated molecules to S3 and return a presigned URL.
        output_s3_prefix: S3 prefix for the uploaded CSV. Default `generated`.
            Final key: `{output_s3_prefix}/{case_or_gan_default}/{uuid}.csv`.
        return_inline_results: When True, include the raw dict of arrays in
            the response alongside the S3 link.

    Returns:
        Dict[str, Any]:
            - `status`: `ok` / `case_not_found` / `case_not_trained` /
              `weights_not_found` / `weights_load_failed`. The agent should
              branch on this.
            - `case`: case used (or `gan_default`).
            - `requested_count`: value of `num`.
            - On `status="ok"`: `generated_count`, `columns`,
              `results_s3_key`, `results_presigned_url`, `expires_in`,
              `bucket_name` (when `upload_results_to_s3=True`); `results`
              (raw dict) when `return_inline_results=True`.
            - On error: `message`, `s3_prefix`, `weights_dir` (where present).

    Raises:
        ValueError: If `num < 1`.
        RuntimeError: If generative endpoint is unavailable or returns
            invalid response format.
    """
    normalized_case = case.strip() if isinstance(case, str) else None
    if normalized_case == "":
        normalized_case = None

    raw = _post_generation_request("gan_case_generator", numb_mol=num, case=normalized_case)
    return _maybe_upload_generation_result(
        raw_result=raw,
        case=normalized_case or "gan_default",
        requested_count=num,
        upload_results_to_s3=upload_results_to_s3,
        output_s3_prefix=output_s3_prefix,
        return_inline_results=return_inline_results,
    )


def _generate_case_mols(endpoint: str, num: int) -> Dict[str, Any]:
    return _post_generation_request(endpoint, numb_mol=num)


@mcp.tool()
def generate_case_mols(
    case: str,
    num: int = 10,
    upload_results_to_s3: bool = True,
    output_s3_prefix: str = "generated",
    return_inline_results: bool = False,
) -> Dict[str, Any]:
    """
    Generate molecules for a HARDCODED disease case using a multi-property
    conditional VAE — SLOWER but tuned for specific therapeutic targets.

    Model: 8-property conditional CVAE (`multi_generator` on the FastAPI side
    invoked through one of the `/search_*` endpoints) with case-specific
    pre-bundled weights:

        alzheimer    -> autotrain/many_prop_CVAE/weights_8p_alzhmr/
        skleroz      -> autotrain/many_prop_CVAE/weights_8p_sklrz/
        cancer       -> autotrain/many_prop_CVAE/weights_8p_cnsr/
        parkinson    -> autotrain/many_prop_CVAE/weights_parkinson/
        dyslipidemia -> autotrain/many_prop_CVAE/weights_dislip/
        drug_resist  -> autotrain/many_prop_CVAE/weights_8p_tablet/

    Unlike `generate_mols`, the backend applies property constraints
    (`spec_conds` — windows on docking_score, QED, Synthetic Accessibility,
    PAINS/SureChEMBL/Glaxo/Brenk flags) and ITERATIVELY regenerates until
    enough valid + novel molecules accumulate. Novelty is checked against the
    disease-specific ChEMBL training set
    (`docked_data_for_train/data_*.csv`). Because of this loop, latency is
    significantly higher than `generate_mols` — expect tens of seconds to a
    few minutes for a full batch.

    When to use this tool vs `generate_mols`:
        - Use `generate_case_mols` ONLY for the 6 cases listed above and only
          when you need property-constrained, case-tuned molecules (typical
          for lead-discovery / SAR workflows).
        - For any other target, fast generic generation, or when you just
          need raw SMILES to feed into `predict_ml`, use `generate_mols`
          (universal GAN, no constraints, no per-case weights required).

    Hard cap: the FastAPI side enforces `numb_mol <= 100` per request — any
    larger value is silently clamped to 100.

    Output contract:
        By default the full generated table is saved to S3 as a CSV under
        `{output_s3_prefix}/{case}/{uuid}.csv` and a presigned URL is returned.
        The raw inline arrays are omitted to keep the agent response small;
        set `return_inline_results=True` to also include them.

    Args:
        case:
            Disease case selector (case-insensitive). Supported values:
            - `skleroz` -> multiple sclerosis (BTK target).
            - `parkinson` -> Parkinson's disease (tyrosine-protein kinase ABL).
            - `cancer` -> cancer (8afb protein).
            - `dyslipidemia` -> dyslipidemia (ATP citrate synthase).
            - `drug_resist` -> drug-resistance ("tablet" weights).
            - `alzheimer` -> Alzheimer's disease (4j1r target).
        num:
            Number of molecules requested. Default `10`. Backend hard-caps
            at 100 — larger values are silently truncated.
        upload_results_to_s3: When True (default), upload CSV with generated
            molecules to S3 and return a presigned URL.
        output_s3_prefix: S3 prefix for the uploaded CSV. Default `generated`.
        return_inline_results: When True, include the raw dict alongside the
            S3 link.

    Returns:
        Dict[str, Any]:
            - `case`: case slug used in the upload path.
            - `requested_count`: value of `num`.
            - `generated_count`, `columns`: high-level summary.
            - `results_s3_key`, `results_presigned_url`, `expires_in`,
              `bucket_name`: present when `upload_results_to_s3=True`.
            - `results`: raw inline dict, only when `return_inline_results=True`.

    Raises:
        ValueError: If case is not supported.
    """
    case_key = (case or "").strip().lower()
    endpoint = CASE_ENDPOINTS.get(case_key)
    if endpoint is None:
        supported = ", ".join(sorted(CASE_ENDPOINTS.keys()))
        raise ValueError(f"Unsupported case '{case}'. Supported cases: {supported}")
    raw = _generate_case_mols(endpoint, num)
    return _maybe_upload_generation_result(
        raw_result=raw,
        case=case_key,
        requested_count=num,
        upload_results_to_s3=upload_results_to_s3,
        output_s3_prefix=output_s3_prefix,
        return_inline_results=return_inline_results,
    )


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MOLS_GEN_MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
