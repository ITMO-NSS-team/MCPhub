from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Union

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
    feature_column:  List[str] = ["Smiles"],
    epochs: int = 10,
    fine_tune: bool = True
) -> Dict[str, Any]:
    """
    Starts GAN generative training with existing train dataset in s3 database for a specific case.

    Args:
        case_name:
            Unique case identifier used to name and track the training run. It is the same name with existing train dataset in s3 database.
            Use `list_s3_train_cases` to get available values. Example: `Alzhmr`.
        feature_column:
            list with name of feature column in downloaded CSV.
            Example: `["Smiles"]`.
        epochs:
            Number of GAN training steps/epochs passed to server.
        fine_tune:
            Whether to fine-tune existing GAN weights.
        
    Returns:
        Dict[str, Any]:
            Metadata about submitted training request and server response.

    Raises:
        ValueError:
            If required fields are empty.
        RuntimeError:
            If request to training service fails.
    """
    case_name = case_name.strip()
    s3_key = f'train/{case_name}.csv'

    if not case_name:
        raise ValueError("case_name must not be empty")
    if not s3_key:
        raise ValueError("s3_key must not be empty")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")


    payload: Dict[str, Any] = {
        "case": case_name,
        "s3_key": s3_key,
        "feature_column": feature_column,
        "epochs": int(epochs),
        "fine_tune": bool(fine_tune),
    }

    response = _request_json(
        "POST",
        f"{GEN_BASE_URL}/train_gan",
        json=payload,
        timeout_s=_gan_http_timeout(),
    )

    result: Dict[str, Any] = {
        "case_name": case_name,
        "status": "training_started",
        "message": f"GAN training request sent for case '{case_name}'.",
        "s3_key": s3_key,
        "endpoint": f"{GEN_BASE_URL}/train_gan",
    }
    if response is not None:
        result["server_response"] = response
    return result


@mcp.tool()
def generate_mols(
    num: int = 10,
    case: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates drug molecules and calculated properties. It uses GAN-based generator that generate molecules fast without case specific training.
    But if you want to generate molecules from specific TRAINED case, you can use `case` parameter. In that case, it will use case-specific generator, that can generate molecules with properties similar to the ones in training data of that case.
    Args:
        num:
            Number of molecules requested.
            Default 10.
        case:
            Optional trained GAN case name. Used to generate molecules from that case by model, that trainded before.
            If omitted, server default case behavior is used.

    Returns:
        Dict[str, list]:
            Dictionary of column-like arrays returned by `gan_auto_generator`.
            Expected keys:
            - `Smiles`
            - `Brenk`, `QED`, `Synthetic Accessibility`, `LogP`,
              `Polar Surface Area`, `H-bond Donors`, `H-bond Acceptors`,
              `Rotatable Bonds`, `Aromatic Rings`, `Glaxo`, `SureChEMBL`, `PAINS`
            - `Validity`, `Duplicates`
            All value lists are aligned by index with `Smiles`.

    Raises:
        ValueError:
            If `num < 1`.
        RuntimeError:
            If generative endpoint is unavailable or returns invalid response format.
    """
    normalized_case = case.strip() if isinstance(case, str) else None
    if normalized_case == "":
        normalized_case = None
    return _post_generation_request("gan_case_generator", numb_mol=num, case=normalized_case)


def _generate_case_mols(endpoint: str, num: int) -> Dict[str, Any]:
    return _post_generation_request(endpoint, numb_mol=num)


@mcp.tool()
def generate_case_mols(case: str, num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for a selected disease case using case-specific generator.

    Supported cases: `skleroz`, `parkinson`, `cancer`, `dyslipidemia`, `drug_resist`, `alzheimer`.

    Args:
        case:
            Disease case selector (case-insensitive).
            Supported values:
            - `skleroz` -> multiple sclerosis endpoint.
            - `parkinson` -> Parkinson's disease endpoint.
            - `cancer` -> cancer endpoint.
            - `dyslipidemia` -> dyslipidemia endpoint.
            - `drug_resist` -> drug-resistance endpoint.
            - `alzheimer` -> Alzheimer's disease endpoint.
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`,.

    Raises:
        ValueError:
            If case is not supported.
    """
    case_key = (case or "").strip().lower()
    endpoint = CASE_ENDPOINTS.get(case_key)
    if endpoint is None:
        supported = ", ".join(sorted(CASE_ENDPOINTS.keys()))
        raise ValueError(f"Unsupported case '{case}'. Supported cases: {supported}")
    return _generate_case_mols(endpoint, num)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MOLS_GEN_MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
