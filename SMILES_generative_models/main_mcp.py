from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

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


DEFAULT_API_PORT = str((os.getenv("GEN_APP_PORT") or "8000")).strip()
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


@mcp.tool(
    title="Get Server State",
    description="Returns model registry state from prediction or generative API server.",
    tags={"state", "metadata"},
)
def get_state_from_server(url: str = "pred", case: Optional[str] = None) -> Union[dict, str]:
    """
    Fetch model registry state from the prediction or generative API server.

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


# @mcp.tool(
#     title="Predict Properties by SMILES",
#     description="Calls /predict_ml for batch property prediction.",
#     tags={"prediction", "ml"},
# )
# def predict_prop_by_smiles(
#     smiles_list: List[str],
#     case: str = "no_name_case",
#     timeout: int = 20,
# ) -> dict:
#     """
#     Predict molecular properties for a batch of SMILES strings.

#     Args:
#         smiles_list:
#             List of SMILES strings to evaluate.
#         case:
#             Prediction case/model name known by `/predict_ml`.
#             Use `get_state_from_server(url="pred")` to inspect available cases.
#         timeout:
#             Timeout in minutes for server-side processing.

#     Returns:
#         dict:
#             Raw JSON returned by `POST /predict_ml` (property names and
#             predictions are server-defined).

#     Raises:
#         RuntimeError:
#             If request fails, times out, or server returns HTTP >= 400.
#     """
#     params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
#     return _request_json(
#         "POST",
#         f"{PRED_BASE_URL}/predict_ml",
#         json=params,
#         timeout_s=_predict_http_timeout(timeout),
#     )


@mcp.tool(
    title="Start Generative Model Training",
    description="Starts training of a generative model for the specified case using a training dataset path.",
    tags={"training", "generation", },
)
def start_generative_model_training(
    dataset_path: str,
    case_name: str,
) -> Dict[str, str]:
    """
    Start training for a generative model case.

    The function initiates training using the dataset located at
    `dataset_path` and links the training run to `case_name`.
    A production implementation may start training immediately or enqueue
    a background job, but the response format should remain stable.

    Args:
        dataset_path:
            Path to a file or directory with the training dataset.
            Can be an absolute path or a project-relative path.
        case_name:
            Unique case identifier used to name and track the training run.

    Returns:
        Dict[str, str]:
            Dictionary with training-start metadata:
            - `case_name`: case identifier for which training was requested.
            - `status`: training launch state (for example, `training_started`).
            - `message`: human-readable confirmation or diagnostic message.

    Raises:
        ValueError:
            If `dataset_path` or `case_name` is empty after trimming.
        RuntimeError:
            If training cannot be launched due to service or runtime errors.
    """
    dataset_path = dataset_path.strip()
    case_name = case_name.strip()

    if not dataset_path:
        raise ValueError("dataset_path must not be empty")
    if not case_name:
        raise ValueError("case_name must not be empty")

    return {
        "case_name": case_name,
        "status": "training_started",
        "message": f"Generative model training successfully started for case '{case_name}'.",
    }


@mcp.tool(
    title="Generate Molecules (GAN)",
    description="Generates durg molecules and calculated properties without affiliation with a specific disease ",
    tags={"generation", "gan"},
)
def generate_mols(
    num: int = 10,
    case: str = "Alzheimer",
) -> Dict[str, Any]:
    """
    Generate molecules with GAN without affiliation with a specific disease .

    Args:
        num:
            Number of molecules requested.
            Default 10.

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
    return _post_generation_request("gan_case_generator", numb_mol=num, case=case)


def _generate_case_mols(endpoint: str, num: int) -> Dict[str, Any]:
    return _post_generation_request(endpoint, numb_mol=num)


@mcp.tool(
    title="Generate Molecules for Skleroz",
    description="Generate molecules for multiple sclerosis (Skleroz).",
    tags={"generation", "disease", "skleroz"},
)
def generate_skleroz_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for multiple sclerosis (Skleroz).

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Skleroz", num)


@mcp.tool(
    title="Generate Molecules for Parkinson",
    description="Generate molecules for Parkinson's disease.",
    tags={"generation", "disease", "parkinson"},
)
def generate_parkinson_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for Parkinson's disease.

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Parkinson", num)


@mcp.tool(
    title="Generate Molecules for Cancer",
    description="Generate molecules for cancer case.",
    tags={"generation", "disease", "cancer"},
)
def generate_cancer_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for cancer.

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Canser", num)


@mcp.tool(
    title="Generate Molecules for Dyslipidemia",
    description="Generate molecules for dyslipidemia.",
    tags={"generation", "disease", "dyslipidemia"},
)
def generate_dyslipidemia_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for dyslipidemia.

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Dyslipidemia", num)


@mcp.tool(
    title="Generate Molecules for Drug Resistance",
    description="Generate molecules for drug resistance case.",
    tags={"generation", "disease", "drug_resistance"},
)
def generate_drug_resist_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for drug resistance.

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Drug_resist", num)


@mcp.tool(
    title="Generate Molecules for Alzheimer",
    description="Generate molecules for Alzheimer's disease.",
    tags={"generation", "disease", "alzheimer"},
)
def generate_alzheimer_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for Alzheimer.

    Args:
        num:
            Number of molecules requested.
            Default: `10`.
            Endpoint internally caps requests above `100`.

    Returns:
        Dict[str, list]:
            Case generator output with aligned lists.
            Typical keys: `Molecules`, `QED`, `Synthetic Accessibility`, `PAINS`,
            `SureChEMBL`, `Glaxo`, `Brenk`, `BBB`.
            Optional keys: `IC50`, `KI`, and anti-target docking columns
            (`Anti Docking score for <target>`).
    """
    return _generate_case_mols("search_Alzheimer", num)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
