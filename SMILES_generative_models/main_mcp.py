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

# Base properties returned by disease /search_* endpoints (case_generator).
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


def _predict_http_timeout(timeout_minutes: int) -> int:
    return max(30, int(timeout_minutes) * 60 + 30)


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


@mcp.tool
def get_state_from_server(url: str = "pred", case: Optional[str] = None) -> Union[dict, str]:
    """
    Get model state info from the ML ("pred") or generative ("gen") FastAPI server.

    Args:
        url: "pred" for ML server, "gen" for generative server,
             or a full base URL like "http://10.0.0.1:81".
        case: Optional case name to fetch only one case entry.

    Returns:
        State dictionary or a single case entry if case is provided.
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


@mcp.tool
def predict_prop_by_smiles(
    smiles_list: List[str],
    case: str = "no_name_case",
    timeout: int = 20,
) -> dict:
    """
    Predict molecular properties using the ML prediction server.

    Args:
        smiles_list: List of SMILES strings to evaluate.
        case: Model case name. Use get_state_from_server(url="pred") to see available cases.
        timeout: Timeout (in minutes) passed to the prediction server.

    Returns:
        JSON response with predicted properties.
    """
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    return _request_json(
        "POST",
        f"{PRED_BASE_URL}/predict_ml",
        json=params,
        timeout_s=_predict_http_timeout(timeout),
    )


@mcp.tool
def generate_mols(
    num: int = 10,
    case: str = "Alzheimer",
) -> Dict[str, Any]:
    """
    Generate molecules with GAN using only /gan_case_generator.

    Args:
        num: Number of molecules to generate.
        case: GAN case passed as `case_` to /gan_case_generator.

    Returns:
        Dict of lists with generated molecules and server-calculated properties.
        Expected keys from gan_auto_generator:
        Smiles, Brenk, QED, Synthetic Accessibility, LogP, Polar Surface Area,
        H-bond Donors, H-bond Acceptors, Rotatable Bonds, Aromatic Rings,
        Glaxo, SureChEMBL, PAINS, Validity, Duplicates.
    """
    return _post_generation_request("gan_case_generator", numb_mol=num, case=case)


def _generate_case_mols(endpoint: str, num: int) -> Dict[str, Any]:
    return _post_generation_request(endpoint, numb_mol=num)


@mcp.tool
def generate_skleroz_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for skleroz via /search_Skleroz.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
    """
    return _generate_case_mols("search_Skleroz", num)


@mcp.tool
def generate_parkinson_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for parkinson via /search_Parkinson.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
    """
    return _generate_case_mols("search_Parkinson", num)


@mcp.tool
def generate_cancer_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for cancer via /search_Canser.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
    """
    return _generate_case_mols("search_Canser", num)


@mcp.tool
def generate_dyslipidemia_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for dyslipidemia via /search_Dyslipidemia.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
    """
    return _generate_case_mols("search_Dyslipidemia", num)


@mcp.tool
def generate_drug_resist_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for drug resistance via /search_Drug_resist.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
    """
    return _generate_case_mols("search_Drug_resist", num)


@mcp.tool
def generate_alzheimer_mols(num: int = 10) -> Dict[str, Any]:
    """
    Generate molecules for alzheimer via /search_Alzheimer.

    Returns a dict of lists with case_generator properties:
    Molecules, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo, Brenk, BBB,
    and case-dependent IC50/KI (if available).
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
