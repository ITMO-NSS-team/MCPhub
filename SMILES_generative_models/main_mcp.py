from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
mcp = FastMCP(name="GenerativeModelsMCP")


def _normalize_base_url(base: str) -> str:
    base = base.strip().rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    return base


def _build_base_url(
    base_env: str,
    host_env: str,
    port_env: str,
    default_port: int,
) -> str:
    explicit = os.getenv(base_env)
    if explicit:
        return _normalize_base_url(explicit)

    host = os.getenv(host_env, "127.0.0.1")
    port = os.getenv(port_env, str(default_port))
    host = _normalize_base_url(host)

    parsed = urlparse(host)
    netloc = parsed.netloc or parsed.path
    if ":" in netloc:
        return host
    return f"{host}:{port}"


PRED_BASE_URL = _build_base_url(
    base_env="ML_TOOLS_BASE_URL",
    host_env="ML_TOOLS_IP",
    port_env="ML_TOOLS_PORT",
    default_port=80,
)
GEN_BASE_URL = _build_base_url(
    base_env="DL_TOOLS_BASE_URL",
    host_env="DL_TOOLS_IP",
    port_env="DL_TOOLS_PORT",
    default_port=80,
)


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


def _load_dataframe(data_path: str) -> pd.DataFrame:
    data_path_obj = Path(data_path)
    ext = data_path_obj.suffix.lower()

    if ext in (".csv", ".txt", ".tsv"):
        return pd.read_csv(data_path_obj)
    if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        return pd.read_excel(data_path_obj)
    if ext in (".parquet", ".parq"):
        return pd.read_parquet(data_path_obj)

    raise ValueError(f"Unsupported data format: {data_path_obj.suffix}")


def _predict_http_timeout(timeout_minutes: int) -> int:
    return max(30, int(timeout_minutes) * 60 + 30)


def _train_http_timeout() -> int:
    return int(os.getenv("TRAIN_HTTP_TIMEOUT_S", "10800"))


def _gan_http_timeout() -> int:
    return int(os.getenv("GAN_HTTP_TIMEOUT_S", "60"))


@mcp.tool
def get_state_from_server(url: str = "pred", case: Optional[str] = None) -> Union[dict, str]:
    """
    Get model state info from the ML ("pred") or generative ("gen") FastAPI server.

    Args:
        url: "pred" for ML prediction server, "gen" for generative server,
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


def _gan_generate_once(num: int, case: str, timeout_s: int) -> Dict[str, Any]:
    params = {"case_": case, "numb_mol": num}
    return _request_json(
        "POST",
        f"{GEN_BASE_URL}/gan_case_generator",
        json=params,
        timeout_s=timeout_s,
    )


@mcp.tool
def generate_mols(
    num: int = 10,
    properties_conditions: Optional[Dict[str, str]] = None,
    num_tries: int = 5,
    case: str = "Alzheimer",
) -> Union[List[Any], str]:
    """
    Generate molecules with optional property-based filtering.

    Args:
        num: Number of molecules to return.
        properties_conditions: Dict of conditions, e.g. {"QED": ">=0.8", "Brenk": "==0"}.
        num_tries: Number of parallel generation attempts.
        case: Generative model case name.

    Returns:
        List of (smiles, {prop: value}) pairs or a message if nothing matched.
    """
    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")

    timeout_s = _gan_http_timeout()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(num_tries, 8)) as executor:
        futures = [
            executor.submit(_gan_generate_once, num, case, timeout_s)
            for _ in range(num_tries)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                continue
            if isinstance(result, dict):
                results.append(result)

    if not results:
        return "No results from the generative server."

    available_props = set(results[0].keys()) - {"Smiles"}

    def parse_condition(cond: str) -> tuple[str, float]:
        s = str(cond).strip()
        for op in ("<=", ">=", "==", "<", ">"):
            if op in s:
                return op, float(s.split(op)[1].strip())
        raise ValueError(f"Unsupported condition format: {cond}")

    def evaluator(value: float, op_str: str, threshold: float, tolerance: float = 0.1) -> bool:
        ops = {
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: abs(a - b) < tolerance,
        }
        if op_str not in ops:
            raise ValueError(f"Unsupported operator: {op_str}")
        return ops[op_str](value, threshold)

    required_props: Dict[str, tuple[str, float]] = {}
    if properties_conditions:
        for key, cond in properties_conditions.items():
            required_props[key] = parse_condition(cond)

    all_smiles: List[str] = []
    all_props: Dict[str, List[Any]] = {key: [] for key in available_props}

    for result in results:
        smiles = result.get("Smiles", [])
        all_smiles.extend(smiles)
        for prop in available_props:
            all_props[prop].extend(result.get(prop, []))

    def is_valid(idx: int) -> bool:
        for prop, (op_str, threshold) in required_props.items():
            if prop not in available_props:
                continue
            if not evaluator(all_props[prop][idx], op_str, threshold):
                return False
        return True

    output: List[Any] = []
    for idx, smile in enumerate(all_smiles):
        if is_valid(idx):
            output.append(
                (
                    smile,
                    {prop: all_props.get(prop, [None])[idx] for prop in required_props},
                )
            )

    if not output:
        return "No molecules matched the requested property filters."
    return output[:num] if len(output) > num else output


def _train_ml_with_data(
    case: str,
    data_dict: Dict[str, Any],
    feature_column: List[str],
    target_column: List[str],
    regression_props: List[str],
    classification_props: List[str],
    description: Optional[str],
) -> Any:
    payload: Dict[str, Any] = {
        "case": case,
        "data": data_dict,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": 5,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
    }
    return _request_json(
        "POST",
        f"{PRED_BASE_URL}/train_ml",
        json=payload,
        timeout_s=_train_http_timeout(),
    )


def _train_gen_with_data(
    case: str,
    data_dict: Dict[str, Any],
    feature_column: List[str],
    target_column: List[str],
    regression_props: List[str],
    classification_props: List[str],
    description: Optional[str],
    fine_tune: bool = True,
    n_samples: int = 10,
) -> Any:
    payload: Dict[str, Any] = {
        "case": case,
        "data": data_dict,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": 5,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
        "fine_tune": fine_tune,
        "n_samples": n_samples,
    }
    return _request_json(
        "POST",
        f"{GEN_BASE_URL}/train_gan",
        json=payload,
        timeout_s=_train_http_timeout(),
    )


def _wait_for_training_completion(
    base_url: str,
    case: str,
    model_key: str,
    poll_interval: int = 30,
    max_wait_time: int = 18000,
) -> Dict[str, Any]:
    start_time = time.time()
    last_print_time = start_time

    while time.time() - start_time < max_wait_time:
        try:
            status = _request_json("GET", f"{base_url}/check_state", timeout_s=30)
            state = status.get("state", {}) if isinstance(status, dict) else {}
            case_state = state.get(case, {})
            model_state = case_state.get(model_key, {})
            if model_state.get("status") == "Trained":
                return status
        except Exception:
            pass

        current_time = time.time()
        if current_time - last_print_time >= 60:
            elapsed = current_time - start_time
            print(f"Training in progress for case '{case}' ({elapsed:.0f}s elapsed)")
            last_print_time = current_time

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Training timeout for case {case} after {max_wait_time} seconds"
    )


def _ml_dl_training_sync(
    *,
    case: str,
    data_dict: Dict[str, Any],
    feature_column: List[str],
    target_column: List[str],
    regression_props: List[str],
    classification_props: List[str],
    description: Optional[str],
    poll_interval: int,
    max_wait_time: int,
) -> None:
    print(f"Starting ML/DL training pipeline for case: {case}")

    _train_ml_with_data(
        case=case,
        data_dict=data_dict,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        description=description,
    )
    _wait_for_training_completion(
        PRED_BASE_URL,
        case=case,
        model_key="ml_models",
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
    )

    _train_gen_with_data(
        case=case,
        data_dict=data_dict,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        description=description,
    )
    _wait_for_training_completion(
        GEN_BASE_URL,
        case=case,
        model_key="generative_models",
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
    )

    print(f"Training completed successfully for case: {case}")


@mcp.tool
def run_ml_dl_training_by_daemon(
    case: str,
    path: str,
    feature_column: List[str] = ["smiles"],
    target_column: List[str] = ["docking_score"],
    regression_props: List[str] = ["docking_score"],
    classification_props: List[str] = [],
    description: Optional[str] = None,
    poll_interval: int = 30,
    max_wait_time: int = 18000,
) -> str:
    """
    Start ML + generative model training in a background thread.

    Args:
        case: Unique case name.
        path: Path to dataset on the MCP server machine (CSV/TSV/Excel/Parquet).
        feature_column: Column(s) with SMILES.
        target_column: Target property columns to learn.
        regression_props: Properties for regression training.
        classification_props: Properties for classification training.
        description: Optional description for the case.
        poll_interval: Status polling interval in seconds.
        max_wait_time: Maximum wait time in seconds for each model.

    Returns:
        A status message indicating the training job was started.
    """
    if isinstance(feature_column, str):
        feature_column = [feature_column]
    if isinstance(target_column, str):
        target_column = [target_column]

    if not regression_props and not classification_props:
        regression_props = list(target_column)

    if not target_column:
        raise ValueError("target_column is empty. Provide at least one target column.")
    if not feature_column:
        raise ValueError("feature_column is empty. Provide at least one feature column.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = _load_dataframe(path)
    df_columns = set(df.columns.tolist())

    for column in feature_column:
        if column not in df_columns:
            raise ValueError(
                f'No "{column}" column in data. Available: {df.columns.tolist()}'
            )
    for column in target_column:
        if column not in df_columns:
            raise ValueError(
                f'No "{column}" column in data. Available: {df.columns.tolist()}'
            )

    data_dict = df.to_dict()

    def _run_background() -> None:
        try:
            _ml_dl_training_sync(
                case=case,
                data_dict=data_dict,
                feature_column=feature_column,
                target_column=target_column,
                regression_props=regression_props,
                classification_props=classification_props,
                description=description,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time,
            )
        except Exception as exc:
            print(f"Background training failed for case {case}: {exc}")

    import threading

    thread = threading.Thread(target=_run_background, daemon=False)
    thread.start()
    return f"Training pipeline started for case: {case}"


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
