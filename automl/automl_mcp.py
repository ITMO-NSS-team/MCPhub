import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from huggingface_hub import hf_hub_download

from api_utils import MLData, inference_ml, train_ml_with_data
from utils.base_state import TrainState

IMPORT_PATH = Path(__file__).resolve().parent
STATE_REPO_ID = "SoloWayG/Molecule_transformer"
STATE_FILE = "state.json"

mcp = FastMCP("automl-mcp")


def _sync_state_from_hf() -> str:
    """Download the latest shared `state.json` from Hugging Face Hub.

    Returns:
        Absolute path to downloaded file in local `automl` directory.

    Notes:
        Uses environment variable `HF_TOKEN` for authenticated access.
    """
    return hf_hub_download(
        repo_id=STATE_REPO_ID,
        filename=STATE_FILE,
        local_dir=str(IMPORT_PATH),
        force_download=True,
        token=os.getenv("HF_TOKEN"),
    )


@mcp.tool(
    title="Health Check",
    description=(
        "Liveness probe for the AutoML MCP server. "
        "Returns a static status payload to confirm the server is reachable."
    ),
)
def health_check() -> dict[str, str]:
    """Return server health status.

    Returns:
        Dictionary with fixed key/value: `{\"health_check\": \"ok\"}`.
    """
    return {"health_check": "ok"}


@mcp.tool(
    title="Check Training State",
    description=(
        "Returns current AutoML state and list of calculable molecular properties. "
        "State is read from local `state.json`."
    ),
)
def check_state() -> dict[str, Any]:
    """Get current training registry and available calculable properties.

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


@mcp.tool(
    title="Train ML Model",
    description=(
        "Starts AutoML training for a case. "
        "Before training, refreshes shared `state.json` from Hugging Face Hub."
    ),
)
def train_ml(data: dict[str, Any]) -> dict[str, Any]:
    """Train AutoML pipelines for a case using tabular data and metadata.

    Args:
        data: Training payload compatible with `MLData`.
            Common fields:
            - `case` (str): case identifier.
            - `data` (dict | optional): dataframe-like mapping; if provided it is
              saved to CSV automatically.
            - `data_path` (str | optional): path to existing CSV.
            - `feature_column` (list[str]): feature columns, default `['Smiles']`.
            - `target_column` (list[str]): columns to predict.
            - `regression_props` / `classification_props` (list[str] | optional):
              target properties per task type.
            - `s3_key` (str | optional): key/path to CSV in S3. If provided, dataset
              is downloaded before training.
            - `endpoint_url`, `access_key`, `secret_key`, `bucket_name`
              (str | optional): S3 connection settings; env fallbacks are supported.
            - `timeout` (int): training timeout in minutes.
            - `description` (str): case description.
            - `save_trained_data_to_sync_server` (bool): optional sync flag.

    Returns:
        Dictionary with `status` and `case`, e.g. `{\"status\": \"ok\", \"case\": \"Alzheimer\"}`.

    Raises:
        pydantic.ValidationError: if payload format does not match `MLData`.
        Any exception propagated by data preparation, training, or HF sync logic.
    """
    payload = MLData(**data)
    _sync_state_from_hf()
    train_ml_with_data(payload)
    return {"status": "ok", "case": payload.case}


@mcp.tool(
    title="Predict Molecular Properties",
    description=(
        "Runs inference for a list of SMILES strings in a selected case. "
        "Before inference, refreshes shared `state.json` from Hugging Face Hub."
    ),
)
def predict_ml(data: dict[str, Any]) -> Any:
    """Run AutoML inference for SMILES list in an existing case.

    Args:
        data: Inference payload compatible with `MLData`.
            Required in practice:
            - `case` (str): trained case name.
            - `smiles_list` (list[str]): molecules in SMILES format.
            Optional:
            - `timeout` (int), plus other `MLData` fields if needed.

    Returns:
        Inference result from `inference_ml(...)`. The structure depends on the
        trained case and available predictors.

    Raises:
        pydantic.ValidationError: if payload format does not match `MLData`.
        Any exception propagated by inference or HF sync logic.
    """
    payload = MLData(**data)
    _sync_state_from_hf()
    return inference_ml(payload)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MOLS_ML_MCP_PORT", "8777"))
    mcp.run(transport=transport, host=host, port=port)
