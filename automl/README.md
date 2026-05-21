# automl MCP server

MCP server for AutoML training and inference of molecular-property models
(regression / classification over SMILES features). Trained pipelines and
predictions are exchanged with the agent **via S3 links**, not inline arrays.

## Build & run with Docker

Secrets and runtime config live in a `.env` file at container start time —
the Dockerfile does **not** bake S3 credentials or ports into the image
anymore.

### 1. Prepare `.env`

```bash
cd automl
cp .env.example .env
# edit .env: set ENDPOINT_URL / ACCESS_KEY / SECRET_KEY / BUCKET_NAME
```

Minimum variables (see `.env.example` for the full list):

| Variable | Purpose |
| --- | --- |
| `ENDPOINT_URL` | S3-compatible endpoint (e.g. `http://10.32.1.114:9000`) |
| `ACCESS_KEY` / `SECRET_KEY` | S3 credentials |
| `BUCKET_NAME` | Bucket for `train/`, `ml_weights/`, `predictions/`, `state/` |
| `MOLS_ML_MCP_PORT` | MCP HTTP port inside the container (default `8777`) |
| `MCP_HOST` / `MCP_TRANSPORT` | `0.0.0.0` / `http` |

### 2. Build the image (no `--build-arg` needed)

```bash
docker build -t automl_mcp .
```

The image is reusable across environments — same image, different `.env`.

### 3. Run

Pass `.env` explicitly:

```bash
docker run --name automl_mcp_server \
  --env-file .env \
  -p 8777:8777 \
  automl_mcp
```

The MCP server listens on `http://<host>:${MOLS_ML_MCP_PORT}/mcp/`.

### 4. Verify

From any machine that can reach the host:

```python
from fastmcp.client import Client
import asyncio

async def main():
    async with Client("http://<host>:8777/mcp/") as c:
        for t in await c.list_tools():
            print(t.name)

asyncio.run(main())
```

Expected tools: `list_automl_train_cases`, `get_s3_train_case_columns`,
`health_check`, `get_mcp_logs`, `check_state`, `train_ml`,
`train_ml_job_status`, `predict_ml`.

### Server-side logs

Container entrypoint is `api.sh`, which starts the MCP via
`nohup python automl_mcp.py > mcp.txt 2>&1`. The resulting log file lives
at `/app/automl/mcp.txt` inside the container and is readable through the
MCP itself via `get_mcp_logs(tail_lines=200)` — handy when a training run
ends up in `status="Failed"` and you need the actual worker stack trace.

## MCP contract — data exchange via S3

### `train_ml`

The training CSV is supplied as an HTTP(S) URL (e.g. an S3 presigned URL),
which the backend fetches via plain `requests.get(...)`. No S3 credentials
are required to read the URL — it can point to any reachable endpoint or
account.

```python
train_ml(
    case="Alzheimer_v2",
    train_data_url="http://10.32.1.114:9000/molecule-generative-mcp/train/Alzheimer.csv?X-Amz-...",
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    save_trained_data_to_sync_server=True,  # upload weights to S3 (default)
)
```

The tool returns immediately with a `job_id`; training runs in a background
process. Poll status with `train_ml_job_status(job_id=...)` and read case-level
metrics from `check_state()`.

Trained Fedot pipelines land in
`s3://<bucket>/ml_weights/<case>/trained_data_<case>_<problem>/...`.

### `predict_ml`

Input is either an inline `smiles_list` OR `input_s3_key` (CSV in S3 with
a SMILES column). Predictions are uploaded to
`s3://<bucket>/predictions/<case>/<uuid>.csv` and the response carries a
presigned URL. Set `return_inline_predictions=True` to also include the raw
dict in the response.

Resilience: before inference the tool checks that pipeline weights are
available locally; if missing it tries to download them from
`s3://<bucket>/ml_weights/<case>/...`. If nothing is found a structured error
response is returned (`status: "case_not_found" / "weights_not_found" / ...`)
instead of raising — the agent can branch on `status`.

## Updating an already-deployed container

After pulling new code, rebuild and restart:

```bash
docker build -t automl_mcp .
docker rm -f automl_mcp_server 2>/dev/null
docker run --name automl_mcp_server --env-file .env -p 8777:8777 automl_mcp
```

`.env` changes alone do **not** propagate into a running container — restart
the container after editing `.env`.

## Local development (no Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate     # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env               # then edit
python automl_mcp.py
```
