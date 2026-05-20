# SMILES generative models — MCP server

GAN-based molecule generator served as both a FastAPI service (`main_api.py`)
and an MCP tool layer (`main_mcp.py`). Training datasets are passed in as
HTTP(S) URLs (typically S3 presigned URLs); trained GAN weights and generated
molecules are exchanged with the agent **via S3 links**.

## Build & run with Docker

Secrets and runtime config live in a `.env` file at container start time —
the Dockerfile does **not** bake S3 credentials, HuggingFace tokens or ports
into the image anymore.

### 1. Prepare `.env`

```bash
cd SMILES_generative_models
cp .env.example .env
# edit .env: set ENDPOINT_URL / ACCESS_KEY / SECRET_KEY / BUCKET_NAME / HF_TOKEN / ports
```

Minimum variables (see `.env.example` for the full list):

| Variable | Purpose |
| --- | --- |
| `ENDPOINT_URL` | S3-compatible endpoint (e.g. `http://10.32.1.114:9000`) |
| `ACCESS_KEY` / `SECRET_KEY` | S3 credentials |
| `BUCKET_NAME` | Bucket for `train/`, `gan_weights/`, `generated/`, `state/` |
| `HF_TOKEN` | HuggingFace token (only needed if pushing weights to a private HF repo) |
| `ML_MOLS_MODEL_APP_URL` | URL of the AutoML/predictive ML server (e.g. `http://10.32.11.22:8777`) |
| `GEN_MOLS_MODEL_APP_PORT` | Internal FastAPI port (e.g. `8881`) |
| `MOLS_GEN_MCP_PORT` | MCP HTTP port (e.g. `8882`) — **must differ from `GEN_MOLS_MODEL_APP_PORT`** |
| `MCP_HOST` / `MCP_TRANSPORT` | `0.0.0.0` / `http` |

> Both `main_api.py` and `main_mcp.py` start inside the same container via
> `api.sh`. If both ports collide on `8000`, the second process fails to
> bind and the MCP endpoint will be unreachable from outside.

### 2. Build the image (no `--build-arg` needed)

```bash
docker build -t generative_model_mcp .
```

The Dockerfile is heavy (CUDA base + miniconda + torch+cu116 + HuggingFace
weights download). First build can take 30–60 min and produces a ~10 GB
image. Subsequent rebuilds reuse cached layers as long as upstream files do
not change.

> If the HuggingFace model becomes private, pass the token at build time only
> for the download step: `docker build --build-arg HF_TOK=$HF_TOKEN ...`.
> Runtime `HF_TOKEN` (e.g. for uploading trained weights to HF) still comes
> from `.env`.

### 3. Run

Pass `.env` explicitly and reserve a GPU:

```bash
docker run --name gen_models_mcp_server \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=<your_device_id> \
  -m 64G --cpus="6" \
  --env-file .env \
  -p ${MOLS_GEN_MCP_PORT}:${MOLS_GEN_MCP_PORT} \
  -it --init \
  generative_model_mcp
```

The MCP server listens on `http://<host>:${MOLS_GEN_MCP_PORT}/mcp/`.
The FastAPI side listens on `${GEN_MOLS_MODEL_APP_PORT}` internally and is
called by `main_mcp.py` over localhost; you usually do not need to publish
that port.

### 4. Verify

```bash
python test_mcp_client.py --url http://<host>:${MOLS_GEN_MCP_PORT}/mcp/ --list-only
```

Expected tools: `list_s3_train_cases`, `get_state_from_server`,
`start_generative_model_training`, `generate_mols`, `generate_case_mols`.

## MCP contract — data exchange via S3

### `start_generative_model_training`

Training CSV is always supplied as an HTTP(S) URL (typically an S3 presigned
URL). The backend fetches it via plain `requests.get(...)`; no S3 credentials
are required to read it.

```python
start_generative_model_training(
    case_name="Alzheimer_v2",
    train_data_url="http://10.32.1.114:9000/molecule-generative-mcp/train/Alzheimer.csv?X-Amz-...",
    feature_column=["canonical_smiles"],
    epochs=10,
    fine_tune=True,
    save_trained_data_to_sync_server=True,  # upload weights to S3 (default)
)
```

Trained GAN weights land in
`s3://<bucket>/gan_weights/<case_name>/train_GAN_<case_name>/...`.

### `generate_mols` / `generate_case_mols`

Generated molecules are uploaded to
`s3://<bucket>/generated/<case|gan_default>/<uuid>.csv` and the response
carries a presigned URL plus a summary (`generated_count`, `columns`). Set
`return_inline_results=True` to also include the raw arrays.

Resilience for `generate_mols(case=...)`:

1. Local cache check (`autotrain/GAN_weights/train_GAN_<case>/gan_weights.pkl`).
2. If missing, download from `s3://<bucket>/gan_weights/<case>/train_GAN_<case>/`.
3. If neither — structured error response with
   `status: "case_not_trained" / "weights_not_found" / ...` (no exception).

Generic mode (`case` omitted) uses the bundled fallback GAN — no S3 lookup.

## Updating an already-deployed container

After pulling new code, rebuild and restart:

```bash
docker build -t generative_model_mcp .
docker rm -f gen_models_mcp_server 2>/dev/null
docker run --name gen_models_mcp_server \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<id> \
  -m 64G --cpus="6" --env-file .env \
  -p ${MOLS_GEN_MCP_PORT}:${MOLS_GEN_MCP_PORT} \
  -it --init generative_model_mcp
```

`.env` changes alone do **not** propagate into a running container — restart
the container after editing `.env`.

## Notes & gotchas

- **Windows checkout (`git core.autocrlf=true`)**: `api.sh` must have LF line
  endings, otherwise bash inside the Linux container reads each trailing
  `\r` as part of the last token (`cd target`, `conda env name`, …) and the
  entrypoint silently breaks. The Dockerfile defensively runs
  `sed -i 's/\r$//' api.sh`, and `.gitattributes` pins shell scripts to LF;
  do not bypass these.
- **Port conflict**: do not set `GEN_MOLS_MODEL_APP_PORT == MOLS_GEN_MCP_PORT`.

## Local development (no Docker) — UV environments

See [README_mcp.md](README_mcp.md) for the two-env setup (`molgen` on Python
3.8, `mcp` on Python 3.13) used for working with the code locally without
spinning up the full container.
