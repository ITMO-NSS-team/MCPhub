# llm_agents_chemistry

# Installation

```
path/to/python3.10.exe -m venv env

pip install -r requirements.txt

source env/Scripts/activate
```

# Docker

Build image (`automl` Dockerfile now accepts only `CSNTS_...` build args):

```bash
cd automl

docker build -t automl_mcp \
  --build-arg CSNTS_MOLS_ML_MCP_PORT=8777 \
  --build-arg CSNTS_S3_ENDPOINT_URL=http://10.32.1.114:9000 \
  --build-arg CSNTS_S3_ACCESS_KEY='user' \
  --build-arg CSNTS_S3_SECRET_KEY = SECRET_KEY \
  --build-arg CSNTS_S3_BUCKET_NAME=molecule-generative-mcp \
  -f automl/Dockerfile .
```

Run container:

```bash
docker run --name automl_mcp_server --rm -it \
  -p 8777:8777 \
   automl_mcp
```

# Environment (S3 + MCP)

Set these variables for downloading train datasets from S3:

```
ENDPOINT_URL=<s3_endpoint_url>
ACCESS_KEY=<s3_access_key>
SECRET_KEY=<s3_secret_key>
BUCKET_NAME=<s3_bucket_name>
STATE_S3_KEY=state/state.json
MOLS_ML_MCP_PORT=8777
```

If `save_trained_data_to_sync_server=true`, trained model artifacts are uploaded to:
`ml_weights/<case>/...` in the same S3 bucket.

Run MCP server:

```
python automl_mcp.py
```

`train_ml` in MCP runs in background process and returns `job_id` immediately.
Use `train_ml_job_status(job_id=...)` to poll process status, and `check_state` to see case-level training status/metrics.
