# MCPhub

Three independent MCP servers that share an S3-based exchange protocol:
training data, model weights and generated artifacts move through an
S3-compatible bucket; agents see only short presigned URLs in MCP responses.

| Server | Purpose | Docs |
| --- | --- | --- |
| [automl/](automl/) | AutoML training & inference of molecular-property models (regression / classification) | [automl/README.md](automl/README.md) |
| [SMILES_generative_models/](SMILES_generative_models/) | GAN-based SMILES generator + case-specific generators | [SMILES_generative_models/README.md](SMILES_generative_models/README.md) |
| [chemical_mcp/](chemical_mcp/) | Chemistry utilities: SMILES↔name, properties, docking, OCR, retrosynthesis | [chemical_mcp/README.md](chemical_mcp/README.md) |

## Common build flow

All three containers use the same pattern — secrets and runtime config live
in a per-project `.env`, the Dockerfile is image-portable:

```bash
cd <project_dir>          # automl, SMILES_generative_models, or chemical_mcp
cp .env.example .env      # then fill in S3 / HF / port values
docker compose up -d --build
```

Or with plain `docker run`:

```bash
docker build -t <image_name> .
docker run --env-file .env -p <host_port>:<container_port> <image_name>
```

See each project's README for project-specific build flags (GPU reservation
for `SMILES_generative_models`, compose profiles for `chemical_mcp`, etc.).

## S3 exchange protocol

Across servers the bucket is organised as:

| Prefix | Written by | Read by |
| --- | --- | --- |
| `train/<case>.csv` | external pipelines (agent uploads here) | `train_ml`, `start_generative_model_training` |
| `ml_weights/<case>/...` | `train_ml` (after fit) | `predict_ml` (auto-downloads on a fresh container) |
| `gan_weights/<case>/train_GAN_<case>/...` | `start_generative_model_training` | `generate_mols(case=...)` (auto-downloads) |
| `predictions/<case>/<uuid>.csv` | `predict_ml` | agent (via presigned URL) |
| `generated/<case_or_gan_default>/<uuid>.csv` | `generate_mols`, `generate_case_mols` | agent (via presigned URL) |
| `state/state.json` | both training services | both services (synchronised on every MCP call) |

MCP tools accept training data either as an S3 key/URI **or** as a full
HTTP(S) URL (e.g. a presigned URL — see `train_data_url`). Generated /
predicted artifacts always come back as presigned URLs by default; pass the
`return_inline_*` flag to also get the raw arrays.
