# chemical-mcp-server

Standalone MCP (Model Context Protocol) server for chemistry tools: SMILES, molecular properties, docking, molecule/reaction OCR, visualization, retrosynthesis, and forward synthesis prediction.

Runs independently with **uv**, **Docker** or **docker-compose**.

## Requirements

- Python 3.11+
- Optional: [uv](https://docs.astral.sh/uv/) for install/run
- Optional: Docker & docker-compose for containerized run

## Setup (local with uv)

```bash
cd chemical_mcp
cp .env.example .env
# Edit .env if needed (CHEM_SERVICES_HOST, CHEM_SERVICES_PORT)

uv sync
```

## Run (local)

```bash
# With uv
uv run python -m server.chemical_server

# Or after uv sync
chemical-mcp-server
```

Server listens on `http://0.0.0.0:7331/mcp`.

## Run with Docker

Compose defines two modes via [profiles](https://docs.docker.com/compose/how-tos/profiles/):

| Command | What starts |
|---------|-------------|
| `docker compose up --build` | **chemical-mcp-server** only. Chemistry and retrosynthesis endpoints must be reachable at the hosts/ports set in `.env` (e.g. other machines or services you run yourself). |
| `docker compose --profile tools up --build` | **chemical-mcp-server** plus **chemical-service** (OpenChemIE/docking stack) and **retrosynthesis-api** (ASKCOS HTTP wrapper). |

```bash
cp .env.example .env
# MCP only (default profile):
docker compose up --build

# MCP + bundled chemical_tools_service (chemical-service + retrosynthesis-api):
docker compose --profile tools up --build
```

### `.env` when using `--profile tools`

Containers talk to each other on the internal Docker network. Point the MCP server at the compose service names and **container** ports:

| Variable | Value |
|----------|--------|
| `CHEM_SERVICES_HOST` | `chemical-service` |
| `CHEM_SERVICES_PORT` | `80` |
| `RETROSYNTHESIS_SERVICES_HOST` | `retrosynthesis-api` |
| `RETROSYNTHESIS_SERVICES_PORT` | `8001` |

Host port mappings (for debugging or external clients): **8000** → chemical API, **8001** → retrosynthesis API, **7331** → MCP.

### Retrosynthesis container credentials

Before `docker compose --profile tools up`, create `chemical_tools_service/retrosynthesis/.env-non-dev` (see `chemical_tools_service/retrosynthesis/README.md`: copy from `.env_example` and set `USER_ASKCOS`, `PASSWORD_ASKCOS`, `ASKCOS_BASE_URL`).

### Running only the tool services

You can still use `chemical_tools_service/docker-compose.yaml` if you only want **chemical-service** and **retrosynthesis-api** without the MCP container (e.g. different host networking). Adjust the root `.env` to match where those services listen.

## Run with Docker (one-off)

```bash
docker build -t chemical-mcp-server .
docker run -p 7331:7331 --env-file .env chemical-mcp-server
```

## Environment (.env)

| Variable | Description | Default |
|----------|-------------|--------|
| `CHEM_SERVICES_HOST` | Host of the chemistry API (OpenChemIE/docking) | `localhost` |
| `CHEM_SERVICES_PORT` | Port of the chemistry API | `8005` |
| `CHEM_SERVICES_TIMEOUT` | Request timeout for chemistry API (seconds) | `60` |
| `RETROSYNTHESIS_SERVICES_HOST` | Host of the retrosynthesis/ASKCOS API | `localhost` |
| `RETROSYNTHESIS_SERVICES_PORT` | Port of the retrosynthesis/ASKCOS API | `8001` |
| `RETROSYNTHESIS_REQUEST_TIMEOUT` | Request timeout for retrosynthesis API (seconds) | `60` |
| `S3_ENDPOINT_URL` | S3-compatible storage endpoint URL | — |
| `S3_BUCKET_NAME` | S3 bucket for storing images and visualizations | — |
| `S3_ACCESS_KEY` | S3 access key | — |
| `S3_SECRET_KEY` | S3 secret key | — |
| `CHEM_MCP_HOST` | MCP server bind address | `0.0.0.0` |
| `CHEM_MCP_PORT` | MCP server port | `7331` |
| `CHEM_MCP_PATH` | MCP server HTTP path | `/mcp` |

Copy `.env.example` to `.env` and adjust as needed.

## Tools exposed via MCP

### Molecule utilities
- `name2smiles` — convert a molecule name to SMILES via PubChem
- `smiles2name` — convert SMILES to IUPAC name via PubChem
- `smiles2prop` — calculate RDKit molecular descriptors from SMILES
- `visualize_molecule` — render interactive 3-D HTML viewer (uploaded to S3, presigned URL returned)

### Activity data
- `fetch_activity_data` — fetch protein–ligand activity data from BindingDB or ChEMBL; saves to CSV

### OCR / image analysis
- `extract_molecules` — detect molecular structures in images (URLs); returns SMILES + annotated image
- `extract_reactions` — detect chemical reactions in images (URLs); returns reaction dicts + annotated image

### Docking
- `calculate_docking` — compute docking score for a SMILES against a PDB receptor; visualization uploaded to S3

### Retrosynthesis & synthesis prediction
- `retrosynthesis_tree_search` — plan retrosynthetic routes for a target SMILES using ASKCOS tree search.
  Accepts `mode` (`fast` / `balanced` / `deep`). Returns ranked routes with steps, reactants, scores,
  and per-route reaction-strip images (uploaded to S3, presigned URLs in `metadata.route_images`).

- `classify_reaction` — classify reaction SMILES (`A.B>>C`) into named reaction classes using ASKCOS.
  Returns ranked hits with class/superclass names and confidence scores.

- `forward_predict` — predict reaction products from reactant SMILES using ASKCOS forward models
  (`wldn5`, `graph2smiles`, `augmented_transformer`). Returns a ranked list of predicted products with
  scores, plus two images uploaded to S3:
  - `metadata.predictions_image` — grid of all predicted product structures
  - `metadata.top_reactions_image` — reaction drawings for the top 3 predictions (reactants → product)

### S3 storage
Result images (route visualizations, product grids, reaction drawings, docking HTML, molecule viewers)
are stored in S3 under the following prefixes:

| Prefix | Content |
|--------|---------|
| `chemical_mcp/molecule_visualizations/` | 3-D molecule HTML |
| `chemical_mcp/annotated_images/` | OCR-annotated images |
| `chemical_mcp/docking_results/` | Docking HTML viewers |
| `chemical_mcp/retrosynthesis/` | Retrosynthesis route images |
| `chemical_mcp/forward_prediction/` | Forward-prediction product grids and reaction images |

All presigned URLs expire after **1 hour**.
