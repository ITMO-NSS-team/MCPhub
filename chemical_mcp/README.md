# chemical-mcp-server

MCP server for chemistry tools: SMILES conversion, molecular properties calculations, docking, molecule/reaction OCR, visualization, retrosynthesis, and forward synthesis prediction.

## Requirements

- Python 3.11+
- Optional: [uv](https://docs.astral.sh/uv/) for install/run
- Optional: Docker & docker-compose for containerized run

## Run with Docker

Compose uses [profiles](https://docs.docker.com/compose/how-tos/profiles/) and **`network_mode: host`** for all services in `docker-compose.yml`: they share the **host** network namespace (no published `ports:` mapping). Each process listens on the host on its configured port (e.g. MCP **7331**, chemical API **8000**, retrosynthesis wrapper **8001**—match your app settings).

| Command | What starts |
| ------- | ----------- |
| `docker compose up --build` | **chemical-mcp-server** only (profile default). |
| `docker compose --profile tools up --build` | MCP + **chemical-service** + **retrosynthesis-api** (ASKCOS HTTP wrapper). |

```bash
cp .env.example .env
docker compose up --build
docker compose --profile tools up --build
```

**Linux note:** `network_mode: host` ignores Compose `ports:`; avoid port clashes with other daemons on the same machine.

### `.env` with `--profile tools` and host networking

All containers see **localhost** as the host. Point the MCP server at **host ports**, not Docker DNS names:

| Variable | Typical value |
| -------- | ------------- |
| `CHEM_SERVICES_HOST` | `localhost` |
| `CHEM_SERVICES_PORT` | `8000` |
| `RETROSYNTHESIS_SERVICES_HOST` | `localhost` |
| `RETROSYNTHESIS_SERVICES_PORT` | `8001` |

### Retrosynthesis → ASKCOS (also on host)

Create `chemical_tools_service/retrosynthesis/.env-non-dev` (see `chemical_tools_service/retrosynthesis/README.md`: copy from `.env_example`, set `USER_ASKCOS`, `PASSWORD_ASKCOS`).

If **ASKCOS** is deployed with **`network_mode: host`** as well (default `askcos2_core` layout for `app` / `web`), the API is on the host (commonly **9100**). Use:

```env
ASKCOS_BASE_URL=http://127.0.0.1:9100
```

(or `http://localhost:9100`).

## Makefile (`chemical_mcp/Makefile`)

From `chemical_mcp/`, **`make compose`** runs **`docker compose --profile tools up --build -d`**. ASKCOS is driven from **`chemical_tools_service/retrosynthesis/ASKCOSv2/askcos2_core`** via its own `Makefile` (`deploy` / `update`).

**Requirements:** Docker, `make`, and host **Python 3** with **`pip`** for ASKCOS deploy scripts. Cloning ASKCOS module repos needs **SSH** (`git@gitlab.com`) or **HTTPS** URLs in the chosen `module_config_*.py`.

| Target | What it does |
| ------ | ------------ |
| `make` / `make help` | List targets |
| `make all-deploy` | Compose (**tools**) + ASKCOS **`deploy`** (includes **seed-db** when deploy runs it) |
| `make all` | Compose (**tools**) + ASKCOS **`update`** (no Mongo **seed-db**) | 
| `make compose` | Compose **tools** only (`up --build -d`) |
| `make askcos` | ASKCOS **`deploy`** only |
| `make askcos-update` | ASKCOS **`update`** only |
| `make askcos-deps` | Ensure **PyYAML** on the host Python |

**Typical flow:** once **`make all-deploy`** (or **`make askcos`**) has completed a full ASKCOS install, use **`make all`** for routine restarts. If ASKCOS is already up and you only need MCP + tools containers, run **`make compose`**.

### Running only the tool services

You can use `chemical_tools_service/docker-compose.yaml` for **chemical-service** and **retrosynthesis-api** without the MCP container; set the root `.env` hosts/ports to wherever those APIs listen.

## Run with Docker (one-off)

```bash
docker build -t chemical-mcp-server .
docker run -p 7331:7331 --env-file .env chemical-mcp-server
```

## Environment (.env)


| Variable                         | Description                                      | Default     |
| -------------------------------- | ------------------------------------------------ | ----------- |
| `CHEM_SERVICES_HOST`             | Host of the chemistry API (OpenChemIE/docking)   | `localhost` |
| `CHEM_SERVICES_PORT`             | Port of the chemistry API (host network: app bind port) | `8000` |
| `CHEM_SERVICES_TIMEOUT`          | Request timeout for chemistry API (seconds)      | `60`        |
| `RETROSYNTHESIS_SERVICES_HOST`   | Host of the retrosynthesis/ASKCOS API            | `localhost` |
| `RETROSYNTHESIS_SERVICES_PORT`   | Port of the retrosynthesis/ASKCOS API            | `8001`      |
| `RETROSYNTHESIS_REQUEST_TIMEOUT` | Request timeout for retrosynthesis API (seconds) | `60`        |
| `S3_ENDPOINT_URL`                | S3-compatible storage endpoint URL               | —           |
| `S3_BUCKET_NAME`                 | S3 bucket for storing images and visualizations  | —           |
| `S3_ACCESS_KEY`                  | S3 access key                                    | —           |
| `S3_SECRET_KEY`                  | S3 secret key                                    | —           |
| `CHEM_MCP_HOST`                  | MCP server bind address                          | `0.0.0.0`   |
| `CHEM_MCP_PORT`                  | MCP server port                                  | `7331`      |
| `CHEM_MCP_PATH`                  | MCP server HTTP path                             | `/mcp`      |


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


| Prefix                                  | Content                                              |
| --------------------------------------- | ---------------------------------------------------- |
| `chemical_mcp/molecule_visualizations/` | 3-D molecule HTML                                    |
| `chemical_mcp/annotated_images/`        | OCR-annotated images                                 |
| `chemical_mcp/docking_results/`         | Docking HTML viewers                                 |
| `chemical_mcp/retrosynthesis/`          | Retrosynthesis route images                          |
| `chemical_mcp/forward_prediction/`      | Forward-prediction product grids and reaction images |


All presigned URLs expire after **1 hour**.
