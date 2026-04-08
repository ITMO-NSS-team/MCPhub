# chemical-mcp-server

MCP server for chemistry tools: SMILES conversion, molecular properties calculations, docking, molecule/reaction OCR, visualization, retrosynthesis, and forward synthesis prediction.

---

## Quick Start (step by step)

### Option 1: MCP server only

Use this if the chemical API and retrosynthesis services are **already running externally** (e.g., deployed on a remote server or started separately). This option only starts the MCP server itself — it does not launch any backend services.

**Step 1.** Copy the environment variables file:
```bash
cp .env.example .env
```

Point the MCP server to your already-running services by setting the correct hosts and ports in `.env`:
```env
CHEM_SERVICES_HOST=<host of your chemical API>
CHEM_SERVICES_PORT=8000
RETROSYNTHESIS_SERVICES_HOST=<host of your retrosynthesis API>
RETROSYNTHESIS_SERVICES_PORT=8001
S3_ENDPOINT_URL= https://storage.yandexcloud.net
S3_BUCKET_NAME=coscientist
S3_ACCESS_KEY=YCXXX
S3_SECRET_KEY=YCXXX
```

**Step 2.** Start the MCP server container:
```bash
docker compose up --build
```

The MCP server will be available at `http://localhost:7331/mcp`.

---

### Option 2: MCP server + chemical API + retrosynthesis (recommended)

Required for docking, retrosynthesis, and forward synthesis. A separately deployed **ASKCOS** instance is needed.

**Step 1.** Copy and edit `.env` (If you don't make this in previous Option 1):

```bash
cp .env.example .env
```

Make sure the correct hosts and ports are set in `.env`:
```env
CHEM_SERVICES_HOST=localhost
CHEM_SERVICES_PORT=8000
RETROSYNTHESIS_SERVICES_HOST=localhost
RETROSYNTHESIS_SERVICES_PORT=8001
```

**Step 2.** Configure ASKCOS access — create the credentials file (any values will work, so simply run):

```bash
cp chemical_tools_service/retrosynthesis/.env_example \
   chemical_tools_service/retrosynthesis/.env
```

**Step 3.** If ASKCOS is not yet deployed — do a full deploy (first time only):
```bash
make all-deploy
```
> This will start MCP + chemical API + retrosynthesis + ASKCOS with database initialization.

For subsequent runs (ASKCOS already installed):
```bash
make all
```

Or, if ASKCOS is already running and you only need MCP + tools containers:
```bash
make compose
```

## Requirements

- Python 3.11+
- Optional: [uv](https://docs.astral.sh/uv/) for install/run
- Optional: Docker & docker-compose for containerized run

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
