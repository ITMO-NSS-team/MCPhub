# Retrosynthesis Proxy Service

## Overview
The `retrosynthesis` service is a FastAPI application that proxies requests to an upstream ASKCOS instance and normalizes the responses for three workflows:
- Backward retrosynthesis search (`/api/v1/retrosynthesis`)
- Forward reaction prediction (`/api/v1/forward`)
- Reaction classification (`/api/v1/reaction-classification`)

## Running with Docker
1. Copy the example environment file and provide your ASKCOS credentials:
   ```sh
   cd retrosynthesis
   cp .env_example .env-non-dev
   # edit .env-non-dev to set USER_ASKCOS, PASSWORD_ASKCOS, ASKCOS_BASE_URL
   ```
2. Build and start the API with Docker Compose:
   ```sh
   1. cd ..
   2. docker compose up --build
   3. cd retrosynthesis/ASKCOSv2/askcos2_core
   4. make deploy
   ```
   The service listens on `http://localhost:8001`.

### Environment variables
`retrosynthesis/.env` is automatically loaded by Docker Compose and the application settings. Supported keys:

| Variable | Description | Default |
| --- | --- | --- |
| `ASKCOS_BASE_URL` | Base URL of the upstream ASKCOS deployment | `http://localhost:9100` |
| `USER_ASKCOS` | Username for ASKCOS (if required by upstream) | — |
| `PASSWORD_ASKCOS` | Password for ASKCOS (if required) | — |
| `HTTP_TIMEOUT` | Timeout (seconds) for upstream HTTP calls | `60.0` |

## API Reference
All responses are JSON and errors follow FastAPI's default error envelope or proxy upstream status codes.

### POST `/api/v1/retrosynthesis/result`
Computes retrosynthesis routes for a target molecule.

- **Query parameters**
  - `mode` (optional): `"fast" | "balanced" | "deep"` (`"fast"` by default). Selects one of the predefined ASKCOS presets.
- **Request body**

```json
{
  "smiles": "CCCC"
}
```

- **Success response (`200 OK`)**

```json
{
  "target": "CCCC",
  "routes": [
    {
      "id": "route_1",
      "depth": 3,
      "precursor_cost": 2.4,
      "score": 0.78,
      "min_step_plausibility": 0.65,
      "avg_step_plausibility": 0.71,
      "steps": [
        {
          "reaction_smiles": "A.B>>C",
          "mapped_smiles": "...",
          "plausibility": 0.67,
          "precursor_rank": 1,
          "precursor_score": -0.04,
          "model_score": 0.12,
          "template": {
            "reaction_smarts": "...",
            "template_rank": 5,
            "num_examples": 4038
          },
          "reactants": [
            {
              "smiles": "A",
              "terminal": true,
              "buy_link": "https://..."
            }
          ],
          "products": [
            {
              "smiles": "C",
              "terminal": false,
              "buy_link": null,
              "stoichiometry": 1
            }
          ]
        }
      ]
    }
  ]
}
```

Errors from the upstream ASKCOS service are forwarded with their status code and message.

### POST `/api/v1/forward/predict`
Predicts reaction products (forward synthesis).

- **Request body**

```json
{
  "backend": "wldn5",
  "model_name": "pistachio",
  "smiles": ["CCO.CC>>"],
  "reagents": "NaOH",
  "solvent": "H2O"
}
```

- **Success response (`200 OK`)**

```json
{
  "inputs": ["CCO.CC>>", "NaOH", "H2O"],
  "backend": "wldn5",
  "model_name": "pistachio",
  "predictions": [
    {
      "smiles": "CCOCC",
      "score": 0.82
    }
  ]
}
```

### POST `/api/v1/reaction-classification/classify`
Classifies reactions into ASKCOS reaction classes.

- **Request body**

```json
{
  "smiles": ["CCO.CC>>CCOC"],
  "num_results": 5
}
```

- **Success response (`200 OK`)**

```json
{
  "status_code": 0,
  "message": "OK",
  "result": [
    {
      "rank": 1,
      "reaction_num": "1.2.3",
      "reaction_name": "Example reaction",
      "reaction_classnum": "3.B.2",
      "reaction_classname": "Substitution Reaction",
      "reaction_superclassnum": "3",
      "reaction_superclassname": "C-C Bond Formation",
      "prediction_certainty": 0.91
    }
  ]
}
```

## Health and Monitoring
The application exposes the standard FastAPI OpenAPI schema at `/openapi.json` and interactive Swagger documentation at `/docs`. Use these for exploratory testing once the container is running.

