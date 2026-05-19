FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
    #MCP_TRANSPORT=stdio

WORKDIR /app/automl

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libexpat1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./

# Runtime configuration (S3 credentials, ports, state key, etc.) is provided
# via a `.env` file at container start time. See `.env.example` for the full
# list of variables and `docker-compose.yml` (env_file: .env) for wiring.
# Python-dotenv (`load_dotenv()` in automl_mcp.py / s3_utils.py) reads it from
# the WORKDIR (/app/automl/.env). Alternatively use `docker run --env-file .env`.

ENTRYPOINT ["python", "automl_mcp.py"]
