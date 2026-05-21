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
# list of variables. Pass it in with `docker run --env-file .env`, or rely on
# python-dotenv (`load_dotenv()` in automl_mcp.py / s3_utils.py), which reads
# the file from the WORKDIR (/app/automl/.env).

# Belt-and-suspenders against Windows CRLF: if the repo was checked out with
# core.autocrlf=true, api.sh would have \r\n line endings and bash would
# silently break (cd target, etc.). Strip them.
RUN sed -i 's/\r$//' /app/automl/api.sh && chmod +x /app/automl/api.sh

# Output of the MCP server is captured to /app/automl/mcp.txt by api.sh and
# can be read back through the `get_mcp_logs` MCP tool.
ENTRYPOINT ["/bin/bash", "/app/automl/api.sh"]
