FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 
    #MCP_TRANSPORT=stdio

ARG CSNTS_MOLS_ML_MCP_PORT
ARG CSNTS_S3_ENDPOINT_URL
ARG CSNTS_S3_ACCESS_KEY
ARG CSNTS_S3_SECRET_KEY
ARG CSNTS_S3_BUCKET_NAME="molecule-generative-mcp"
ARG CSNTS_STATE_S3_KEY="/state/state.json"
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

ENV MOLS_ML_MCP_PORT=$CSNTS_MOLS_ML_MCP_PORT

ENV ENDPOINT_URL=$CSNTS_S3_ENDPOINT_URL
ENV ACCESS_KEY=$CSNTS_S3_ACCESS_KEY
ENV SECRET_KEY=$CSNTS_S3_SECRET_KEY
ENV BUCKET_NAME=$CSNTS_S3_BUCKET_NAME
ENV STATE_S3_KEY=$CSNTS_STATE_S3_KEY

ENTRYPOINT ["python", "automl_mcp.py"]
