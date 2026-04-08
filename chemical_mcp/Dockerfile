FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libfontconfig1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
RUN uv sync --no-dev

COPY mcp_server/ ./mcp_server/

ENV UV_SYSTEM_PYTHON=1


EXPOSE 7331

CMD ["uv", "run", "python", "-m", "mcp_server.chemical_server"]
