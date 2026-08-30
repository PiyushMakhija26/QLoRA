FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency configs
COPY pyproject.toml ./

# Install package dependencies
RUN uv pip install --system --no-cache .

# Copy repo code
COPY src/ ./src
COPY configs/ ./configs
COPY scripts/ ./scripts
COPY tests/ ./tests

# Install the package in editable mode
RUN uv pip install --system --no-cache -e .

# Set default entry point
ENTRYPOINT ["python"]
