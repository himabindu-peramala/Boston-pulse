# docker/ml-training.Dockerfile
#
# ML Training container for Boston Pulse crime risk model.
# This image is built by CI and tagged with the git SHA.
#
# Usage:
#   docker build -f docker/ml-training.Dockerfile -t ml-training:local .
#   docker run ml-training:local python -m models.crime_navigate.cli train --help
#
# The DAG pulls this image and runs:
#   python -m models.crime_navigate.cli train \
#       --execution-date 2026-03-23 \
#       --stage staging \
#       --output-json /tmp/results.json

FROM python:3.11-slim

# Metadata
LABEL maintainer="boston-pulse-team"
LABEL description="Boston Pulse ML Training Container"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud CLI (for Artifact Registry operations)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser
WORKDIR /app

# Install Python dependencies first (for better layer caching)
COPY ml/pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "pandas>=2.2,<3.0" \
    "pyarrow>=15.0" \
    "scipy>=1.13" \
    "pydantic>=2.0" \
    "pyyaml" \
    "requests" \
    "google-cloud-storage" \
    "google-cloud-firestore" \
    "google-cloud-artifact-registry>=1.11" \
    "google-auth>=2.0" \
    "fairlearn>=0.10.0" \
    "lightgbm>=4.0" \
    "optuna>=3.5" \
    "mlflow>=2.10" \
    "shap>=0.44" \
    "scikit-learn>=1.4" \
    "matplotlib"

# Copy ML package source code
COPY ml/models /app/models
COPY ml/shared /app/shared
COPY ml/configs /app/configs

# Set ownership
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Set Python path so modules can be imported
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command shows help
CMD ["python", "-m", "models.crime_navigate.cli", "--help"]
