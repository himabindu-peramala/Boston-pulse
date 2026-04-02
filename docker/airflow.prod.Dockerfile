# docker/airflow.prod.Dockerfile
# Production Airflow image for GCE VM.
# Bakes in ALL dependencies from both data-pipeline and ml at build time.
# Every container (scheduler, webserver) starts with everything installed.

FROM apache/airflow:2.7.3-python3.11

USER root

# Install system dependencies + Google Cloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud CLI (needed for Artifact Registry push and GCS sync)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# ── data-pipeline dependencies ────────────────────────────────────────────────
# Install from pyproject.toml dependency list directly
# (we copy just the toml, not the full src, to maximise Docker layer caching)
RUN pip install --no-cache-dir \
    "numpy>=2.0,<3" \
    "pandas>=2.2,<3.0.0" \
    "pyarrow>=15.0" \
    "scipy>=1.13" \
    "geopandas>=1.0" \
    "shapely>=2.0" \
    "openpyxl" \
    "h3" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0" \
    "pyyaml" \
    "requests" \
    "httpx" \
    "tenacity" \
    "google-cloud-storage" \
    "google-cloud-firestore" \
    "great-expectations>=0.18.0" \
    "evidently>=0.4.0" \
    "fairlearn>=0.10.0" \
    "python-dotenv>=1.0" \
    "structlog" \
    "rich" \
    "lightgbm>=4.0" \
    "optuna>=3.5" \
    "mlflow>=2.10" \
    "shap>=0.44" \
    "scikit-learn>=1.4" \
    "google-cloud-artifact-registry>=1.11" \
    "google-auth>=2.0"
