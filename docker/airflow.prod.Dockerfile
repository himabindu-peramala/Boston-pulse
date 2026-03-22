# docker/airflow.prod.Dockerfile
# Production Airflow image for GCE VM.
# Bakes in ALL dependencies from both data-pipeline and ml at build time.
# Every container (scheduler, webserver) starts with everything installed.

FROM apache/airflow:2.7.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# ── data-pipeline dependencies ────────────────────────────────────────────────
# Install from pyproject.toml dependency list directly
# (we copy just the toml, not the full src, to maximise Docker layer caching)
RUN pip install --no-cache-dir \
    "pandas>=2.0.0,<3.0.0" \
    "numpy" \
    "pyarrow" \
    "scipy" \
    "openpyxl" \
    "h3" \
    "pydantic" \
    "pydantic-settings" \
    "pyyaml" \
    "requests" \
    "httpx" \
    "tenacity" \
    "google-cloud-storage" \
    "google-cloud-firestore" \
    "great-expectations>=0.18.0" \
    "evidently>=0.4.0" \
    "fairlearn>=0.10.0" \
    "geopandas" \
    "shapely" \
    "python-dotenv" \
    "structlog" \
    "rich"

# ── ml dependencies ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "lightgbm>=4.0" \
    "optuna>=3.5" \
    "mlflow>=2.10" \
    "shap>=0.44" \
    "scikit-learn>=1.4" \
    "google-cloud-artifact-registry>=1.11" \
    "pydantic-settings>=2.0" \
    "python-dotenv>=1.0"