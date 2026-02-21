# docker/airflow.Dockerfile
FROM apache/airflow:2.7.3-python3.11

# Switch to root for system deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy pyproject.toml from data-pipeline
COPY data-pipeline/pyproject.toml /tmp/pyproject.toml

# Install everything defined in pyproject.toml
RUN pip install --no-cache-dir /tmp
