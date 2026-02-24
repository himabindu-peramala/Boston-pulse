Boston Pulse – Data Pipeline
============================

This directory contains the **Boston Pulse Data Pipeline**: an Airflow‑orchestrated ETL system that ingests Boston open data.

## Quick links

- **Contributor guide**: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- **DAGs overview**: [`dags/README.md`](./dags/README.md)
- **Schemas & validation**: [`schemas/README.md`](./schemas/README.md)

---

## Quickstart (run locally)
### 0) Navigate to `Boston-pulse/data-pipeline/`
```bash
cd data-pipeline
pwd # should print /username/{...}/Boston-pulse/data-pipeline
```

### 1) Configure environment

```bash
cp .env.example .env
```

Fill in required values in `data-pipeline/.env` (GCP project/bucket, Postgres creds, Airflow admin creds, etc.).

### 2) Add GCP key (local only)

Place your service account key at:

```text
data-pipeline/secrets/gcp-key.json
```

The local Airflow stack mounts it into containers at:

```text
/opt/airflow/secrets/gcp-key.json
```

### 3) Install dependencies

```bash
make setup
```

### 4) Start Airflow

```bash
make airflow-up-dp
```

Then open:

- Airflow UI: `http://localhost:8080`

### 5) Run tests (recommended)

```bash
make test
```

---

## Reproducibility (GCS versioning + lineage)

This pipeline does **not** rely on DVC. Reproducibility comes from:

- Writing artifacts to GCS by dataset/date/stage (raw/processed/features)
- Capturing the **GCS generation number** for each artifact at the end of each run
- Persisting a lineage record per run under:

```text
gs://<bucket>/lineage/<dataset>/dt=<YYYY-MM-DD>/lineage.json
```

To learn how lineage is recorded and queried, see:

- [`src/shared/lineage.py`](./src/shared/lineage.py)
- [`dags/utils/lineage_utils.py`](./dags/utils/lineage_utils.py)

---

## Structure (high level)

```text
data-pipeline/
├── Makefile
├── .env.example
├── docker/                 # Local Airflow Docker setup
├── dags/                   # Airflow DAGs + DAG utilities
├── src/                    # Dataset, validation, bias, alerting, shared logic
├── schemas/                # Schema definitions for validation
├── configs/                # Environment + dataset config
└── tests/                  # Test suite
```

For the full developer view and “how to add a dataset”, see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
