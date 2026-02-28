Boston Pulse – Data Pipeline
============================

This directory contains the **Boston Pulse Data Pipeline**: an Airflow‑orchestrated ETL system that ingests Boston open data.

## Quick links

- **End-to-end Pipeline Report**: [`DataPipeline Report.docx`](https://northeastern-my.sharepoint.com/:w:/g/personal/sridarhaladhi_m_northeastern_edu/IQBQb8CmV97JQZ9FnYcRcMfUAf045FBVfK8XWY_f5K6Npdo?e=YNG2c8)
- **Contributor guide**: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- **DAGs overview**: [`dags/README.md`](./dags/README.md)
- **Schemas & validation**: [`schemas/README.md`](./schemas/README.md)

---

## Data Pipeline Flow Diagram
<img width="1507" height="1692" alt="DataPipeline_FlowDiagram" src="https://github.com/user-attachments/assets/f1ef29bf-6f09-4d72-8260-02e1c73026d0" />

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

The `gcp-key.json` file should have the below template:
```text
{
  "type": "service_account",
  "project_id": "<gcp-project-id>",
  "private_key_id": <private-key-id>,
  "private_key": <private-key-here>,
  "client_email": "<client-email>",
  "client_id": "<client-id>",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "<client-cert-url>",
  "universe_domain": "googleapis.com"
}
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
