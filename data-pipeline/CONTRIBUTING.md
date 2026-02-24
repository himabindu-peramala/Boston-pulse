Boston Pulse Data Pipeline – Contributing Guide
==============================================

This guide explains how to **develop**, **run**, and **extend** the Boston Pulse data pipeline.

If you only want the quickstart, see [`README.md`](./README.md).

---

## 1. Local development setup

From `Boston-pulse/data-pipeline/`:

```bash
# 1) Create your local env file
cp .env.example .env

# 2) Install dependencies (dev)
make setup-dev

# 3) Run tests
make test
```

### 1.1 Environment variables

Use `.env.example` as the source of truth for required variables.

Common variables:

- **GCP**: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_BUCKET_NAME`
- **Airflow metadata DB**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- **Airflow admin (UI login)**: `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`, `AIRFLOW_EMAIL`
- **Alerting** (optional): `SLACK_WEBHOOK_URL`
- **Config environment**: `BP_ENVIRONMENT` (usually `dev` locally)

---

## 2. Running Airflow locally (Docker)

Local Airflow uses `docker/docker-compose.airflow.yml`.

From `data-pipeline/`:

```bash
make airflow-up-dp # To run Airflow
make airflow-logs-dp # To view Airflow logs
make airflow-down-dp # To shut down Airflow containers
```

The Airflow UI is:

- `http://localhost:8080`

Credentials come from your `.env`:

- Username: `AIRFLOW_USERNAME`
- Password: `AIRFLOW_PASSWORD` <!-- pragma: allowlist secret -->

### 2.1 Verifying mounts (common “blank DAGs” issue)

If DAGs don’t appear in the UI, first verify that `/opt/airflow/dags` is populated inside the scheduler container:

```bash
cd docker
docker compose --env-file ../.env -f docker-compose.airflow.yml exec airflow-scheduler \
  ls -la /opt/airflow/dags
```

If it’s empty, your compose `volumes:` paths are likely incorrect for your repo layout.

---

## 3. Code layout (developer view)

```text
configs/          # YAML configs (datasets, environments, alerting)
dags/             # Airflow DAGs and DAG utilities (GCS I/O, watermark, lineage, callbacks)
schemas/          # Dataset schema definitions for validation
secrets/          # Local-only secrets (gitignored)
src/              # Core pipeline implementation (datasets, validation, bias, alerting, shared)
tests/            # Unit + integration tests
docker/           # Local Airflow Dockerfile + compose
```

Deep dives:

- DAG conventions and utilities: [`dags/README.md`](./dags/README.md)
- Schema structure and validation stages: [`schemas/README.md`](./schemas/README.md)

---

## 4. Linting, formatting, and tests

From `data-pipeline/`:

```bash
make lint
make lint-fix
make format
make test
make test-unit
make test-cov
```

---

## 5. Adding a new dataset

Use the existing dataset DAGs as templates (checkout `crime_dag.py`).

Typical steps:

1. **Scaffold**

```bash
make create-dataset NAME=my_dataset
```

2. **Implement dataset logic**

- `src/datasets/my_dataset/ingest.py`
- `src/datasets/my_dataset/preprocess.py`
- `src/datasets/my_dataset/features.py`

3. **Define schemas**

- Add schemas for raw, processed, and features (`.json`) in `schemas/my_dataset/` (see [`schemas/README.md`](./schemas/README.md))

4. **Add dataset config**

- Create `configs/datasets/my_dataset.yaml`

5. **Create a DAG**

- Add `dags/datasets/my_dataset_dag.py`
- Follow the standard pipeline stages and reuse helpers from `dags/utils/`

6. **Add tests**

- Add unit tests under `tests/unit/datasets/my_dataset/`

---

## 6. Reproducibility (GCS versioning + lineage)

This pipeline is reproducible without DVC by relying on:

- **Deterministic GCS paths** per dataset/date/stage
- **GCS object generations** to identify immutable artifact versions
- A lineage record written at the end of each DAG run

Lineage implementation:

- Core: [`src/shared/lineage.py`](./src/shared/lineage.py)
- DAG wrapper utilities: [`dags/utils/lineage_utils.py`](./dags/utils/lineage_utils.py)

Lineage records are stored in GCS under:

```text
gs://<bucket>/lineage/<dataset>/dt=<YYYY-MM-DD>/lineage.json
```

Use lineage to:

- Inspect which exact artifact generations were used/produced by a run
- Compare runs to see what changed
- Generate `gsutil cp` restore commands for specific generations

---

## 7. Secrets and production guidance

- Do not commit `.env` or any GCP key JSON.
- For production deployments on GCP (yet to be done), prefer **Workload Identity** / attached service accounts so you do not need JSON keys.
- For local development, store keys under `data-pipeline/secrets/` and mount them via Docker compose.
