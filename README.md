# Boston Pulse

Boston Pulse is a machine-learning–driven “digital twin” of the City of Boston. It unifies municipal open data, real-time feeds, and analytics into a conversational and navigation system that helps residents and newcomers understand, navigate, and make decisions about city life.

The system:

- Integrates heterogeneous Boston city datasets into a unified city state
- Builds predictive models for civic services, neighborhood recommendations, and urban risk
- Enables natural-language interaction with structured and real-time city data
- Demonstrates an end-to-end ML pipeline grounded in public open data

This repository is organized as a **monorepo of micro-services**: each top-level directory is its own component with its own README and setup, so you can work on the data pipeline, backend, and frontend independently.

## Monorepo Structure

From the repo root:

```text
Boston-pulse/
├── data-pipeline/    # Airflow ETL, features in GCS, lineage
├── ml/               # Training code, Docker image definition, tests
├── backend/          # API (WIP)
├── frontend/         # UI (WIP)
├── notebooks/        # EDA
├── infrastructure/   # GCP bootstrap (`gcp-setup.sh`), secrets reference
├── scripts/          # Cloud Run bootstrap, GCS deploy, VM sync
├── docker/           # Production Airflow compose / Dockerfiles
├── data/             # Small samples (not full production data)
├── secrets/          # Local-only; gitignored
└── .github/workflows/# CI (e.g. ML training pipeline)
```

Each of these acts as a separate micro‑service:

- `**backend/` (WIP)** – Backend APIs and model endpoints used by the UI and chatbot.
- `**frontend/`(WIP)** – Single‑page application that talks to `backend/`.
- `**data-pipeline/`** – Data pipeline stages:
  - Airflow DAGs for ingest → validate → preprocess → features
  - Strict schema & quality validation
  - Bias/fairness checks and model cards
  - GCS‑native lineage using GCS object generations

## Setup: step by step

### Step 1 — Clone the repository

```bash
git clone https://github.com/himabindu-peramala/boston-pulse.git
cd boston-pulse
```

### Step 2 — Local development only (no GCP CI)

- **Airflow + pipeline:** follow `[data-pipeline/README.md](./data-pipeline/README.md)` (`.env` from `.env.example`, `make` targets).
- **ML package:** follow `[ml/README.md](./ml/README.md)` (`cd ml && make install-dev && make test`).

### Step 3 — GCP resources (once per project)

**When:** Before CI can push images or run Cloud Run.
**Where:** Your machine, with [Google Cloud SDK](https://cloud.google.com/sdk) installed.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Preview, then apply
export GCP_PROJECT_ID=YOUR_PROJECT_ID   # optional if you use the script default
./infrastructure/gcp-setup.sh --dry-run
./infrastructure/gcp-setup.sh
```

This creates/verifies buckets, Artifact Registry repos, APIs, and related wiring (see script header for details).

### Step 4 — GitHub Actions secrets (already present)

**When:** After Step 3, before you expect the **ML** workflow to build the image and run training.
**Where in GitHub:** **Repository → Settings → Secrets and variables → Actions → New repository secret.**


| Secret name              | Required?                  | Purpose                                                             |
| ------------------------ | -------------------------- | ------------------------------------------------------------------- |
| `WIF_PROVIDER`           | Yes (for GCP from Actions) | Workload Identity Federation provider resource name                 |
| `WIF_SERVICE_ACCOUNT`    | Yes                        | Service account email GitHub assumes via WIF                        |
| `GCP_PROJECT_ID`         | Yes                        | GCP project ID (**all lowercase** — Docker image tags require it)   |
| `GCS_BUCKET`             | Yes                        | Main data bucket the training job uses (same idea as pipeline data) |
| `CLOUD_RUN_TRAINING_JOB` | No                         | Cloud Run **Job** name; if unset, CI uses default `ml-training-job` |
| `SLACK_WEBHOOK_URL`      | No                         | If set, the ML workflow posts a summary to Slack                    |


**Optional (other automation):** `AIRFLOW_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` — see `[infrastructure/SECRETS.md](./infrastructure/SECRETS.md)` for full lists, examples, and how to obtain `WIF_PROVIDER`.

### Step 5 — Create the Cloud Run training job (once)

**When:** After Artifact Registry exists and you have a training image (or use `:latest` as in the script).
**Where:** Your machine, `gcloud` authenticated to the same project.

```bash
export GCP_PROJECT_ID=YOUR_PROJECT_ID
export GCP_REGION=us-east1
./scripts/bootstrap-cloud-run-training-job.sh
```

Optional environment variables are documented in the script (`TRAINING_IMAGE`, `GCS_BUCKET`, `TRAINING_JOB_SA`, etc.).

### Step 6 — Airflow production VM (optional)

**When:** You run Airflow on a VM and want DAGs/ML synced from GCS instead of git-only.
**Where:** On the VM — configure `docker/docker-compose.prod.yml` (or `.env`) using `[infrastructure/SECRETS.md](./infrastructure/SECRETS.md)` (Airflow section).
**Sync:** install/run `[scripts/gcs-sync.sh](./scripts/gcs-sync.sh)` (often via `[scripts/gcs-sync.service](./scripts/gcs-sync.service)`).

### Scripts you rarely run by hand


| Script                                                   | When                                                | Where / env                                              |
| -------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| `[scripts/deploy-to-gcs.sh](./scripts/deploy-to-gcs.sh)` | CI uploads deploy bundle; manual only for debugging | Needs `GCS_DEPLOY_BUCKET`, `GITHUB_SHA`, `GITHUB_RUN_ID` |
| `[scripts/gcs-sync.sh](./scripts/gcs-sync.sh)`           | VM pulls new deploy from GCS                        | Env vars in script header; `--once` for a single run     |


---

## Where credentials live (summary)


| Location                                  | What goes there                                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **GitHub → Settings → Secrets → Actions** | `WIF_`*, `GCP_PROJECT_ID`, `GCS_BUCKET`, optional `CLOUD_RUN_TRAINING_JOB`, `SLACK_WEBHOOK_URL`, etc.                                    |
| **GCP**                                   | Projects, buckets, IAM, WIF pool/provider (often created via `gcp-setup.sh`)                                                             |
| **Airflow VM**                            | `.env` / compose for `GCS_BUCKET`, `GCP_PROJECT_ID`, Airflow keys, Slack, MLflow URI (see SECRETS.md)                                    |
| **Cloud Run job**                         | Runtime env vars are updated by **CI** when the workflow runs (`gcloud run jobs update`); no manual key paste for that path              |
| **Your laptop**                           | `data-pipeline/.env` (local Airflow); never commit keys. Optional `.secrets` for `[act](https://github.com/nektos/act)` — see SECRETS.md |


---

## Contributing

- Keep each service documented in its own README.
- **Never** commit API keys or service account JSON.

```bash
pip install pre-commit
pre-commit install
```

---

## Data sources

Analyze Boston and related open data, including 311, crime, fire, food inspections, Vision Zero, and BERDO.