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
├── infrastructure/   # Terraform for all GCP resources + secrets reference
├── scripts/          # Deploy helpers called by CI / Makefile
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

## Quickstart: one-command demo on a fresh GCP project

Boston Pulse provisions everything (GCP project APIs, Artifact Registry, GCS
buckets with bucket-level IAM, Firestore, service accounts, Workload Identity
Federation, Cloud Monitoring dashboards/alerts, and an Airflow GCE VM) via
Terraform, and wraps the whole flow in two Make targets.

### Prerequisites

- `gcloud` CLI authenticated (`gcloud auth login` **and** `gcloud auth application-default login`)
- `terraform >= 1.5` (`brew install terraform`)
- A GCP billing account ID — find via `gcloud billing accounts list`
- `gh` CLI authenticated (only required if you want CI to run the training job)

### 1. Clone + configure once

```bash
git clone https://github.com/himabindu-peramala/boston-pulse.git
cd boston-pulse

cp .env.demo.example .env.demo
# edit .env.demo: set GCP_PROJECT_ID, GCP_BILLING_ACCOUNT, GITHUB_REPOSITORY
```

### 2. Bring everything up

```bash
make demo-up
```

This creates the GCP project if it doesn't exist, links billing, runs
`terraform apply` for all infra, and prints the Airflow URL + admin password.

### 3. Use it

```bash
make demo-wait-vm            # ~5 min for the VM to finish building Airflow
make demo-airflow-url        # open in browser, log in as admin
make demo-airflow-password   # admin password

make demo-trigger-data       # unpause + trigger crime_navigate_pipeline on the VM
# ...wait for features.parquet to land in GCS...

make demo-wif                # prints `gh secret set` commands for CI
# paste those into your shell (or copy values into GitHub UI)

gh workflow run ml.yml       # kick off CI-managed training
# or, after CI has created the job:
make demo-trigger-training
```

Inspect everything at any time:

```bash
make demo-status
```

### 4. Tear it all down

```bash
make demo-down
```

Destroys all Terraform-managed resources, asks whether to also delete the GCP
project (30-day grace period), and wipes local `.terraform/` state.

### Local development (no GCP)

- **Airflow + pipeline:** follow [`data-pipeline/README.md`](./data-pipeline/README.md) (`.env` from `.env.example`, `make` targets).
- **ML package:** follow [`ml/README.md`](./ml/README.md) (`cd ml && make install-dev && make test`).

### GitHub Actions secrets (reference)

`make demo-wif` prints the exact `gh secret set` commands. For reference, the ML
workflow expects:

| Secret name              | Required?                  | Purpose                                                             |
| ------------------------ | -------------------------- | ------------------------------------------------------------------- |
| `WIF_PROVIDER`           | Yes                        | Workload Identity Federation provider resource name                 |
| `WIF_SERVICE_ACCOUNT`    | Yes                        | Service account email GitHub assumes via WIF                        |
| `GCP_PROJECT_ID`         | Yes                        | GCP project ID (lowercase — Docker image tags require it)           |
| `GCS_BUCKET`             | Yes                        | Main data bucket the training job reads features from              |
| `CLOUD_RUN_TRAINING_JOB` | No                         | Cloud Run Job name; defaults to `ml-training-job`                   |
| `SLACK_WEBHOOK_URL`      | No                         | If set, the ML workflow posts a summary to Slack                    |

See [`infrastructure/SECRETS.md`](./infrastructure/SECRETS.md) for the full list and how to obtain each value.


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