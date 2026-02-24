# Boston Pulse

Boston Pulse is a machine‑learning–driven “digital twin” of the City of Boston. It unifies municipal open data, real‑time feeds, and analytics into a conversational and navigation system that helps residents and newcomers understand, navigate, and make decisions about city life.

This repository is organized as a **monorepo of micro‑services**: each top‑level directory is its own component with its own README and setup, so you can learn and work on each part (data pipeline, backend, frontend) independently.

## Project Objectives

- Integrate heterogeneous Boston city datasets into a unified **city state**
- Build predictive models for civic services, neighborhood recommendations, and urban risk
- Enable **natural‑language interaction** with structured and real‑time city data
- Demonstrate an **end‑to‑end ML pipeline** grounded in public open data

## Monorepo Structure

From the repo root:

```text
Boston-pulse/
├── backend/          # API + model serving service
├── frontend/         # Web UI
├── data-pipeline/    # Airflow-based ETL + GCS bucket store & versioning
├── notebooks/        # Exploratory analysis & research
├── docker/           # Shared infra / docker-compose (root-level)
├── data/             # Small sample or config data (not full raw data)
├── secrets/          # Local-only secrets (gitignored)
└── .github/          # CI workflows (tests, lint, etc.)
```

Each of these acts as a **separate micro‑service**:

- **`backend/` (WIP)** – Backend APIs and model endpoints used by the UI and chatbot.
- **`frontend/`(WIP)** – Single‑page application that talks to `backend/`.
- **`data-pipeline/`** – Data pipeline stages:
  - Airflow DAGs for ingest → validate → preprocess → features
  - Strict schema & quality validation
  - Bias/fairness checks and model cards
  - GCS‑native lineage using GCS object generations

> If you are interested in the data pipeline, go directly to [`data-pipeline/`](./data-pipeline/), then read:
> - [`data-pipeline/README.md`](./data-pipeline/README.md) – quickstart (env, Airflow, tests)
> - [`data-pipeline/CONTRIBUTING.md`](./data-pipeline/CONTRIBUTING.md) – deep‑dive for contributors
- **`notebooks/`** – Jupyter notebooks used for EDA, prototyping, and documenting experiments.


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/himabindu-peramala/boston-pulse.git
cd boston-pulse
```

### 2. Pick a component to work on

- **Data Pipeline**
  See [`data-pipeline/README.md`](./data-pipeline/README.md) for:
  - copying `.env.example` → `.env`
  - `make setup-dev`
  - `make airflow-up-dp` to run Airflow locally

- **Backend API**
  See `backend/` for how to run the API service and connect it to the pipeline outputs.

- **Frontend**
  See `frontend/` for the UI setup (Node.js, dev server, etc.).

- **Notebooks**
  Open `notebooks/` in Jupyter or VS Code to explore the data and experiments.

## Contributing

- Keep each micro‑service **self‑contained** with its own README and clear entry points.
- Use environment variables / `.env` files (gitignored) for secrets; **never** commit API keys or service account JSON.

You can also enable [pre‑commit](https://pre-commit.com/) locally to mirror CI checks:

```bash
pip install pre-commit
pre-commit install
```

# Boston Pulse

Boston Pulse is a machine learning–driven digital twin of the City of Boston. It unifies municipal open data, real-time transit and weather feeds, and community sentiment into a single conversational system that helps residents and newcomers understand, navigate, and make decisions about city life. Instead of interacting with fragmented city portals and dashboards, Boston Pulse synthesizes civic, safety, housing, and mobility data into actionable, context-aware insights delivered through a natural language interface.

## Project Objectives

- Integrate heterogeneous Boston city datasets into a unified City State
- Build predictive models for civic services, neighborhood recommendations, and urban risk
- Enable natural language interaction with structured and real-time city data
- Demonstrate an end-to-end data and machine learning pipeline grounded in public data

## Data Sources

The project primarily relies on datasets from Analyze Boston, including:

- 311 Service Requests
- Crime Incident Reports
- Fire Incident Reporting
- Public Works Violations
- RentSmart Housing Data
- Street Address Management (SAM) and Street Segments
- Bluebike Stations

Additional real-time data sources include MBTA alerts and weather APIs.

## Repository Structure

The repository is organized as a full-stack machine learning system with clearly separated frontend, backend, data, and infrastructure components.

- `frontend/` – React.js frontend for the Boston Pulse user interface
- `backend/` – Backend APIs and machine learning logic
  - `app/` – API entry point and route definitions
  - `src/` – Data ingestion, preprocessing, feature engineering, and modeling code
  - `config/` – Configuration files and parameters
- `data/` – Raw, processed, and feature-engineered datasets
- `notebooks/` – Exploratory analysis and experimentation
- `docker/` – Dockerfiles and container orchestration configuration
- `.github/` – GitHub workflows and repository configuration files

This structure supports modular development, reproducibility, and future deployment.
