DAGs – Boston Pulse Data Pipeline
================================

This folder contains all **Airflow DAGs** and DAG‑side utilities used to run the Boston Pulse pipeline.

If you’re new, start with:

- Top-level quickstart: [`../README.md`](../README.md)
- Contributor guide: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)

---

## Layout

```text
dags/
├── datasets/                 # One DAG per dataset
│   ├── crime_dag.py
│   ├── 311_dag.py
│   ├── food_inspections_dag.py
│   └── cityscore_dag.py
│   ....
│
├── aggregation/              # (WIP) DAGs that combine datasets / compute rollups
├── maintenance/              # (WIP) Maintenance DAGs (schema updates, snapshots, etc.)
└── utils/                    # Shared helpers used by DAGs
    ├── gcs_io.py             # Read/write to GCS + generation helpers
    ├── watermark.py          # Watermark manager for incremental ingestion
    ├── lineage_utils.py      # Lineage recording task wrapper
    ├── callbacks.py          # Airflow callbacks (success/failure hooks)
    └── __init__.py
```

---

## DAG conventions

Dataset DAGs follow a consistent pattern:

- `DAG_ID` is `<dataset>_pipeline` (e.g. `crime_pipeline`)
- `DATASET` matches the dataset folder under `src/datasets/`
- Stages typically include:
  - ingest → validate raw → preprocess → validate processed → features → validate features
  - drift/anomaly checks (optional depending on dataset)
  - bias/fairness checks + mitigation strategies
  - model card generation (where enabled)
  - record lineage task at the end of the DAG

The **reference implementation** is `datasets/crime_dag.py`.

---

## Lineage task (end of DAG)

All dataset DAGs should include a terminal task called `record_lineage` that:

- pulls key stats from XCom (rows ingested/processed, drift, fairness)
- records a `LineageRecord` with GCS generation numbers for artifacts

Implementation:

- DAG wrapper: `utils/lineage_utils.py`
- Core: `src/shared/lineage.py`

---

## Debugging DAG import problems

If DAGs are blank in the Airflow UI, check imports and mounts:

1. Verify DAGs are mounted into the scheduler container:

```bash
cd data-pipeline/docker
docker compose --env-file ../.env -f docker-compose.airflow.yml exec airflow-scheduler \
  ls -la /opt/airflow/dags
```

2. Check Airflow UI:

- **Browse → Import Errors**

3. Validate DAG imports locally (requires Airflow installed in your local Python env):
