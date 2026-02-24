Schemas – Boston Pulse Data Pipeline
===================================

This folder contains **schema definitions** used to validate dataset artifacts at multiple pipeline stages.

If you’re new, start with:

- Quickstart: [`../README.md`](../README.md)
- Contributor guide: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)

---

## What schemas are for

Schemas exist to prevent silent upstream changes from breaking downstream systems.

In this pipeline:

- validation is **strict** by design (fail the DAG on schema violations)
- schemas are typically defined per dataset and per stage:
  - `raw`
  - `processed`
  - `features`

The exact enforcement logic lives under `src/validation/` (e.g., schema registry + schema enforcer).

---

## Expected layout

Schemas are organized by dataset:

```text
schemas/
├── crime/
├── service_311/
├── food_inspections/
└── cityscore/
└── ...
```

Within each dataset directory, the project commonly uses one schema per stage (file names may vary by dataset):

```text
schemas/<dataset>/
  raw_schema.json
  processed_schema.json
  features_schema.json
```

If you add a new dataset, create `schemas/<dataset>/` and add the three stage schemas.

---

## How schemas are used in DAGs

Dataset DAGs enforce schemas at multiple checkpoints:

1. After ingestion (raw artifact)
2. After preprocessing (processed artifact)
3. After feature engineering (features artifact)

See `dags/datasets/crime_dag.py` for the reference ordering.

---

## Adding / updating schemas

When you:

- add a new column
- change a type
- adjust allowed ranges/domains

you should:

1. Update the relevant schema under `schemas/<dataset>/`
2. Add/update tests under `tests/unit/validation/` or dataset-specific tests
3. Validate by running:

```bash
cd data-pipeline
make test
```

---

## Reproducibility note (GCS generations)

Schema and data artifacts are versioned in GCS via **object generations**.

At the end of a run, lineage captures which schema and data generations were used so you can reproduce and debug historical runs:

- Core lineage: `src/shared/lineage.py`
- DAG wrapper: `dags/utils/lineage_utils.py`
