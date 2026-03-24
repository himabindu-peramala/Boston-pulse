# Boston Pulse ML Training Pipeline

Weekly training pipeline for Navigate crime risk scoring using LightGBM.

## Architecture

```
ml/
├── models/crime_navigate/     # Navigate crime risk model
│   ├── feature_loader.py      # Load features from GCS
│   ├── target_builder.py      # Build crime_count labels
│   ├── tuner.py               # Optuna hyperparameter search
│   ├── trainer.py             # LightGBM training
│   ├── validator.py           # RMSE gate + SHAP analysis
│   ├── bias_checker.py        # Fairlearn bias detection
│   ├── scorer.py              # Risk score inference
│   └── publisher.py           # Firestore upsert
├── shared/                    # Reusable utilities
├── dags/                      # Airflow DAGs
├── configs/                   # Training configs (YAML)
└── tests/                     # Unit and integration tests
    └── unit/shared/           # Tests for shared/ (config, GCS, MLflow, registry, …)
```

## Quick Start

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Start local MLflow
make mlflow-local

# Run linting
make lint
```

## Design Principles

1. **Separation of concerns**: `ml/` reads from `data-pipeline/` via GCS paths only
2. **Gates block deployment**: RMSE and bias gates fail the pipeline, not warn
3. **Full retraining weekly**: All historical data from 2023-01-01
4. **Config-driven**: All thresholds in YAML, no hardcoded values

## GCS Path Contract

```
gs://boston-pulse-data-pipeline/
├── features/crime_navigate/dt=YYYY-MM-DD/features.parquet  # Read by ML
├── processed/crime_navigate/dt=YYYY-MM-DD/data.parquet     # Read by ML
├── ml/scores/crime_navigate/dt=YYYY-MM-DD/scores.parquet   # Written by ML
└── ml/bias_reports/crime_navigate/dt=YYYY-MM-DD/report.json
```

## Training DAG Schedule

- **Schedule**: Every Sunday at 2 AM UTC
- **Duration**: ~30 minutes
- **Output**: Risk scores in Firestore `h3_scores` collection
