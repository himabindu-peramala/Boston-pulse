"""
Boston Pulse ML - Test Fixtures.

Shared fixtures for unit and integration tests.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cfg() -> dict[str, Any]:
    """Sample training configuration for tests."""
    return {
        "model": {
            "name": "crime-navigate-model",
            "version_prefix": "crime_navigate",
            "objective": "regression_l1",
            "metric": "rmse",
        },
        "data": {
            "bucket": "test-bucket",
            "features_prefix": "features/crime_navigate",
            "processed_prefix": "processed/crime_navigate",
            "scores_prefix": "ml/scores/crime_navigate",
            "bias_reports_prefix": "ml/bias_reports/crime_navigate",
            "history_start": "2023-01-01",
        },
        "features": {
            "input_columns": [
                "weighted_score_3d",
                "weighted_score_30d",
                "weighted_score_90d",
                "incident_count_30d",
                "trend_3v10",
                "trend_10v30",
                "trend_30v90",
                "gun_incident_count_30d",
                "high_severity_ratio_30d",
                "night_score_ratio",
                "evening_score_ratio",
                "weekend_score_ratio",
                "neighbor_weighted_score_30d",
                "neighbor_trend_3v10",
                "neighbor_gun_count_30d",
                "hour_bucket",
            ],
            "categorical_columns": ["hour_bucket"],
            "target_column": "danger_rate",
            "join_key": ["h3_index", "hour_bucket"],
        },
        "training": {
            "val_fraction": 0.20,
            "random_seed": 42,
            "early_stopping_rounds": 10,
            "verbose_eval": -1,
        },
        "tuning": {
            "n_trials": 3,
            "direction": "minimize",
            "timeout_seconds": 60,
            "search_space": {
                "num_leaves": [10, 30],
                "learning_rate": [0.05, 0.2],
                "min_child_samples": [10, 20],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "n_estimators": [50, 100],
                "reg_alpha": [0.0, 0.5],
                "reg_lambda": [0.0, 0.5],
            },
        },
        "validation": {
            "rmse_gate": 1.0,
            "overfit_ratio_gate": 2.0,
            "min_val_cells": 10,
        },
        "bias": {
            "slice_dimensions": ["district", "hour_bucket"],
            "max_slice_rmse_deviation": 0.30,
            "min_slice_size": 5,
            "max_slice_rmse_multiplier": 3.0,
        },
        "scoring": {
            "scale_within_bucket": True,
            "tiers": {
                "low": [0, 33],
                "medium": [33, 66],
                "high": [66, 100],
            },
            "active_cell_lookback_days": 30,
        },
        "registry": {
            "location": "us-east1",
            "project": "test-project",
            "repository": "ml-models",
            "package": "navigate/crime-risk",
            "artifact_bucket": "test-artifacts",
        },
        "mlflow": {
            "experiment_name": "test-experiment",
        },
        "firestore": {
            "collection": "test_h3_scores",
            "batch_size": 100,
        },
        "promotion": {
            "tolerance": 0.02,
            "force_promote": False,
        },
    }


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """
    Sample features DataFrame for tests.

    One row per (h3_index, hour_bucket) — matches the cross-sectional design.
    """
    np.random.seed(42)

    h3_cells = [f"h3_{i:03d}" for i in range(20)]
    hour_buckets = [0, 1, 2, 3, 4, 5]

    # Create all combinations of h3_index and hour_bucket
    rows = []
    for h3 in h3_cells:
        for hb in hour_buckets:
            rows.append({"h3_index": h3, "hour_bucket": hb})

    df = pd.DataFrame(rows)
    n_rows = len(df)

    # Add feature columns
    df["weighted_score_3d"] = np.random.uniform(0, 10, n_rows)
    df["weighted_score_30d"] = np.random.uniform(0, 20, n_rows)
    df["weighted_score_90d"] = np.random.uniform(0, 50, n_rows)
    df["incident_count_30d"] = np.random.randint(0, 30, n_rows)
    df["trend_3v10"] = np.random.uniform(-1, 1, n_rows)
    df["trend_10v30"] = np.random.uniform(-1, 1, n_rows)
    df["trend_30v90"] = np.random.uniform(-1, 1, n_rows)
    df["gun_incident_count_30d"] = np.random.randint(0, 5, n_rows)
    df["high_severity_ratio_30d"] = np.random.uniform(0, 1, n_rows)
    df["night_score_ratio"] = np.random.uniform(0, 1, n_rows)
    df["evening_score_ratio"] = np.random.uniform(0, 1, n_rows)
    df["weekend_score_ratio"] = np.random.uniform(0, 1, n_rows)
    df["neighbor_weighted_score_30d"] = np.random.uniform(0, 15, n_rows)
    df["neighbor_trend_3v10"] = np.random.uniform(-1, 1, n_rows)
    df["neighbor_gun_count_30d"] = np.random.randint(0, 10, n_rows)
    df["computed_date"] = "2024-01-15"

    return df


@pytest.fixture
def sample_training_df(sample_features_df: pd.DataFrame) -> pd.DataFrame:
    """Sample training DataFrame with danger_rate label."""
    df = sample_features_df.copy()
    np.random.seed(42)

    # Generate danger_rate based on features (with some noise)
    df["danger_rate"] = (
        df["weighted_score_30d"] * 0.1
        + df["incident_count_30d"] * 0.05
        + np.random.exponential(0.5, len(df))
    ).clip(0)

    df["incident_count"] = np.random.randint(0, 50, len(df))
    df["district"] = np.random.choice(["A1", "B2", "C3", "D4"], len(df))

    return df


@pytest.fixture
def sample_val_df(sample_training_df: pd.DataFrame) -> pd.DataFrame:
    """Sample validation DataFrame (20% of training data)."""
    np.random.seed(42)
    return sample_training_df.sample(frac=0.2, random_state=42)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_gcs_loader(mocker: Any) -> Any:
    """Mock GCSLoader for tests."""
    mock = mocker.patch("shared.gcs_loader.GCSLoader")
    return mock


@pytest.fixture
def mock_mlflow(mocker: Any) -> Any:
    """Mock MLflow for tests."""
    mock = mocker.patch("mlflow.start_run")
    mock.return_value.__enter__ = mocker.Mock(
        return_value=mocker.Mock(info=mocker.Mock(run_id="test-run-id"))
    )
    mock.return_value.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("mlflow.log_param")
    mocker.patch("mlflow.log_params")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.log_artifact")
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.set_experiment")
    mocker.patch("mlflow.end_run")
    mocker.patch("mlflow.lightgbm.log_model")

    return mock


@pytest.fixture
def mock_firestore(mocker: Any) -> Any:
    """Mock Firestore client for tests."""
    mock = mocker.patch("google.cloud.firestore.Client")
    return mock


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_env() -> Generator[None, None, None]:
    """Set up test environment variables."""
    original_env = os.environ.copy()

    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///test_mlflow.db"
    os.environ["GCS_BUCKET"] = "test-bucket"
    os.environ["GCP_PROJECT_ID"] = "test-project"

    yield

    os.environ.clear()
    os.environ.update(original_env)
