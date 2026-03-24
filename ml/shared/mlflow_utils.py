"""
Boston Pulse ML - MLflow Utilities.

Helpers for MLflow experiment tracking and run management.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: dict[str, Any]) -> None:
    """
    Configure MLflow tracking URI from environment.
    Call once at DAG start or before training.

    Args:
        cfg: Training config with mlflow.experiment_name
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)

    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "default-experiment")
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow configured: uri={uri}, experiment={experiment_name}")


def get_or_create_run(cfg: dict[str, Any], execution_date: str) -> str:
    """
    Create (or resume) the parent MLflow run for this training execution.

    Args:
        cfg: Training config
        execution_date: Execution date (YYYY-MM-DD)

    Returns:
        MLflow run ID
    """
    setup_mlflow(cfg)

    version_prefix = cfg.get("model", {}).get("version_prefix", "model")
    run_name = f"{version_prefix}_{execution_date}"

    run = mlflow.start_run(
        run_name=run_name,
        tags={
            "execution_date": execution_date,
            "model_name": cfg.get("model", {}).get("name", "unknown"),
        },
    )
    run_id = run.info.run_id

    # End immediately — child tasks reopen via run_id
    mlflow.end_run()

    logger.info(f"Created MLflow run: {run_name} (id={run_id})")
    return run_id


def log_params_safe(params: dict[str, Any]) -> None:
    """
    Log parameters to MLflow, handling nested dicts and long values.

    Args:
        params: Dictionary of parameters to log
    """
    for key, value in params.items():
        try:
            if isinstance(value, dict):
                # Flatten nested dicts
                for subkey, subval in value.items():
                    mlflow.log_param(f"{key}.{subkey}", str(subval)[:250])
            elif isinstance(value, list | tuple):
                mlflow.log_param(key, str(value)[:250])
            else:
                mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log param {key}: {e}")


def log_metrics_safe(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log metrics to MLflow with error handling.

    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number
    """
    for key, value in metrics.items():
        try:
            if value is not None and not (
                isinstance(value, float) and (value != value)
            ):  # NaN check
                mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")


def get_run_artifact_uri(run_id: str) -> str:
    """Get the artifact URI for a run."""
    run = mlflow.get_run(run_id)
    return run.info.artifact_uri


def log_model_info(
    run_id: str,
    model_path: str,
    feature_columns: list[str],
    train_rmse: float,
    val_rmse: float,
) -> None:
    """
    Log model information to MLflow run.

    Args:
        run_id: MLflow run ID
        model_path: Local path to saved model
        feature_columns: List of feature column names
        train_rmse: Training RMSE
        val_rmse: Validation RMSE
    """
    import json

    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("features", json.dumps(feature_columns))
        mlflow.log_metric("rmse_train", train_rmse)
        mlflow.log_metric("rmse_val", val_rmse)
