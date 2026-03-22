"""
Boston Pulse ML - Crime Navigate Training DAG.

Navigate crime risk scoring — weekly training DAG.

Schedule: every Sunday at 2 AM UTC (independent of data pipeline DAG)
Trigger also: on push to ml/ via GitHub Actions ml.yml

Task chain:
  load_features           → features_df cached to worker temp dir
    → build_targets       → joins features + danger_rate label → training_df
      → tune_hyperparams  → Optuna 20 trials, each a child MLflow run
        → train_lgbm      → random 80/20 split, regression_l1, MLflow logging
          → validate_model    → RMSE gate + SHAP — HALTS if fails
            → check_bias      → Fairlearn by district — HALTS if fails
              → push_to_registry  → dated always; latest/ only if both gates pass
                → score_cells     → inference on features_df (reuse cache, no re-read)
                  → publish_scores    → Firestore upsert
                    → pipeline_complete → Slack summary


Gate philosophy (inherited from data-pipeline):
  Gates FAIL the pipeline. They do not warn and continue.
  If validate_model or check_bias fails:
    - Dated model version is still written to registry (for forensics)
    - latest/ pointer is NOT updated
    - Production model is unchanged
    - Slack critical alert is sent
"""

from __future__ import annotations

import os
import pickle
import tempfile
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "crime_navigate_train"
DATASET = "crime_navigate"
BUCKET = os.getenv("GCS_BUCKET", "boston-pulse-data-pipeline")

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

# Worker-local cache directory for DataFrames and models
_cache_dir = tempfile.mkdtemp()


def _cfg() -> dict[str, Any]:
    """Load training configuration."""
    from shared.config_loader import load_training_config

    return load_training_config("crime_navigate_train")


def _cache_df(df: Any, name: str, date: str) -> str:
    """Cache a DataFrame to local pickle file."""
    path = f"{_cache_dir}/{name}_{date}.pkl"
    df.to_pickle(path)
    return path


def _load_cached_df(path: str) -> Any:
    """Load a cached DataFrame."""
    import pandas as pd

    return pd.read_pickle(path)


def _cache_model(model: Any, date: str) -> str:
    """Cache a model to local pickle file."""
    path = f"{_cache_dir}/model_{date}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def _load_cached_model(date: str) -> Any:
    """Load a cached model."""
    path = f"{_cache_dir}/model_{date}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Task Functions
# =============================================================================


def task_load_features(**context: Any) -> dict[str, Any]:
    """Load latest features per (h3_index, hour_bucket) from GCS."""
    from models.crime_navigate.feature_loader import load_features
    from shared.alerting import alert_training_start

    cfg = _cfg()
    execution_date = context["ds"]

    # Send training start alert
    alert_training_start(DATASET, execution_date, DAG_ID)

    df, result = load_features(execution_date, cfg, BUCKET)

    if not result.success:
        raise RuntimeError(f"Feature loading failed: {result.error}")

    # Cache features_df for reuse in build_targets AND score_cells
    context["ti"].xcom_push(key="features_df_path", value=_cache_df(df, "features", execution_date))
    return result.to_dict()


def task_build_targets(**context: Any) -> dict[str, Any]:
    """Build training matrix by joining features with danger_rate label."""
    from models.crime_navigate.target_builder import build_targets

    cfg = _cfg()
    execution_date = context["ds"]

    features_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="load_features", key="features_df_path")
    )

    df, result = build_targets(features_df, execution_date, cfg, BUCKET)

    if not result.success:
        raise RuntimeError(f"Target building failed: {result.error}")

    context["ti"].xcom_push(key="training_df_path", value=_cache_df(df, "training", execution_date))
    return result.to_dict()


def task_tune_hyperparams(**context: Any) -> dict[str, Any]:
    """Run Optuna hyperparameter search with random 80/20 split."""
    from models.crime_navigate.trainer import random_split
    from models.crime_navigate.tuner import tune_hyperparams
    from shared.mlflow_utils import get_or_create_run

    cfg = _cfg()
    execution_date = context["ds"]

    df = _load_cached_df(context["ti"].xcom_pull(task_ids="build_targets", key="training_df_path"))

    # Random 80/20 split — cross-sectional model, NOT temporal
    train_df, val_df = random_split(df, cfg)

    # Cache split DataFrames for reuse in train and validate
    context["ti"].xcom_push(key="train_df_path", value=_cache_df(train_df, "train", execution_date))
    context["ti"].xcom_push(key="val_df_path", value=_cache_df(val_df, "val", execution_date))

    # Create parent MLflow run
    parent_run_id = get_or_create_run(cfg, execution_date)

    best_params, result = tune_hyperparams(train_df, val_df, cfg, parent_run_id)

    context["ti"].xcom_push(key="best_params", value=best_params)
    context["ti"].xcom_push(key="mlflow_run_id", value=parent_run_id)

    return result.to_dict()


def task_train_lgbm(**context: Any) -> dict[str, Any]:
    """Train LightGBM with best hyperparameters."""
    from models.crime_navigate.trainer import train_model

    cfg = _cfg()
    execution_date = context["ds"]

    # Load cached train/val splits
    train_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="train_df_path")
    )
    val_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_df_path")
    )
    best_params = context["ti"].xcom_pull(task_ids="tune_hyperparams", key="best_params")
    run_id = context["ti"].xcom_pull(task_ids="tune_hyperparams", key="mlflow_run_id")

    model, model_path, result = train_model(train_df, val_df, best_params, cfg, run_id)

    # Cache model for downstream tasks
    _cache_model(model, execution_date)

    context["ti"].xcom_push(key="model_path", value=model_path)
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    context["ti"].xcom_push(key="training_result", value=result.to_dict())

    return result.to_dict()


def task_validate_model(**context: Any) -> dict[str, Any]:
    """Run RMSE gate and SHAP analysis."""
    from models.crime_navigate.validator import ValidationGateError, validate_model
    from shared.alerting import alert_gate_failure
    from shared.schemas import TrainingResult

    cfg = _cfg()
    execution_date = context["ds"]

    # Load cached validation set
    val_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_df_path")
    )
    run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")
    training_dict = context["ti"].xcom_pull(task_ids="train_lgbm", key="training_result")

    model = _load_cached_model(execution_date)

    # Reconstruct TrainingResult from XCom dict
    training_result = TrainingResult(**training_dict)

    try:
        result = validate_model(model, val_df, training_result, cfg, run_id)
        return result.to_dict()
    except ValidationGateError as e:
        alert_gate_failure(DATASET, execution_date, "RMSE/Overfit Gate", str(e), DAG_ID)
        raise


def task_check_bias(**context: Any) -> dict[str, Any]:
    """Run Fairlearn bias check."""
    from models.crime_navigate.bias_checker import BiasGateError, check_bias
    from shared.alerting import alert_gate_failure

    cfg = _cfg()
    execution_date = context["ds"]

    # Load cached validation set
    val_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_df_path")
    )
    run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")

    model = _load_cached_model(execution_date)

    try:
        result = check_bias(model, val_df, execution_date, cfg, BUCKET, run_id)
        return result.to_dict()
    except BiasGateError as e:
        alert_gate_failure(DATASET, execution_date, "Bias Gate", str(e), DAG_ID)
        raise


def task_push_to_registry(**context: Any) -> dict[str, Any]:
    """Push validated model to registry."""
    from shared.alerting import alert_model_pushed
    from shared.registry import ModelRegistry

    cfg = _cfg()
    execution_date = context["ds"]

    model_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_path")
    run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")
    val_result = context["ti"].xcom_pull(task_ids="validate_model")
    bias_result = context["ti"].xcom_pull(task_ids="check_bias")

    version = execution_date.replace("-", "")

    registry = ModelRegistry(cfg)
    metadata = {
        "execution_date": execution_date,
        "mlflow_run_id": run_id,
        "rmse_val": val_result["rmse_val"],
        "rmse_train": val_result["rmse_train"],
        "overfit_ratio": val_result["overfit_ratio"],
        "bias_passed": bias_result["passed"],
        "bias_worst_slice": bias_result.get("worst_slice"),
        "bias_worst_deviation_pct": bias_result.get("worst_deviation_pct"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "feature_list": cfg["features"]["input_columns"],
    }

    # Both gates passed to reach this task — safe to update latest/
    shap_path = val_result.get("shap_artifact_path")
    uri = registry.push(model_path, version, metadata, update_latest=True, shap_path=shap_path)

    # Send alert
    alert_model_pushed(DATASET, execution_date, version, uri, val_result["rmse_val"], DAG_ID)

    return {"model_uri": uri, "version": version}


def task_score_cells(**context: Any) -> dict[str, Any]:
    """Score all active H3 cells using cached features_df."""
    from models.crime_navigate.scorer import score_all_cells

    cfg = _cfg()
    execution_date = context["ds"]
    version = context["ti"].xcom_pull(task_ids="push_to_registry")["version"]

    # Reuse features_df from load_features — no GCS re-read
    features_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="load_features", key="features_df_path")
    )

    model = _load_cached_model(execution_date)

    scores_df, result = score_all_cells(model, features_df, execution_date, cfg, BUCKET, version)

    context["ti"].xcom_push(
        key="scores_df_path", value=_cache_df(scores_df, "scores", execution_date)
    )

    return result.to_dict()


def task_publish_scores(**context: Any) -> dict[str, Any]:
    """Publish scores to Firestore."""
    from models.crime_navigate.publisher import publish_scores
    from shared.alerting import alert_scores_published

    cfg = _cfg()
    execution_date = context["ds"]

    scores_df = _load_cached_df(
        context["ti"].xcom_pull(task_ids="score_cells", key="scores_df_path")
    )
    version = context["ti"].xcom_pull(task_ids="push_to_registry")["version"]

    result = publish_scores(scores_df, cfg, version, execution_date)

    # Send alert
    alert_scores_published(
        DATASET,
        execution_date,
        result.rows_upserted,
        scores_df["h3_index"].nunique(),
        result.duration_seconds,
        DAG_ID,
    )

    return result.to_dict()


def task_pipeline_complete(**context: Any) -> dict[str, Any]:
    """Send completion alert with summary."""
    from shared.alerting import alert_training_complete

    ti = context["ti"]
    execution_date = context["ds"]

    alert_training_complete(
        dataset=DATASET,
        execution_date=execution_date,
        train_result=ti.xcom_pull(task_ids="train_lgbm"),
        val_result=ti.xcom_pull(task_ids="validate_model"),
        bias_result=ti.xcom_pull(task_ids="check_bias"),
        score_result=ti.xcom_pull(task_ids="score_cells"),
        publish_result=ti.xcom_pull(task_ids="publish_scores"),
        dag_id=DAG_ID,
    )

    return {"status": "complete", "execution_date": execution_date}


# =============================================================================
# DAG Definition
# =============================================================================


def on_task_failure(context: Any) -> None:
    """Callback for task failures."""
    from shared.alerting import alert_gate_failure

    task_id = context["task_instance"].task_id
    execution_date = context["ds"]
    exception = context.get("exception", "Unknown error")

    alert_gate_failure(DATASET, execution_date, f"Task: {task_id}", str(exception), DAG_ID)


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Navigate crime risk scoring — weekly model training",
    schedule_interval="0 2 * * 0",  # Every Sunday 2 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["navigate", "ml", "crime", "training"],
) as dag:

    t_load = PythonOperator(
        task_id="load_features",
        python_callable=task_load_features,
        on_failure_callback=on_task_failure,
    )

    t_targets = PythonOperator(
        task_id="build_targets",
        python_callable=task_build_targets,
        on_failure_callback=on_task_failure,
    )

    t_tune = PythonOperator(
        task_id="tune_hyperparams",
        python_callable=task_tune_hyperparams,
        on_failure_callback=on_task_failure,
    )

    t_train = PythonOperator(
        task_id="train_lgbm",
        python_callable=task_train_lgbm,
        on_failure_callback=on_task_failure,
    )

    t_validate = PythonOperator(
        task_id="validate_model",
        python_callable=task_validate_model,
        on_failure_callback=on_task_failure,
    )

    t_bias = PythonOperator(
        task_id="check_bias",
        python_callable=task_check_bias,
        on_failure_callback=on_task_failure,
    )

    t_registry = PythonOperator(
        task_id="push_to_registry",
        python_callable=task_push_to_registry,
        on_failure_callback=on_task_failure,
    )

    t_score = PythonOperator(
        task_id="score_cells",
        python_callable=task_score_cells,
        on_failure_callback=on_task_failure,
    )

    t_publish = PythonOperator(
        task_id="publish_scores",
        python_callable=task_publish_scores,
        on_failure_callback=on_task_failure,
    )

    t_complete = PythonOperator(
        task_id="pipeline_complete",
        python_callable=task_pipeline_complete,
        trigger_rule="all_success",
    )

    # Task dependencies
    (
        t_load
        >> t_targets
        >> t_tune
        >> t_train
        >> t_validate
        >> t_bias
        >> t_registry
        >> t_score
        >> t_publish
        >> t_complete
    )
