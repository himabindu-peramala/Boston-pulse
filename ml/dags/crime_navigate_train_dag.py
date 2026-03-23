"""
Boston Pulse ML - Crime Navigate Training DAG.

Navigate crime risk scoring — weekly training DAG.

Schedule: every Sunday at 2 AM UTC (independent of data pipeline DAG)
Trigger also: on push to ml/ via GitHub Actions ml.yml

Task chain:
  load_features           → features_df written to GCS, path in XCom
    → build_targets       → training_df written to GCS, path in XCom
      → tune_hyperparams  → Optuna 20 trials, each a child MLflow run
        → train_lgbm      → model.lgb written to GCS, path in XCom
          → validate_model    → RMSE gate + SHAP — HALTS if fails
            → check_bias      → Fairlearn by district — HALTS if fails
              → push_to_registry  → dated always; latest/ only if both gates pass
                → score_cells     → inference on features_df (re-read from GCS)
                  → publish_scores    → Firestore upsert
                    → pipeline_complete → Slack summary

Inter-task artifact passing pattern:
  - DataFrames and model files → GCS
    Written to ml/run_artifacts/{run_id}/{name}.parquet or .lgb
    Read back by downstream tasks via GCS path from XCom
  - Small values (metrics, params, flags) → XCom directly
    Strings, dicts, numbers — never large objects

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
import tempfile
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "crime_navigate_train"
DATASET = "crime_navigate"
BUCKET = os.getenv("GCS_BUCKET", "boston-pulse-data-pipeline")

# GCS prefix for temporary run artifacts (DataFrames, models)
# These are written per-run and can be lifecycle-deleted after 30 days
RUN_ARTIFACTS_PREFIX = "ml/run_artifacts"

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}


# =============================================================================
# GCS Artifact Helpers
# =============================================================================


def _artifact_prefix(run_id: str) -> str:
    """GCS prefix for artifacts from this specific DAG run."""
    # Use run_id to namespace artifacts so parallel runs don't collide
    safe_run_id = run_id.replace(":", "_").replace("+", "_").replace(" ", "_")
    return f"{RUN_ARTIFACTS_PREFIX}/{safe_run_id}"


def _write_df_to_gcs(df: Any, name: str, run_id: str) -> str:
    """
    Write a DataFrame to GCS as parquet. Returns the full GCS path.

    Uses run_id in the path so multiple concurrent runs never collide.
    """
    from io import BytesIO
    from google.cloud import storage

    path = f"{_artifact_prefix(run_id)}/{name}.parquet"
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(path)

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    gcs_path = f"gs://{BUCKET}/{path}"
    return gcs_path


def _read_df_from_gcs(gcs_path: str) -> Any:
    """Read a DataFrame from a GCS parquet path."""
    import pandas as pd
    return pd.read_parquet(gcs_path)


def _write_model_to_gcs(model: Any, run_id: str) -> str:
    """
    Write a LightGBM model to GCS. Returns the full GCS path.

    LightGBM models have a native save_model() that produces a text file.
    This is more reliable than pickle and readable by any LightGBM version.
    """
    import tempfile
    from pathlib import Path
    from google.cloud import storage

    path = f"{_artifact_prefix(run_id)}/model.lgb"
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(path)

    # Save to a temp file then upload — LightGBM save_model requires a filepath
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as f:
        tmp_path = f.name

    try:
        model.save_model(tmp_path)
        blob.upload_from_filename(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    gcs_path = f"gs://{BUCKET}/{path}"
    return gcs_path


def _read_model_from_gcs(gcs_path: str) -> Any:
    """
    Read a LightGBM model from GCS.

    Downloads to a temp file, loads with lgb.Booster, then cleans up.
    """
    import tempfile
    from pathlib import Path
    import lightgbm as lgb
    from google.cloud import storage

    # Parse bucket and blob path from gs:// URI
    without_prefix = gcs_path[len("gs://"):]
    bucket_name, blob_path = without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as f:
        tmp_path = f.name

    try:
        blob.download_to_filename(tmp_path)
        model = lgb.Booster(model_file=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return model


def _cfg() -> dict[str, Any]:
    """Load training configuration."""
    from shared.config_loader import load_training_config
    return load_training_config("crime_navigate_train")


# =============================================================================
# Task Functions
# =============================================================================


def task_load_features(**context: Any) -> dict[str, Any]:
    """Load latest features per (h3_index, hour_bucket) from GCS."""
    from models.crime_navigate.feature_loader import load_features
    from shared.alerting import alert_training_start

    cfg = _cfg()
    execution_date = context["ds"]
    run_id = context["run_id"]

    alert_training_start(DATASET, execution_date, DAG_ID)

    df, result = load_features(execution_date, cfg, BUCKET)

    if not result.success:
        raise RuntimeError(f"Feature loading failed: {result.error}")

    # Write to GCS — path passed to downstream tasks via XCom
    gcs_path = _write_df_to_gcs(df, "features", run_id)
    context["ti"].xcom_push(key="features_gcs_path", value=gcs_path)

    return result.to_dict()


def task_build_targets(**context: Any) -> dict[str, Any]:
    """Build training matrix by joining features with danger_rate label."""
    from models.crime_navigate.target_builder import build_targets

    cfg = _cfg()
    execution_date = context["ds"]
    run_id = context["run_id"]

    # Read features DataFrame from GCS — deterministic path from XCom
    features_gcs_path = context["ti"].xcom_pull(
        task_ids="load_features", key="features_gcs_path"
    )
    features_df = _read_df_from_gcs(features_gcs_path)

    df, result = build_targets(features_df, execution_date, cfg, BUCKET)

    if not result.success:
        raise RuntimeError(f"Target building failed: {result.error}")

    gcs_path = _write_df_to_gcs(df, "training", run_id)
    context["ti"].xcom_push(key="training_gcs_path", value=gcs_path)

    return result.to_dict()


def task_tune_hyperparams(**context: Any) -> dict[str, Any]:
    """Run Optuna hyperparameter search with random 80/20 split."""
    from models.crime_navigate.trainer import random_split
    from models.crime_navigate.tuner import tune_hyperparams
    from shared.mlflow_utils import get_or_create_run

    cfg = _cfg()
    execution_date = context["ds"]
    run_id = context["run_id"]

    training_gcs_path = context["ti"].xcom_pull(
        task_ids="build_targets", key="training_gcs_path"
    )
    df = _read_df_from_gcs(training_gcs_path)

    # Random 80/20 split — cross-sectional model, NOT temporal
    train_df, val_df = random_split(df, cfg)

    # Write split DataFrames to GCS
    train_gcs = _write_df_to_gcs(train_df, "train", run_id)
    val_gcs = _write_df_to_gcs(val_df, "val", run_id)
    context["ti"].xcom_push(key="train_gcs_path", value=train_gcs)
    context["ti"].xcom_push(key="val_gcs_path", value=val_gcs)

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
    run_id = context["run_id"]

    train_df = _read_df_from_gcs(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="train_gcs_path")
    )
    val_df = _read_df_from_gcs(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_gcs_path")
    )
    best_params = context["ti"].xcom_pull(task_ids="tune_hyperparams", key="best_params")
    mlflow_run_id = context["ti"].xcom_pull(task_ids="tune_hyperparams", key="mlflow_run_id")

    model, model_path, result = train_model(train_df, val_df, best_params, cfg, mlflow_run_id)

    # Write model to GCS — use LightGBM native format, not pickle
    model_gcs_path = _write_model_to_gcs(model, run_id)

    context["ti"].xcom_push(key="model_gcs_path", value=model_gcs_path)
    # Also pass the local model_path for push_to_registry which needs a local file
    context["ti"].xcom_push(key="model_local_path", value=model_path)
    context["ti"].xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    context["ti"].xcom_push(key="training_result", value=result.to_dict())

    return result.to_dict()


def task_validate_model(**context: Any) -> dict[str, Any]:
    """Run RMSE gate and SHAP analysis."""
    from models.crime_navigate.validator import ValidationGateError, validate_model
    from shared.alerting import alert_gate_failure
    from shared.schemas import TrainingResult

    cfg = _cfg()
    execution_date = context["ds"]

    val_df = _read_df_from_gcs(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_gcs_path")
    )
    mlflow_run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")
    training_dict = context["ti"].xcom_pull(task_ids="train_lgbm", key="training_result")

    # Load model from GCS — always works regardless of which subprocess runs this
    model_gcs_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_gcs_path")
    model = _read_model_from_gcs(model_gcs_path)

    training_result = TrainingResult(**training_dict)

    try:
        result = validate_model(model, val_df, training_result, cfg, mlflow_run_id)
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

    val_df = _read_df_from_gcs(
        context["ti"].xcom_pull(task_ids="tune_hyperparams", key="val_gcs_path")
    )
    mlflow_run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")

    model_gcs_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_gcs_path")
    model = _read_model_from_gcs(model_gcs_path)

    try:
        result = check_bias(model, val_df, execution_date, cfg, BUCKET, mlflow_run_id)
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

    # Use the local model path written by train_lgbm in the same process run
    # If that is gone (retry), re-download from GCS
    model_local_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_local_path")
    model_gcs_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_gcs_path")

    import os
    if not model_local_path or not os.path.exists(model_local_path):
        # Local path gone — download from GCS to a fresh temp file
        import tempfile
        from pathlib import Path
        from google.cloud import storage

        without_prefix = model_gcs_path[len("gs://"):]
        bucket_name, blob_path = without_prefix.split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)

        tmp = tempfile.NamedTemporaryFile(suffix=".lgb", delete=False)
        tmp.close()
        blob.download_to_filename(tmp.name)
        model_local_path = tmp.name

    mlflow_run_id = context["ti"].xcom_pull(task_ids="train_lgbm", key="mlflow_run_id")
    val_result = context["ti"].xcom_pull(task_ids="validate_model")
    bias_result = context["ti"].xcom_pull(task_ids="check_bias")

    version = execution_date.replace("-", "")

    registry = ModelRegistry(cfg)
    metadata = {
        "execution_date": execution_date,
        "mlflow_run_id": mlflow_run_id,
        "rmse_val": val_result["rmse_val"],
        "rmse_train": val_result["rmse_train"],
        "overfit_ratio": val_result["overfit_ratio"],
        "bias_passed": bias_result["passed"],
        "bias_worst_slice": bias_result.get("worst_slice"),
        "bias_worst_deviation_pct": bias_result.get("worst_deviation_pct"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "feature_list": cfg["features"]["input_columns"],
    }

    shap_path = val_result.get("shap_artifact_path")
    uri = registry.push(model_local_path, version, metadata, update_latest=True, shap_path=shap_path)

    alert_model_pushed(DATASET, execution_date, version, uri, val_result["rmse_val"], DAG_ID)

    return {"model_uri": uri, "version": version}


def task_score_cells(**context: Any) -> dict[str, Any]:
    """Score all active H3 cells."""
    from models.crime_navigate.scorer import score_all_cells

    cfg = _cfg()
    execution_date = context["ds"]
    version = context["ti"].xcom_pull(task_ids="push_to_registry")["version"]

    # Re-read features from GCS — same data, guaranteed available
    features_gcs_path = context["ti"].xcom_pull(
        task_ids="load_features", key="features_gcs_path"
    )
    features_df = _read_df_from_gcs(features_gcs_path)

    model_gcs_path = context["ti"].xcom_pull(task_ids="train_lgbm", key="model_gcs_path")
    model = _read_model_from_gcs(model_gcs_path)

    scores_df, result = score_all_cells(model, features_df, execution_date, cfg, BUCKET, version)

    # Write scores to GCS for publish_scores
    run_id = context["run_id"]
    scores_gcs_path = _write_df_to_gcs(scores_df, "scores", run_id)
    context["ti"].xcom_push(key="scores_gcs_path", value=scores_gcs_path)

    return result.to_dict()


def task_publish_scores(**context: Any) -> dict[str, Any]:
    """Publish scores to Firestore."""
    from models.crime_navigate.publisher import publish_scores
    from shared.alerting import alert_scores_published

    cfg = _cfg()
    execution_date = context["ds"]

    scores_gcs_path = context["ti"].xcom_pull(
        task_ids="score_cells", key="scores_gcs_path"
    )
    scores_df = _read_df_from_gcs(scores_gcs_path)
    version = context["ti"].xcom_pull(task_ids="push_to_registry")["version"]

    result = publish_scores(scores_df, cfg, version, execution_date)

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