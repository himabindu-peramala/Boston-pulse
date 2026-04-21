"""
Boston Pulse ML - Daily Drift Monitoring DAG.

Runs at 04:00 UTC daily (2 hours after ML training completes).

For each monitored model:
  1. Load training baseline sample from GCS
  2. Load last 7 days of processed/features data from data pipeline
  3. Run Evidently DataDriftPreset — produce HTML report
  4. Compute summary stats (n features drifted, max PSI, overall drift score)
  5. Upload HTML + summary JSON to GCS
  6. Emit metrics to Cloud Monitoring
  7. Push a Slack alert with report link
  8. Return summary via XCom for the retrain-trigger task
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DAG_ID = "ml_drift_monitoring"
SCHEDULE = "0 4 * * *"  # Daily at 04:00 UTC (after ML training at 02:00)

default_args = {
    "owner": "ml-ops",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# Models to monitor — extend this list for additional models
MONITORED_MODELS = [
    {
        "name": "crime_navigate",
        "baseline_model_name": "crime-risk",
        "feature_prefix": "features/crime_navigate",
        "lookback_days": 7,
    },
]

# GCS bucket configuration
DATA_BUCKET = os.environ.get("DATA_BUCKET", "boston-pulse-data-pipeline")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "boston-pulse-mlflow-artifacts")


def run_evidently_drift(**context: Any) -> dict[str, Any]:
    """Compute drift using Evidently AI."""
    import pandas as pd
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report
    from google.cloud import storage

    execution_date = context["ds"]
    model = MONITORED_MODELS[0]  # Extend loop for multiple models later
    model_name = model["name"]
    baseline_model_name = model["baseline_model_name"]
    lookback_days = model["lookback_days"]

    client = storage.Client()
    data_bucket = client.bucket(DATA_BUCKET)
    artifact_bucket = client.bucket(ARTIFACT_BUCKET)

    # 1. Load training baseline (reference)
    ref_blob = artifact_bucket.blob(
        f"monitoring/baselines/{baseline_model_name}/latest/feature_sample.parquet"
    )
    if not ref_blob.exists():
        raise RuntimeError(
            "No training baseline found. Run backfill_baseline.py or train a new model first."
        )
    ref_bytes = ref_blob.download_as_bytes()
    reference_df = pd.read_parquet(io.BytesIO(ref_bytes))
    logger.info(f"Loaded reference baseline: {len(reference_df)} rows")

    # 2. Load last N days of features (current)
    end_date = datetime.strptime(execution_date, "%Y-%m-%d")
    current_frames = []
    for d in range(lookback_days):
        date_str = (end_date - timedelta(days=d)).strftime("%Y-%m-%d")
        blob_path = f"{model['feature_prefix']}/dt={date_str}/features.parquet"
        blob = data_bucket.blob(blob_path)
        if not blob.exists():
            logger.warning(f"Missing feature data for {date_str}")
            continue
        current_frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
        logger.info(f"Loaded features for {date_str}")

    if not current_frames:
        raise RuntimeError(f"No current feature data in last {lookback_days} days")

    current_df = pd.concat(current_frames, ignore_index=True)
    logger.info(f"Total current data: {len(current_df)} rows from {len(current_frames)} days")

    # Align columns — use intersection
    common_cols = list(set[Any](reference_df.columns) & set(current_df.columns))
    if not common_cols:
        raise RuntimeError("No common columns between reference and current data")

    reference_df = reference_df[common_cols]
    current_df = current_df[common_cols]
    logger.info(f"Analyzing {len(common_cols)} common columns")

    # 3. Run Evidently
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )
    report.run(reference_data=reference_df, current_data=current_df)

    # 4. Extract summary
    report_dict = report.as_dict()
    drift_metric = next(
        (m for m in report_dict["metrics"] if "DataDriftTable" in m.get("metric", "")),
        None,
    )
    if drift_metric:
        result = drift_metric.get("result", {})
        n_drifted = result.get("number_of_drifted_columns", 0)
        n_columns = result.get("number_of_columns", 1)
        drift_share = result.get("share_of_drifted_columns", 0)
        dataset_drift = result.get("dataset_drift", False)
    else:
        n_drifted, n_columns, drift_share, dataset_drift = 0, 0, 0.0, False

    # Per-feature drift details
    per_feature: dict[str, dict[str, Any]] = {}
    if drift_metric:
        drift_by_columns = drift_metric.get("result", {}).get("drift_by_columns", {})
        for col, data in drift_by_columns.items():
            per_feature[col] = {
                "drift_detected": data.get("drift_detected", False),
                "drift_score": data.get("drift_score", 0),
                "stattest_name": data.get("stattest_name", "unknown"),
            }

    summary: dict[str, Any] = {
        "execution_date": execution_date,
        "model": model_name,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "n_features_total": n_columns,
        "n_features_drifted": n_drifted,
        "drift_share": drift_share,
        "dataset_drift_detected": dataset_drift,
        "per_feature": per_feature,
    }

    # 5. Upload HTML + summary to GCS
    html_path = f"/tmp/drift_report_{execution_date}.html"
    report.save_html(html_path)

    html_gcs_key = f"monitoring/drift_reports/{model_name}/dt={execution_date}/report.html"
    summary_gcs_key = f"monitoring/drift_reports/{model_name}/dt={execution_date}/summary.json"

    artifact_bucket.blob(html_gcs_key).upload_from_filename(html_path)
    artifact_bucket.blob(summary_gcs_key).upload_from_string(
        json.dumps(summary, indent=2, default=str),
        content_type="application/json",
    )

    # Also copy HTML to latest/ for easy Slack linking
    artifact_bucket.blob(
        f"monitoring/drift_reports/{model_name}/latest/report.html"
    ).upload_from_filename(html_path)

    summary["html_gcs_uri"] = f"gs://{ARTIFACT_BUCKET}/{html_gcs_key}"
    summary["summary_gcs_uri"] = f"gs://{ARTIFACT_BUCKET}/{summary_gcs_key}"

    logger.info(
        f"Drift report complete: {n_drifted}/{n_columns} features drifted "
        f"({drift_share:.1%}), dataset_drift={dataset_drift}"
    )
    return summary


def emit_cloud_monitoring_metrics(**context: Any) -> dict[str, Any]:
    """Emit drift metrics to Google Cloud Monitoring."""
    summary = context["ti"].xcom_pull(task_ids="run_evidently_drift")
    if not summary:
        logger.warning("No drift summary available — skipping metrics emission")
        return {"emitted": False, "reason": "no_summary"}

    try:
        from google.cloud import monitoring_v3

        project_id = os.environ.get("GCP_PROJECT_ID", "bostonpulse")
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        now = time.time()

        def _create_time_series(
            metric_type: str, value: float, labels: dict[str, str]
        ) -> monitoring_v3.TimeSeries:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/bostonpulse/ml/{metric_type}"
            for k, v in labels.items():
                series.metric.labels[k] = str(v)
            series.resource.type = "global"
            point = monitoring_v3.Point()
            point.value.double_value = float(value)
            point.interval.end_time.seconds = int(now)
            series.points = [point]
            return series

        batch = [
            _create_time_series("drift_share", summary["drift_share"], {"model": summary["model"]}),
            _create_time_series(
                "n_features_drifted",
                summary["n_features_drifted"],
                {"model": summary["model"]},
            ),
            _create_time_series(
                "dataset_drift_detected",
                1.0 if summary["dataset_drift_detected"] else 0.0,
                {"model": summary["model"]},
            ),
        ]

        # Per-feature drift scores
        for feat, d in summary.get("per_feature", {}).items():
            batch.append(
                _create_time_series(
                    "feature_drift_score",
                    d.get("drift_score", 0),
                    {"model": summary["model"], "feature": feat},
                )
            )

        client.create_time_series(name=project_name, time_series=batch)
        logger.info(f"Emitted {len(batch)} metrics to Cloud Monitoring")
        return {"emitted": True, "n_metrics": len(batch)}

    except Exception as e:
        logger.warning(f"Failed to emit Cloud Monitoring metrics: {e}")
        return {"emitted": False, "error": str(e)}


def alert_drift_report_ready(**context: Any) -> dict[str, Any]:
    """Send Slack alert with drift report summary."""
    summary = context["ti"].xcom_pull(task_ids="run_evidently_drift")
    if not summary:
        logger.warning("No drift summary available — skipping alert")
        return {"alerted": False, "reason": "no_summary"}

    try:
        # Import from ml shared module
        import sys

        sys.path.insert(0, "/home/airflow/gcs/dags/ml")
        from shared.alerting import alert_drift_report

        alert_drift_report(summary)
        logger.info("Drift report alert sent to Slack")
        return {"alerted": True}
    except ImportError:
        # Fallback: log the summary if alerting module not available
        logger.warning("Alerting module not available — logging summary instead")
        logger.info(f"Drift Report Summary: {json.dumps(summary, indent=2, default=str)}")
        return {"alerted": False, "reason": "module_not_available"}
    except Exception as e:
        logger.warning(f"Failed to send drift alert: {e}")
        return {"alerted": False, "error": str(e)}


# =============================================================================
# Retrain Trigger Functions
# =============================================================================


def _already_triggered_recently(cooldown_hours: int, model_name: str) -> bool:
    """Check GCS marker to respect cooldown."""

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(ARTIFACT_BUCKET).blob(
        f"monitoring/retrain_triggers/{model_name}/last_trigger.txt"
    )
    if not blob.exists():
        return False

    try:
        last = datetime.fromisoformat(blob.download_as_text().strip())
        # Ensure last is timezone-aware
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return (datetime.now(UTC) - last) < timedelta(hours=cooldown_hours)
    except Exception as e:
        logger.warning(f"Failed to parse last trigger time: {e}")
        return False


def _mark_triggered(model_name: str) -> None:
    """Mark that a retrain was triggered (for cooldown tracking)."""

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(ARTIFACT_BUCKET).blob(
        f"monitoring/retrain_triggers/{model_name}/last_trigger.txt"
    )
    blob.upload_from_string(datetime.now(UTC).isoformat())
    logger.info(f"Marked retrain trigger for {model_name}")


def maybe_trigger_retrain(**context: Any) -> dict[str, Any]:
    """
    Evaluate drift thresholds and trigger retrain if exceeded.

    Triggers GitHub Actions workflow_dispatch on ml.yml.
    """
    import sys

    import requests
    import yaml

    summary = context["ti"].xcom_pull(task_ids="run_evidently_drift")
    if not summary:
        return {"triggered": False, "reason": "no drift summary"}

    # Load thresholds from config
    config_paths = [
        "/home/airflow/gcs/dags/data-pipeline/configs/environments/base.yaml",
        "/opt/airflow/dags/data-pipeline/configs/environments/base.yaml",
        "configs/environments/base.yaml",
    ]

    thresholds = None
    for config_path in config_paths:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            thresholds = (
                cfg.get("ml_monitoring", {}).get("crime_navigate", {}).get("retrain_triggers")
            )
            if thresholds:
                logger.info(f"Loaded retrain thresholds from {config_path}")
                break
        except FileNotFoundError:
            continue

    if not thresholds:
        # Use defaults if config not found
        logger.warning("Config not found — using default retrain thresholds")
        thresholds = {
            "max_feature_drift_score": 0.25,
            "max_drift_share": 0.30,
            "on_dataset_drift": True,
            "cooldown_hours": 24,
        }

    model_name = summary["model"]

    # Check cooldown
    if _already_triggered_recently(thresholds["cooldown_hours"], model_name):
        logger.info(f"Cooldown active for {model_name} — skipping retrain trigger")
        return {"triggered": False, "reason": "cooldown active"}

    # Evaluate triggers
    reasons = []

    # Check max feature drift score
    max_feature_score = max(
        (d["drift_score"] for d in summary.get("per_feature", {}).values()),
        default=0,
    )
    if max_feature_score > thresholds["max_feature_drift_score"]:
        reasons.append(
            f"max feature drift {max_feature_score:.3f} > "
            f"{thresholds['max_feature_drift_score']}"
        )

    # Check drift share
    if summary["drift_share"] > thresholds["max_drift_share"]:
        reasons.append(
            f"drift share {summary['drift_share']:.2%} > " f"{thresholds['max_drift_share']:.0%}"
        )

    # Check dataset drift flag
    if summary["dataset_drift_detected"] and thresholds["on_dataset_drift"]:
        reasons.append("Evidently flagged dataset drift")

    if not reasons:
        logger.info("All drift thresholds within bounds — no retrain needed")
        return {"triggered": False, "reason": "all thresholds within bounds"}

    logger.warning(f"Retrain triggered! Reasons: {reasons}")

    # Trigger GitHub workflow_dispatch
    gh_token = os.environ.get("GITHUB_PAT")
    if not gh_token:
        logger.error("GITHUB_PAT not set — cannot trigger retrain workflow")
        return {
            "triggered": False,
            "reason": "GITHUB_PAT not configured",
            "would_trigger_reasons": reasons,
        }

    owner = os.environ.get("GH_OWNER", "your-org")
    repo = os.environ.get("GH_REPO", "Boston-pulse")

    try:
        response = requests.post(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/ml.yml/dispatches",
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "ref": "main",
                "inputs": {
                    "execution_date": context["ds"],
                    "skip_training": "false",
                    "skip_image_build": "true",
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        logger.info(f"Successfully triggered retrain workflow for {model_name}")

    except requests.RequestException as e:
        logger.error(f"Failed to trigger GitHub workflow: {e}")
        return {
            "triggered": False,
            "reason": f"GitHub API error: {e}",
            "would_trigger_reasons": reasons,
        }

    # Mark triggered for cooldown
    _mark_triggered(model_name)

    # Send Slack alert
    try:
        sys.path.insert(0, "/home/airflow/gcs/dags/ml")
        from shared.alerting import alert_retrain_triggered

        alert_retrain_triggered(model=model_name, reasons=reasons, summary=summary)
    except Exception as e:
        logger.warning(f"Failed to send retrain alert: {e}")

    return {
        "triggered": True,
        "reasons": reasons,
        "max_feature_drift_score": max_feature_score,
        "drift_share": summary["drift_share"],
    }


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Daily drift monitoring for ML models using Evidently AI",
    schedule_interval=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml-ops", "monitoring", "drift", "evidently"],
) as dag:
    t_drift = PythonOperator(
        task_id="run_evidently_drift",
        python_callable=run_evidently_drift,
    )

    t_metrics = PythonOperator(
        task_id="emit_cloud_monitoring",
        python_callable=emit_cloud_monitoring_metrics,
    )

    t_alert = PythonOperator(
        task_id="alert_drift_report",
        python_callable=alert_drift_report_ready,
    )

    t_retrain = PythonOperator(
        task_id="maybe_trigger_retrain",
        python_callable=maybe_trigger_retrain,
    )

    # Task dependencies: run drift first, then emit metrics, alert, and retrain check in parallel
    t_drift >> [t_metrics, t_alert, t_retrain]
