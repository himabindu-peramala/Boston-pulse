"""
Boston Pulse - Airflow Callbacks

Task and DAG callbacks for Airflow operations.
Provides consistent error handling, alerting, and logging across all DAGs.

Usage:
    from dags.utils.callbacks import (
        on_task_failure,
        on_task_success,
        on_dag_failure,
        on_dag_success,
    )

    with DAG(
        ...
        on_failure_callback=on_dag_failure,
        on_success_callback=on_dag_success,
    ) as dag:
        task = PythonOperator(
            ...
            on_failure_callback=on_task_failure,
            on_success_callback=on_task_success,
        )
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from airflow.models import TaskInstance

from src.alerting import send_alert

logger = logging.getLogger(__name__)


def _extract_context_info(context: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant information from Airflow context."""
    ti: TaskInstance | None = context.get("task_instance") or context.get("ti")
    dag_run = context.get("dag_run")

    info = {
        "dag_id": context.get("dag", {}).dag_id if context.get("dag") else None,
        "task_id": ti.task_id if ti else None,
        "execution_date": str(context.get("execution_date", "")),
        "run_id": dag_run.run_id if dag_run else None,
        "try_number": ti.try_number if ti else None,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Extract exception info if available
    exception = context.get("exception")
    if exception:
        info["exception_type"] = type(exception).__name__
        info["exception_message"] = str(exception)

    return info


def _get_dataset_from_dag_id(dag_id: str) -> str | None:
    """Extract dataset name from DAG ID (e.g., 'crime_pipeline' -> 'crime')."""
    if not dag_id:
        return None

    # Common patterns: crime_pipeline, crime_dag, ingest_crime
    for suffix in ["_pipeline", "_dag", "_etl"]:
        if dag_id.endswith(suffix):
            return dag_id.replace(suffix, "")

    for prefix in ["ingest_", "process_", "feature_"]:
        if dag_id.startswith(prefix):
            return dag_id.replace(prefix, "")

    return dag_id


# =============================================================================
# Task Callbacks
# =============================================================================


def on_task_failure(context: dict[str, Any]) -> None:
    """
    Callback for task failures.

    Sends an alert with task failure details.
    """
    # config = get_config()
    info = _extract_context_info(context)
    dataset = _get_dataset_from_dag_id(info.get("dag_id", ""))

    logger.error(
        f"Task failed: {info['dag_id']}.{info['task_id']}",
        extra=info,
    )

    # Send alert
    try:
        send_alert(
            title=f"Task Failed: {info['task_id']}",
            message=(
                f"**DAG:** {info['dag_id']}\n"
                f"**Task:** {info['task_id']}\n"
                f"**Execution Date:** {info['execution_date']}\n"
                f"**Try Number:** {info['try_number']}\n"
                f"**Error:** {info.get('exception_message', 'Unknown error')}"
            ),
            severity="critical",
            dataset=dataset,
            dag_id=info.get("dag_id"),
            task_id=info.get("task_id"),
            execution_date=info.get("execution_date"),
        )
    except Exception as e:
        logger.error(f"Failed to send task failure alert: {e}")


def on_task_success(context: dict[str, Any]) -> None:
    """
    Callback for task success.

    Logs success and optionally sends info-level alert.
    """
    info = _extract_context_info(context)

    logger.info(
        f"Task succeeded: {info['dag_id']}.{info['task_id']}",
        extra=info,
    )


def on_task_retry(context: dict[str, Any]) -> None:
    """
    Callback for task retries.

    Sends a warning alert when a task is being retried.
    """
    info = _extract_context_info(context)
    dataset = _get_dataset_from_dag_id(info.get("dag_id", ""))

    logger.warning(
        f"Task retrying: {info['dag_id']}.{info['task_id']} (attempt {info['try_number']})",
        extra=info,
    )

    # Only alert after first retry
    if info.get("try_number", 1) > 1:
        try:
            send_alert(
                title=f"Task Retrying: {info['task_id']}",
                message=(
                    f"**DAG:** {info['dag_id']}\n"
                    f"**Task:** {info['task_id']}\n"
                    f"**Attempt:** {info['try_number']}\n"
                    f"**Error:** {info.get('exception_message', 'Unknown error')}"
                ),
                severity="warning",
                dataset=dataset,
                dag_id=info.get("dag_id"),
                task_id=info.get("task_id"),
                execution_date=info.get("execution_date"),
            )
        except Exception as e:
            logger.error(f"Failed to send task retry alert: {e}")


# =============================================================================
# DAG Callbacks
# =============================================================================


def on_dag_failure(context: dict[str, Any]) -> None:
    """
    Callback for DAG failures.

    Sends a critical alert when a DAG fails completely.
    """
    info = _extract_context_info(context)
    dataset = _get_dataset_from_dag_id(info.get("dag_id", ""))

    logger.error(
        f"DAG failed: {info['dag_id']}",
        extra=info,
    )

    # Send alert
    try:
        send_alert(
            title=f"DAG Failed: {info['dag_id']}",
            message=(
                f"**DAG:** {info['dag_id']}\n"
                f"**Execution Date:** {info['execution_date']}\n"
                f"**Run ID:** {info['run_id']}\n"
                f"The DAG has failed after all retries."
            ),
            severity="critical",
            dataset=dataset,
            dag_id=info.get("dag_id"),
            execution_date=info.get("execution_date"),
        )
    except Exception as e:
        logger.error(f"Failed to send DAG failure alert: {e}")


def on_dag_success(context: dict[str, Any]) -> None:
    """
    Callback for DAG success.

    Logs success for monitoring.
    """
    info = _extract_context_info(context)

    logger.info(
        f"DAG succeeded: {info['dag_id']}",
        extra=info,
    )


# =============================================================================
# Specialized Callbacks
# =============================================================================


def on_validation_failure(context: dict[str, Any]) -> None:
    """
    Specialized callback for validation task failures.

    Includes validation-specific details in the alert.
    """
    info = _extract_context_info(context)
    dataset = _get_dataset_from_dag_id(info.get("dag_id", ""))

    # Try to get validation result from XCom
    ti = context.get("task_instance")
    validation_result = None
    if ti:
        with suppress(Exception):
            validation_result = ti.xcom_pull(task_ids=ti.task_id)

    logger.error(
        f"Validation failed: {info['dag_id']}.{info['task_id']}",
        extra={**info, "validation_result": validation_result},
    )

    message = (
        f"**DAG:** {info['dag_id']}\n"
        f"**Task:** {info['task_id']}\n"
        f"**Execution Date:** {info['execution_date']}\n"
    )

    if validation_result and isinstance(validation_result, dict):
        errors = validation_result.get("errors", [])
        if errors:
            message += "**Errors:**\n"
            for err in errors[:5]:  # Limit to 5 errors
                message += f"  - {err}\n"
            if len(errors) > 5:
                message += f"  ... and {len(errors) - 5} more errors"

    try:
        send_alert(
            title=f"Validation Failed: {dataset or info['task_id']}",
            message=message,
            severity="critical",
            dataset=dataset,
            dag_id=info.get("dag_id"),
            task_id=info.get("task_id"),
            execution_date=info.get("execution_date"),
        )
    except Exception as e:
        logger.error(f"Failed to send validation failure alert: {e}")


def on_drift_detected(context: dict[str, Any]) -> None:
    """
    Specialized callback for drift detection alerts.

    Sends a warning when data drift is detected.
    """
    info = _extract_context_info(context)
    dataset = _get_dataset_from_dag_id(info.get("dag_id", ""))

    # Try to get drift result from XCom
    ti = context.get("task_instance")
    drift_result = None
    if ti:
        with suppress(Exception):
            drift_result = ti.xcom_pull(task_ids=ti.task_id)

    logger.warning(
        f"Drift detected: {info['dag_id']}.{info['task_id']}",
        extra={**info, "drift_result": drift_result},
    )

    message = f"**DAG:** {info['dag_id']}\n" f"**Execution Date:** {info['execution_date']}\n"

    if drift_result and isinstance(drift_result, dict):
        drifted_features = drift_result.get("drifted_features", [])
        if drifted_features:
            message += "**Drifted Features:**\n"
            for feature in drifted_features[:10]:
                message += f"  - {feature}\n"

    severity = "warning"
    if drift_result and drift_result.get("severity") == "critical":
        severity = "critical"

    try:
        send_alert(
            title=f"Data Drift Detected: {dataset or info['dag_id']}",
            message=message,
            severity=severity,
            dataset=dataset,
            dag_id=info.get("dag_id"),
            task_id=info.get("task_id"),
            execution_date=info.get("execution_date"),
        )
    except Exception as e:
        logger.error(f"Failed to send drift detection alert: {e}")


# =============================================================================
# Callback Factories
# =============================================================================


def create_failure_callback(
    dataset: str | None = None,
    severity: str = "critical",
) -> callable:
    """
    Create a customized failure callback.

    Args:
        dataset: Dataset name for the alert
        severity: Alert severity level

    Returns:
        Callback function
    """

    def callback(context: dict[str, Any]) -> None:
        info = _extract_context_info(context)
        ds = dataset or _get_dataset_from_dag_id(info.get("dag_id", ""))

        logger.error(
            f"Task failed: {info['dag_id']}.{info['task_id']}",
            extra=info,
        )

        try:
            send_alert(
                title=f"Task Failed: {info['task_id']}",
                message=(
                    f"**DAG:** {info['dag_id']}\n"
                    f"**Task:** {info['task_id']}\n"
                    f"**Error:** {info.get('exception_message', 'Unknown error')}"
                ),
                severity=severity,
                dataset=ds,
                dag_id=info.get("dag_id"),
                task_id=info.get("task_id"),
                execution_date=info.get("execution_date"),
            )
        except Exception as e:
            logger.error(f"Failed to send failure alert: {e}")

    return callback
