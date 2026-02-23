"""
Boston Pulse - DAG Alerting Utilities

DAG-specific alerting utilities that wrap the core alerting system.
Provides convenient functions for sending alerts from DAG tasks.

Usage:
    from dags.utils.alerting import (
        alert_ingestion_complete,
        alert_validation_failure,
        alert_drift_detected,
    )

    alert_ingestion_complete("crime", 1000, "2024-01-15")
"""

from __future__ import annotations

import logging
from typing import Any

from src.alerting import AlertManager, send_alert
from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class DAGAlertManager:
    """
    DAG-specific alert manager.

    Provides convenient methods for common DAG alerting scenarios.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize DAG alert manager."""
        self.config = config or get_config()
        self.alert_manager = AlertManager(config)

    def alert_ingestion_complete(
        self,
        dataset: str,
        rows_fetched: int,
        execution_date: str,
        duration_seconds: float | None = None,
        dag_id: str | None = None,
    ) -> None:
        """
        Send alert for successful ingestion.

        Only sends alert for large ingestions or if there are concerns.
        """
        message = (
            f"Ingestion completed for **{dataset}**\n"
            f"- Rows fetched: {rows_fetched:,}\n"
            f"- Execution date: {execution_date}"
        )

        if duration_seconds:
            message += f"\n- Duration: {duration_seconds:.1f}s"

        # Only alert if rows are significant or zero (potential issue)
        severity = "info"
        if rows_fetched == 0:
            severity = "warning"
            message = f"{message}\n- No data fetched - possible data source issue"

        send_alert(
            title=f"Ingestion Complete: {dataset}",
            message=message,
            severity=severity,
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
            metadata={"rows_fetched": rows_fetched, "duration_seconds": duration_seconds},
        )

    def alert_preprocessing_complete(
        self,
        dataset: str,
        rows_input: int,
        rows_output: int,
        rows_dropped: int,
        execution_date: str,
        dag_id: str | None = None,
    ) -> None:
        """Send alert for preprocessing completion."""
        drop_rate = (rows_dropped / rows_input * 100) if rows_input > 0 else 0

        severity = "info"
        if drop_rate > 20:
            severity = "warning"
        if drop_rate > 50:
            severity = "critical"

        message = (
            f"Preprocessing completed for **{dataset}**\n"
            f"- Rows input: {rows_input:,}\n"
            f"- Rows output: {rows_output:,}\n"
            f"- Rows dropped: {rows_dropped:,} ({drop_rate:.1f}%)"
        )

        if severity != "info":
            message += "\n- High drop rate detected"

        send_alert(
            title=f"Preprocessing Complete: {dataset}",
            message=message,
            severity=severity,
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
        )

    def alert_validation_failure(
        self,
        dataset: str,
        stage: str,
        errors: list[str],
        execution_date: str,
        dag_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Send alert for validation failures."""
        error_summary = "\n".join([f"  - {e}" for e in errors[:10]])
        if len(errors) > 10:
            error_summary += f"\n  - ... and {len(errors) - 10} more errors"

        message = (
            f"Validation failed for **{dataset}** at stage **{stage}**\n"
            f"- Errors ({len(errors)} total):\n{error_summary}"
        )

        send_alert(
            title=f"Validation Failed: {dataset} ({stage})",
            message=message,
            severity="critical",
            dataset=dataset,
            dag_id=dag_id,
            task_id=task_id,
            execution_date=execution_date,
            metadata={"stage": stage, "error_count": len(errors)},
        )

    def alert_drift_detected(
        self,
        dataset: str,
        drifted_features: list[str],
        psi_scores: dict[str, float],
        severity: str,
        execution_date: str,
        dag_id: str | None = None,
    ) -> None:
        """Send alert for drift detection."""
        feature_details = "\n".join(
            [f"  - {f}: PSI={psi_scores.get(f, 'N/A'):.3f}" for f in drifted_features[:10]]
        )
        if len(drifted_features) > 10:
            feature_details += f"\n  - ... and {len(drifted_features) - 10} more"

        message = (
            f"Data drift detected for **{dataset}**\n"
            f"- Drifted features ({len(drifted_features)}):\n{feature_details}"
        )

        send_alert(
            title=f"Drift Detected: {dataset}",
            message=message,
            severity=severity,
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
            metadata={"drifted_features": drifted_features, "psi_scores": psi_scores},
        )

    def alert_anomaly_detected(
        self,
        dataset: str,
        anomaly_type: str,
        details: str,
        severity: str,
        execution_date: str,
        dag_id: str | None = None,
    ) -> None:
        """Send alert for anomaly detection."""
        message = (
            f"Anomaly detected for **{dataset}**\n- Type: {anomaly_type}\n- Details: {details}"
        )

        send_alert(
            title=f"Anomaly Detected: {dataset}",
            message=message,
            severity=severity,
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
            metadata={"anomaly_type": anomaly_type},
        )

    def alert_fairness_violation(
        self,
        dataset: str,
        violations: list[dict[str, Any]],
        execution_date: str,
        dag_id: str | None = None,
    ) -> None:
        """Send alert for fairness violations."""

        violation_details = "\n".join(
            [
                f"  - [{v.get('severity', '?').upper()}] {v.get('message', 'No message')}"
                for v in violations[:5]
            ]
        )

        if len(violations) > 5:
            violation_details += f"\n  - ... and {len(violations) - 5} more"

        message = (
            f"Fairness violation detected for **{dataset}**\n"
            f"- Violations ({len(violations)} total):\n{violation_details}"
        )

        send_alert(
            title=f"Fairness Violation: {dataset}",
            message=message,
            severity="critical",
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
            metadata={"violation_count": len(violations)},
        )

    def alert_pipeline_complete(
        self,
        dataset: str,
        execution_date: str,
        duration_seconds: float,
        stats: dict[str, Any],
        dag_id: str | None = None,
    ) -> None:
        """Send alert for successful pipeline completion."""

        # Helper to format numeric stats with thousands separator
        def fmt_stat(val: Any) -> str:
            if isinstance(val, (int, float)):
                return f"{val:,}"
            return str(val)

        message = (
            f"Pipeline completed for **{dataset}**\n"
            f"- Execution date: {execution_date}\n"
            f"- Duration: {duration_seconds:.1f}s\n"
            f"- Rows ingested: {fmt_stat(stats.get('rows_ingested', 'N/A'))}\n"
            f"- Rows processed: {fmt_stat(stats.get('rows_processed', 'N/A'))}\n"
            f"- Features generated: {fmt_stat(stats.get('features_generated', 'N/A'))}"
        )

        send_alert(
            title=f"Pipeline Complete: {dataset}",
            message=message,
            severity="info",
            dataset=dataset,
            dag_id=dag_id,
            execution_date=execution_date,
            metadata=stats,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


_dag_alert_manager: DAGAlertManager | None = None


def _get_dag_alert_manager() -> DAGAlertManager:
    """Get or create DAG alert manager singleton."""
    global _dag_alert_manager
    if _dag_alert_manager is None:
        _dag_alert_manager = DAGAlertManager()
    return _dag_alert_manager


def alert_ingestion_complete(
    dataset: str,
    rows_fetched: int,
    execution_date: str,
    duration_seconds: float | None = None,
    dag_id: str | None = None,
) -> None:
    """Convenience function for ingestion alerts."""
    _get_dag_alert_manager().alert_ingestion_complete(
        dataset, rows_fetched, execution_date, duration_seconds, dag_id
    )


def alert_preprocessing_complete(
    dataset: str,
    rows_input: int,
    rows_output: int,
    rows_dropped: int,
    execution_date: str,
    dag_id: str | None = None,
) -> None:
    """Convenience function for preprocessing alerts."""
    _get_dag_alert_manager().alert_preprocessing_complete(
        dataset, rows_input, rows_output, rows_dropped, execution_date, dag_id
    )


def alert_validation_failure(
    dataset: str,
    stage: str,
    errors: list[str],
    execution_date: str,
    dag_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """Convenience function for validation failure alerts."""
    _get_dag_alert_manager().alert_validation_failure(
        dataset, stage, errors, execution_date, dag_id, task_id
    )


def alert_drift_detected(
    dataset: str,
    drifted_features: list[str],
    psi_scores: dict[str, float],
    severity: str,
    execution_date: str,
    dag_id: str | None = None,
) -> None:
    """Convenience function for drift detection alerts."""
    _get_dag_alert_manager().alert_drift_detected(
        dataset, drifted_features, psi_scores, severity, execution_date, dag_id
    )


def alert_anomaly_detected(
    dataset: str,
    anomaly_type: str,
    details: str,
    severity: str,
    execution_date: str,
    dag_id: str | None = None,
) -> None:
    """Convenience function for anomaly detection alerts."""
    _get_dag_alert_manager().alert_anomaly_detected(
        dataset, anomaly_type, details, severity, execution_date, dag_id
    )


def alert_fairness_violation(
    dataset: str,
    violations: list[dict[str, Any]],
    execution_date: str,
    dag_id: str | None = None,
) -> None:
    """Convenience function for fairness violation alerts."""
    _get_dag_alert_manager().alert_fairness_violation(dataset, violations, execution_date, dag_id)


def alert_pipeline_complete(
    dataset: str,
    execution_date: str,
    duration_seconds: float,
    stats: dict[str, Any],
    dag_id: str | None = None,
) -> None:
    """Convenience function for pipeline completion alerts."""
    _get_dag_alert_manager().alert_pipeline_complete(
        dataset, execution_date, duration_seconds, stats, dag_id
    )
