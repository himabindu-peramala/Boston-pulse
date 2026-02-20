"""
Boston Pulse - DAG Utilities

Common utilities for Airflow DAGs including:
- GCS I/O operations
- Watermark management
- Task callbacks
- DAG alerting

Usage:
    from dags.utils import (
        read_data, write_data, get_latest_data,
        get_watermark, set_watermark,
        on_task_failure, on_dag_failure,
        alert_validation_failure, alert_drift_detected,
    )
"""

from dags.utils.alerting import (
    DAGAlertManager,
    alert_anomaly_detected,
    alert_drift_detected,
    alert_fairness_violation,
    alert_ingestion_complete,
    alert_pipeline_complete,
    alert_preprocessing_complete,
    alert_validation_failure,
)
from dags.utils.callbacks import (
    create_failure_callback,
    on_dag_failure,
    on_dag_success,
    on_drift_detected,
    on_task_failure,
    on_task_retry,
    on_task_success,
    on_validation_failure,
)
from dags.utils.gcs_io import (
    GCSDataIO,
    get_latest_data,
    read_data,
    write_data,
)
from dags.utils.watermark import (
    WatermarkManager,
    get_effective_watermark,
    get_watermark,
    set_watermark,
)

__all__ = [
    # GCS I/O
    "GCSDataIO",
    "read_data",
    "write_data",
    "get_latest_data",
    # Watermark
    "WatermarkManager",
    "get_watermark",
    "set_watermark",
    "get_effective_watermark",
    # Callbacks
    "on_task_failure",
    "on_task_success",
    "on_task_retry",
    "on_dag_failure",
    "on_dag_success",
    "on_validation_failure",
    "on_drift_detected",
    "create_failure_callback",
    # Alerting
    "DAGAlertManager",
    "alert_ingestion_complete",
    "alert_preprocessing_complete",
    "alert_validation_failure",
    "alert_drift_detected",
    "alert_anomaly_detected",
    "alert_fairness_violation",
    "alert_pipeline_complete",
]
