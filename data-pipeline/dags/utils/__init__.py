"""
Boston Pulse - DAG Utilities

Common utilities for Airflow DAGs including:
- GCS I/O operations
- Watermark management
- Task callbacks
- DAG alerting
- Lineage tracking

Usage:
    from dags.utils import (
        read_data, write_data, get_latest_data,
        get_watermark, set_watermark,
        on_task_failure, on_dag_failure,
        alert_validation_failure, alert_drift_detected,
        record_pipeline_lineage, get_lineage_for_date,
    )
"""

from dags.utils.alerting import (
    DAGAlertManager,
    alert_anomaly_detected,
    alert_validation_failure,
    alert_drift_detected,
    alert_fairness_violation,
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
from dags.utils.lineage_utils import (
    compare_runs,
    create_record_lineage_task,
    find_runs_with_schema,
    get_dataset_lineage_history,
    get_lineage_for_date,
    get_restore_commands,
    print_lineage_diff,
    print_lineage_summary,
    print_restore_commands,
    record_pipeline_lineage,
)
from dags.utils.watermark import (
    WatermarkManager,
    get_effective_watermark,
    get_watermark,
    set_watermark,
)
from .callbacks import on_dag_failure, on_dag_success, on_task_failure

__all__ = [
    "get_effective_watermark",
    "set_watermark",
    "read_data",
    "write_data",
    "alert_anomaly_detected",
    "alert_validation_failure",
    "alert_drift_detected",
    "alert_fairness_violation",
    "alert_pipeline_complete",
    # Lineage
    "record_pipeline_lineage",
    "create_record_lineage_task",
    "get_lineage_for_date",
    "get_dataset_lineage_history",
    "compare_runs",
    "find_runs_with_schema",
    "get_restore_commands",
    "print_restore_commands",
    "print_lineage_summary",
    "print_lineage_diff",
]
