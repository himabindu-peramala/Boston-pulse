"""
Boston Pulse - DAG Utilities

This package root uses lazy loading to prevent heavy submodule loads
(like google-cloud-storage and pandas) during Airflow DAG parsing.

All utilities can still be imported from this root for backward compatibility:
    from dags.utils import write_data, on_dag_failure, ...

For new code, direct submodule imports are preferred for clarity:
    from dags.utils.gcs_io import write_data
"""

from __future__ import annotations

import importlib
from typing import Any

# Mapping of attribute names to their submodules
_LAZY_IMPORTS = {
    # GCS I/O
    "read_data": "dags.utils.gcs_io",
    "write_data": "dags.utils.gcs_io",
    # Watermark
    "get_watermark": "dags.utils.watermark",
    "set_watermark": "dags.utils.watermark",
    "get_effective_watermark": "dags.utils.watermark",
    # Callbacks
    "on_dag_failure": "dags.utils.callbacks",
    "on_dag_success": "dags.utils.callbacks",
    "on_task_failure": "dags.utils.callbacks",
    "on_task_success": "dags.utils.callbacks",
    # Alerting
    "alert_ingestion_complete": "dags.utils.alerting",
    "alert_preprocessing_complete": "dags.utils.alerting",
    "alert_validation_failure": "dags.utils.alerting",
    "alert_drift_detected": "dags.utils.alerting",
    "alert_fairness_violation": "dags.utils.alerting",
    "alert_anomaly_detected": "dags.utils.alerting",
    "alert_pipeline_complete": "dags.utils.alerting",
    # Lineage
    "record_pipeline_lineage": "dags.utils.lineage_utils",
}


def __getattr__(name: str) -> Any:
    """Lazy loader for module attributes."""
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    """Ensure dir() shows the lazy attributes for better IDE support."""
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


__all__ = list(_LAZY_IMPORTS.keys())
