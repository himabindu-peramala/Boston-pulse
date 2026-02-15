"""
Boston Pulse - DAG Utilities

Shared utilities for Airflow DAGs:
- GCS I/O operations
- Watermark management
- Task callbacks
- Alert integration
- DVC versioning

Components:
    - gcs_io: GCS read/write utilities
    - watermark: Watermark management for incremental ingestion
    - callbacks: Task success/failure callbacks
    - alerting: DAG alerting integration
    - dvc_utils: DVC versioning utilities
"""

# Components will be implemented in Phase 3
# from dags.utils.gcs_io import read_from_gcs, write_to_gcs
# from dags.utils.watermark import get_watermark, set_watermark
# from dags.utils.callbacks import on_success, on_failure
# from dags.utils.alerting import send_dag_alert
# from dags.utils.dvc_utils import commit_to_dvc

# __all__ = [
#     "read_from_gcs",
#     "write_to_gcs",
#     "get_watermark",
#     "set_watermark",
#     "on_success",
#     "on_failure",
#     "send_dag_alert",
#     "commit_to_dvc",
# ]
