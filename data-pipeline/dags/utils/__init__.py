from .alerting import (
    alert_anomaly_detected,
    alert_drift_detected,
    alert_fairness_violation,
    alert_pipeline_complete,
    alert_preprocessing_complete,
    alert_validation_failure,
)
from .callbacks import on_dag_failure, on_dag_success, on_task_failure
from .gcs_io import read_data, write_data
from .watermark import get_effective_watermark, set_watermark

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
    "alert_preprocessing_complete",
    "on_dag_failure",
    "on_dag_success",
    "on_task_failure",
]


def trigger_chatbot_ingest(**context) -> dict:
    """Trigger chatbot backend to re-ingest latest data into ChromaDB."""
    import os

    import requests

    backend_url = os.getenv(
        "CHATBOT_BACKEND_URL", "https://boston-pulse-chatbot-384523870431.us-central1.run.app"
    )

    try:
        response = requests.post(f"{backend_url}/api/ingest", timeout=10)
        response.raise_for_status()
        return {"status": "triggered", "response": response.json()}
    except Exception as e:
        # Don't fail the pipeline if chatbot ingest fails
        return {"status": "failed", "error": str(e)}
