"""
Service 311 Dataset Anomaly Detector
"""

from src.validation.anomaly_detector import AnomalyDetector, AnomalyResult, AnomalyType


class Service311AnomalyDetector(AnomalyDetector):
    """Anomaly detector tailored for 311 Service Requests dataset."""

    def detect_anomalies(self, df, dataset: str, **kwargs) -> AnomalyResult:
        result = super().detect_anomalies(df, dataset, **kwargs)

        # Filter expected anomalies to prevent Slack alert noise
        filtered = []
        expected_missing = {"close_date", "lat", "long"}

        for anomaly in result.anomalies:
            if anomaly.type == AnomalyType.MISSING_VALUES and anomaly.feature in expected_missing:
                # These fields are naturally sparse/optional
                continue

            filtered.append(anomaly)

        result.anomalies = filtered
        return result
