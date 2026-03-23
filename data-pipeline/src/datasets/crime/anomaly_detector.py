"""
Crime Dataset Anomaly Detector
"""

from src.validation.anomaly_detector import AnomalyDetector, AnomalyResult, AnomalyType


class CrimeAnomalyDetector(AnomalyDetector):
    """Anomaly detector tailored for Crime dataset."""

    def detect_anomalies(self, df, dataset: str, **kwargs) -> AnomalyResult:
        result = super().detect_anomalies(df, dataset, **kwargs)

        # Filter expected anomalies to prevent Slack alert noise
        filtered = []
        expected_missing = {
            "offense_description",
            "reporting_area",
            "shooting",
            "ucr_part",
            "street",
            "lat",
            "long",
            "offense_code",
        }

        for anomaly in result.anomalies:
            if anomaly.type == AnomalyType.MISSING_VALUES and anomaly.feature in expected_missing:
                # These fields are naturally sparse/optional
                continue

            filtered.append(anomaly)

        result.anomalies = filtered
        return result
