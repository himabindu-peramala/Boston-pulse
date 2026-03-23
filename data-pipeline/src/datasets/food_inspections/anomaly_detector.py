"""
Food Inspections Dataset Anomaly Detector
"""

from src.validation.anomaly_detector import AnomalyDetector, AnomalyResult, AnomalyType


class FoodInspectionsAnomalyDetector(AnomalyDetector):
    """Anomaly detector tailored for Food Inspections dataset."""

    def detect_anomalies(self, df, dataset: str, **kwargs) -> AnomalyResult:
        result = super().detect_anomalies(df, dataset, **kwargs)

        # Filter expected anomalies to prevent Slack alert noise
        filtered = []
        expected_missing = {
            "dbaname",
            "licensecat",
            "violation",
            "violdesc",
            "viol_level",
            "comments",
            "descript",
            "zip",
            "lat",
            "long",
            "licenseno",
        }

        for anomaly in result.anomalies:
            if anomaly.type == AnomalyType.MISSING_VALUES and anomaly.feature in expected_missing:
                # These fields are naturally sparse/optional
                continue

            filtered.append(anomaly)

        result.anomalies = filtered
        return result
