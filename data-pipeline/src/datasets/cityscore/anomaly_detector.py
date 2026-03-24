"""
CityScore Dataset Anomaly Detector
"""

from src.validation.anomaly_detector import AnomalyDetector, AnomalyResult


class CityScoreAnomalyDetector(AnomalyDetector):
    """Anomaly detector tailored for CityScore dataset."""

    def detect_anomalies(self, df, dataset: str, **kwargs) -> AnomalyResult:
        result = super().detect_anomalies(df, dataset, **kwargs)
        return result
