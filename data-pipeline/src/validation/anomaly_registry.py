"""
Boston Pulse - Anomaly Registry

Provides the correct AnomalyDetector instance for a given dataset,
falling back to the base AnomalyDetector if a specific one does not exist.
"""

from src.datasets.cityscore.anomaly_detector import CityScoreAnomalyDetector
from src.datasets.crime.anomaly_detector import CrimeAnomalyDetector
from src.datasets.food_inspections.anomaly_detector import FoodInspectionsAnomalyDetector
from src.datasets.service_311.anomaly_detector import Service311AnomalyDetector
from src.datasets.street_sweeping.anomaly_detector import StreetSweepingAnomalyDetector
from src.validation.anomaly_detector import AnomalyDetector

_REGISTRY = {
    "crime": CrimeAnomalyDetector,
    "service_311": Service311AnomalyDetector,
    "food_inspections": FoodInspectionsAnomalyDetector,
    "cityscore": CityScoreAnomalyDetector,
    "street_sweeping": StreetSweepingAnomalyDetector,
}


def get_anomaly_detector(dataset: str) -> AnomalyDetector:
    """Get the appropriate anomaly detector for this dataset."""
    detector_class = _REGISTRY.get(dataset, AnomalyDetector)
    return detector_class()
