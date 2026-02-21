# NOTE/TODO: Right now the `data-pipeline/src/validation/anomaly_detector.py` is a singleton.
# We need to add a registry to allow for multiple anomaly detectors for different datasets, this
# will allow customization and flexibility for the pipeline. Below is a template for the registry.

# from src.datasets.crime.anomaly_detector import CrimeAnomalyDetector
# from src.datasets.mbta.anomaly_detector import MBTAAnomalyDetector

# # Simple dictionary lookup
# _REGISTRY = {
#     "crime": CrimeAnomalyDetector,
#     "mbta":  MBTAAnomalyDetector,
#     "311":   Service311AnomalyDetector,
#     # new datasets just get added here
# }

# def get_anomaly_detector(dataset: str) -> AnomalyDetector:
#     """Get the right detector for this dataset."""
#     detector_class = _REGISTRY.get(dataset, AnomalyDetector)  # fallback to base
#     return detector_class()

# Example usage in the dataset's DAG:
# def detect_anomalies(**context) -> dict:
#     from src.validation.anomaly_registry import get_anomaly_detector

#     df = read_data(DATASET, "raw", execution_date)

#     # Registry figures out which detector to use
#     # For crime_dag.py, DATASET = "crime", so gets CrimeAnomalyDetector
#     detector = get_anomaly_detector(DATASET)
#     result = detector.detect_anomalies(df, DATASET)

#     return {"anomaly_count": len(result.anomalies)}anomalies(df, dataset)
#     return anomalies
