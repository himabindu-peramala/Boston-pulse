"""
Boston Pulse - Validation System

Data validation components for ensuring data quality:
- Schema validation and enforcement
- Drift detection
- Anomaly detection
- Statistics generation

Components:
    - SchemaRegistry: GCS-backed schema storage with versioning
    - SchemaEnforcer: Three-stage validation (raw/processed/features)
    - StatisticsGenerator: TFDV-based statistics
    - DriftDetector: PSI calculation, distribution drift
    - AnomalyDetector: Missing values, outliers, coordinate bounds
"""

# Components will be implemented in Phase 2
# from src.validation.schema_registry import SchemaRegistry
# from src.validation.schema_enforcer import SchemaEnforcer
# from src.validation.statistics_generator import StatisticsGenerator
# from src.validation.drift_detector import DriftDetector
# from src.validation.anomaly_detector import AnomalyDetector

# __all__ = [
#     "SchemaRegistry",
#     "SchemaEnforcer",
#     "StatisticsGenerator",
#     "DriftDetector",
#     "AnomalyDetector",
# ]
