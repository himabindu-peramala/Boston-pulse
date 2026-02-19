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
    - StatisticsGenerator: pandas/numpy-based statistics
    - DriftDetector: PSI calculation, distribution drift
    - AnomalyDetector: Missing values, outliers, coordinate bounds
"""

from src.validation.anomaly_detector import (
    AnomalyDetector,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    detect_anomalies,
)
from src.validation.drift_detector import (
    DriftDetector,
    DriftResult,
    DriftSeverity,
    check_drift,
)
from src.validation.schema_enforcer import (
    SchemaEnforcer,
    ValidationError,
    ValidationResult,
    ValidationStage,
    enforce_validation,
)
from src.validation.schema_registry import SchemaRegistry, create_schema_from_dataframe
from src.validation.statistics_generator import (
    StatisticsGenerator,
    generate_and_save_statistics,
    get_latest_statistics,
)

__version__ = "0.1.0"

__all__ = [
    # Schema Registry
    "SchemaRegistry",
    "create_schema_from_dataframe",
    # Schema Enforcer
    "SchemaEnforcer",
    "ValidationError",
    "ValidationResult",
    "ValidationStage",
    "enforce_validation",
    # Statistics
    "StatisticsGenerator",
    "generate_and_save_statistics",
    "get_latest_statistics",
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "DriftSeverity",
    "check_drift",
    # Anomaly Detection
    "AnomalyDetector",
    "AnomalyResult",
    "AnomalySeverity",
    "AnomalyType",
    "detect_anomalies",
]
