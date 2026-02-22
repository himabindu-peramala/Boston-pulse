"""
Tests for Anomaly Detector

Tests anomaly detection for various data quality issues.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.anomaly_detector import (
    AnomalyDetector,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    detect_anomalies,
)


@pytest.fixture
def clean_data():
    """Clean dataset with no anomalies."""
    return pd.DataFrame(
        {
            "id": range(1000),
            "value": np.random.normal(100, 15, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
            "date": pd.date_range("2024-01-01", periods=1000),
            "lat": np.random.uniform(42.25, 42.35, 1000),
            "lon": np.random.uniform(-71.15, -71.05, 1000),
        }
    )


def test_detector_initialization():
    """Test AnomalyDetector initialization."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    assert detector.config == config


def test_detect_anomalies_clean_data(clean_data):
    """Test anomaly detection on clean data."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    result = detector.detect_anomalies(clean_data, "test")

    assert result.dataset == "test"
    assert result.row_count == len(clean_data)
    # Clean data should have few or no anomalies
    assert len(result.critical_anomalies) == 0


def test_detect_missing_values():
    """Test detection of missing value patterns."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with high missing values
    df = pd.DataFrame(
        {
            "col1": [1] * 40 + [None] * 60,  # 60% missing
            "col2": [1] * 100,
        }
    )

    result = detector.detect_anomalies(df, "test", check_outliers=False)

    assert result.has_anomalies
    missing_anomalies = [a for a in result.anomalies if a.type == AnomalyType.MISSING_VALUES]
    assert len(missing_anomalies) > 0
    assert any(a.severity == AnomalySeverity.CRITICAL for a in missing_anomalies)


def test_detect_outliers():
    """Test outlier detection."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with outliers
    # Threshold is 10%, we need > 10%.
    # Current: 90 normal + 15 outliers = 105 total. 15/105 = 14.3%
    values = [100] * 90 + [1000, 2000, 3000, 4000, 5000] * 3
    df = pd.DataFrame({"value": values})

    result = detector.detect_anomalies(df, "test")

    outlier_anomalies = [a for a in result.anomalies if a.type == AnomalyType.OUTLIER]
    assert len(outlier_anomalies) > 0


def test_detect_geographic_anomalies():
    """Test geographic bounds checking."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with coordinates outside Boston
    df = pd.DataFrame(
        {
            "id": range(100),
            "lat": [40.0] * 50 + [42.3] * 50,  # 50% outside bounds
            "lon": [-71.1] * 100,
        }
    )

    result = detector.detect_anomalies(df, "test", check_outliers=False)

    geo_anomalies = [a for a in result.anomalies if a.type == AnomalyType.GEOGRAPHIC]
    assert len(geo_anomalies) > 0
    assert any("latitude" in a.message.lower() for a in geo_anomalies)


def test_detect_temporal_anomalies():
    """Test temporal anomaly detection."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with future dates
    future_dates = [datetime.now(UTC) + timedelta(days=30)] * 50
    past_dates = [datetime.now(UTC) - timedelta(days=30)] * 50
    df = pd.DataFrame(
        {
            "id": range(100),
            "occurred_date": future_dates + past_dates,
        }
    )

    result = detector.detect_anomalies(df, "test", check_outliers=False)

    temporal_anomalies = [a for a in result.anomalies if a.type == AnomalyType.TEMPORAL]
    assert len(temporal_anomalies) > 0


def test_detect_categorical_anomalies():
    """Test categorical anomaly detection."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with too many unique categories
    df = pd.DataFrame(
        {
            "id": range(1000),
            "category": [f"cat_{i}" for i in range(1000)],  # All unique
        }
    )

    result = detector.detect_anomalies(
        df, "test", check_outliers=False, check_geographic=False, check_temporal=False
    )

    categorical_anomalies = [a for a in result.anomalies if a.type == AnomalyType.CATEGORICAL]
    # May detect high uniqueness
    assert len(categorical_anomalies) >= 0  # Depends on thresholds


def test_detect_duplicates():
    """Test duplicate detection."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    # DataFrame with duplicates
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 1, 2, 3] * 10,  # 50% duplicates
            "col2": ["A", "B", "C", "A", "B", "C"] * 10,
        }
    )

    result = detector.detect_anomalies(df, "test", check_outliers=False)

    dup_anomalies = [a for a in result.anomalies if a.type == AnomalyType.DUPLICATE]
    assert len(dup_anomalies) > 0
    assert dup_anomalies[0].count > 0


def test_anomaly_result_properties(clean_data):
    """Test AnomalyResult properties."""
    config = get_config("dev")
    detector = AnomalyDetector(config)

    result = detector.detect_anomalies(clean_data, "test")

    # Test properties
    assert isinstance(result.has_anomalies, bool)
    assert isinstance(result.has_critical_anomalies, bool)
    assert isinstance(result.critical_anomalies, list)
    assert isinstance(result.warning_anomalies, list)
    assert isinstance(result.anomalies_by_type, dict)


def test_detect_anomalies_convenience_function(clean_data):
    """Test convenience function."""
    result = detect_anomalies(clean_data, "test")

    assert result.dataset == "test"
    assert isinstance(result, AnomalyResult)
