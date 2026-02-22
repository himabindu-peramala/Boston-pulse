"""
Tests for Drift Detector

Tests PSI calculation and distribution drift detection.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.drift_detector import DriftDetector, DriftSeverity, check_drift

# Mock evidently modules before they are imported in the code
sys.modules["evidently"] = MagicMock()
sys.modules["evidently.report"] = MagicMock()
sys.modules["evidently.metric_preset"] = MagicMock()


@pytest.fixture(autouse=True)
def mock_storage_client():
    """Mock storage.Client for all tests in this module."""
    with patch("src.validation.statistics_generator.storage.Client"):
        yield


@pytest.fixture
def reference_data():
    """Reference dataset for drift comparison."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "income": np.random.normal(50000, 15000, 1000),
            "age": np.random.normal(35, 10, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def mock_gcs():
    """Mock GCS client for unit tests."""
    with patch("src.validation.statistics_generator.storage.Client") as mock_client:
        mock_client.return_value = MagicMock()
        yield mock_client


@pytest.fixture
def current_data_no_drift():
    """Current dataset with no drift."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "income": np.random.normal(50000, 15000, 1000),
            "age": np.random.normal(35, 10, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def current_data_with_drift():
    """Current dataset with significant drift."""
    np.random.seed(44)
    return pd.DataFrame(
        {
            "income": np.random.normal(70000, 20000, 1000),  # Mean shifted
            "age": np.random.normal(45, 15, 1000),  # Mean and std shifted
            "category": np.random.choice(
                ["A", "B", "C"], 1000, p=[0.2, 0.3, 0.5]
            ),  # Distribution changed
            "score": np.random.uniform(0, 100, 1000),
        }
    )


def test_detector_initialization(mock_gcs):
    """Test DriftDetector initialization."""
    config = get_config("dev")
    detector = DriftDetector(config)

    assert detector.config == config
    assert detector.psi_warning_threshold == config.drift.psi.warning
    assert detector.psi_critical_threshold == config.drift.psi.critical


def test_detect_drift_no_drift(mock_gcs, reference_data, current_data_no_drift):
    """Test drift detection with no significant drift."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_no_drift, reference_data, "test")

    assert result.dataset == "test"
    assert result.features_analyzed == 4
    # May have some minor drift warnings, but no critical
    assert not result.has_critical_drift


def test_detect_drift_with_drift(mock_gcs, reference_data, current_data_with_drift):
    """Test drift detection with significant drift."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_with_drift, reference_data, "test")

    assert result.features_analyzed == 4
    assert len(result.features_with_drift) > 0
    # Should detect drift in income, age, and category
    assert any(f.feature_name == "income" for f in result.features_with_drift)


def test_detect_numerical_drift(mock_gcs):
    """Test numerical drift detection."""
    config = get_config("dev")
    detector = DriftDetector(config)

    # Create reference and current with known drift
    reference = pd.DataFrame({"value": np.random.normal(0, 1, 1000)})
    current = pd.DataFrame({"value": np.random.normal(2, 1, 1000)})  # Shifted mean

    result = detector.detect_drift(current, reference, "test")

    assert len(result.features_with_drift) > 0
    drift_feature = result.features_with_drift[0]
    assert drift_feature.drift_type == "numerical"
    assert drift_feature.psi > 0


def test_detect_categorical_drift(mock_gcs):
    """Test categorical drift detection."""
    config = get_config("dev")
    detector = DriftDetector(config)

    # Create reference and current with different distributions
    reference = pd.DataFrame(
        {"category": np.random.choice(["A", "B", "C"], 1000, p=[0.7, 0.2, 0.1])}
    )
    current = pd.DataFrame({"category": np.random.choice(["A", "B", "C"], 1000, p=[0.3, 0.4, 0.3])})

    result = detector.detect_drift(current, reference, "test")

    assert len(result.features_with_drift) > 0
    drift_feature = result.features_with_drift[0]
    assert drift_feature.drift_type == "categorical"
    assert drift_feature.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)


def test_psi_calculation_numerical(mock_gcs):
    """Test PSI calculation for numerical features."""
    config = get_config("dev")
    detector = DriftDetector(config)

    # Identical distributions should have PSI ≈ 0
    reference = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(0, 1, 1000))

    psi = detector._calculate_psi_numerical(current, reference)

    assert psi >= 0  # PSI is always non-negative
    assert psi < 0.1  # Should be low for similar distributions


def test_drift_result_properties(mock_gcs, reference_data, current_data_with_drift):
    """Test DriftResult properties."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_with_drift, reference_data, "test")

    # Test properties
    assert isinstance(result.has_warning_drift, bool)
    assert isinstance(result.has_critical_drift, bool)
    assert isinstance(result.warning_features, list)
    assert isinstance(result.critical_features, list)


def test_check_drift_convenience_function(mock_gcs, reference_data, current_data_no_drift):
    """Test convenience function."""
    result = check_drift(current_data_no_drift, reference_data, "test")

    assert result.dataset == "test"
    assert isinstance(result, type(result))  # DriftResult type


def test_detect_drift_with_evidently(reference_data, current_data_with_drift):
    """Test drift detection using Evidently (mocked)."""
    # Get mock from sys.modules
    mock_report_class = sys.modules["evidently.report"].Report

    detector = DriftDetector()

    mock_report = MagicMock()
    mock_report_class.return_value = mock_report
    mock_report.as_dict.return_value = {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "drift_by_columns": {
                        "income": {
                            "drift_detected": True,
                            "drift_score": 0.001,
                            "stattest_name": "ks",
                        }
                    }
                },
            }
        ]
    }

    result = detector.detect_drift_with_evidently(current_data_with_drift, reference_data, "test")

    assert result.dataset == "test"
    assert len(result.features_with_drift) == 1
    assert result.features_with_drift[0].feature_name == "income"
    assert result.features_with_drift[0].severity == DriftSeverity.CRITICAL


def test_determine_severity_from_evidently():
    """Test severity determination from Evidently scores."""
    detector = DriftDetector()

    # P-value based
    assert detector._determine_severity_from_evidently(0.001, "ks") == DriftSeverity.CRITICAL
    assert detector._determine_severity_from_evidently(0.03, "chi2") == DriftSeverity.WARNING
    assert detector._determine_severity_from_evidently(0.1, "ks") == DriftSeverity.NONE

    # Distance based
    detector.psi_warning_threshold = 0.1
    detector.psi_critical_threshold = 0.2
    assert detector._determine_severity_from_evidently(0.3, "other") == DriftSeverity.CRITICAL
    assert detector._determine_severity_from_evidently(0.15, "other") == DriftSeverity.WARNING
    assert detector._determine_severity_from_evidently(0.05, "other") == DriftSeverity.NONE


def test_calculate_psi_from_stats():
    """Test PSI calculation from pre-computed stats."""
    detector = DriftDetector()

    # Numerical
    curr = MagicMock(mean=110, std=10)
    ref = MagicMock(mean=100, std=10)
    psi = detector._calculate_psi_from_stats(curr, ref)
    assert psi > 0

    # Categorical
    curr_cat = MagicMock(mean=None, num_unique=15)
    ref_cat = MagicMock(mean=None, num_unique=10)
    psi_cat = detector._calculate_psi_from_stats(curr_cat, ref_cat)
    assert psi_cat == 0.5


def test_detect_drift_error_handling():
    """Test that drift detection handles per-column errors gracefully."""
    detector = DriftDetector()
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col2": [1, 2]})  # No common columns

    # Should not raise exception
    result = detector.detect_drift(df1, df2, "test")
    assert result.features_analyzed == 0
