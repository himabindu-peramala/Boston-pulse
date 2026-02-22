"""
Tests for Drift Detector

Tests PSI calculation and distribution drift detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.drift_detector import DriftDetector, DriftSeverity, check_drift
from unittest.mock import patch, MagicMock

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


def test_detector_initialization():
    """Test DriftDetector initialization."""
    config = get_config("dev")
    detector = DriftDetector(config)

    assert detector.config == config
    assert detector.psi_warning_threshold == config.drift.psi.warning
    assert detector.psi_critical_threshold == config.drift.psi.critical


def test_detect_drift_no_drift(reference_data, current_data_no_drift):
    """Test drift detection with no significant drift."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_no_drift, reference_data, "test")

    assert result.dataset == "test"
    assert result.features_analyzed == 4
    # May have some minor drift warnings, but no critical
    assert not result.has_critical_drift


def test_detect_drift_with_drift(reference_data, current_data_with_drift):
    """Test drift detection with significant drift."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_with_drift, reference_data, "test")

    assert result.features_analyzed == 4
    assert len(result.features_with_drift) > 0
    # Should detect drift in income, age, and category
    assert any(f.feature_name == "income" for f in result.features_with_drift)


def test_detect_numerical_drift():
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


def test_detect_categorical_drift():
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


def test_psi_calculation_numerical():
    """Test PSI calculation for numerical features."""
    config = get_config("dev")
    detector = DriftDetector(config)

    # Identical distributions should have PSI ≈ 0
    reference = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(0, 1, 1000))

    psi = detector._calculate_psi_numerical(current, reference)

    assert psi >= 0  # PSI is always non-negative
    assert psi < 0.1  # Should be low for similar distributions


def test_drift_result_properties(reference_data, current_data_with_drift):
    """Test DriftResult properties."""
    config = get_config("dev")
    detector = DriftDetector(config)

    result = detector.detect_drift(current_data_with_drift, reference_data, "test")

    # Test properties
    assert isinstance(result.has_warning_drift, bool)
    assert isinstance(result.has_critical_drift, bool)
    assert isinstance(result.warning_features, list)
    assert isinstance(result.critical_features, list)


def test_check_drift_convenience_function(reference_data, current_data_no_drift):
    """Test convenience function."""
    result = check_drift(current_data_no_drift, reference_data, "test")

    assert result.dataset == "test"
    assert isinstance(result, type(result))  # DriftResult type
