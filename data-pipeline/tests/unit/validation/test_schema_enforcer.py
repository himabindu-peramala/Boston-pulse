"""
Tests for Schema Enforcer

Tests three-stage validation system.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.schema_enforcer import (
    SchemaEnforcer,
    ValidationError,
    ValidationStage,
    enforce_validation,
)

@pytest.fixture(autouse=True)
def mock_storage_client():
    """Mock storage.Client for all tests in this module."""
    with patch("src.validation.schema_registry.storage.Client"):
        yield

@pytest.fixture
def mock_schema_registry():
    """Mock SchemaRegistry for testing."""
    with patch("src.validation.schema_enforcer.SchemaRegistry") as mock_registry:
        mock_instance = MagicMock()
        mock_registry.return_value = mock_instance

        # Mock successful validation by default
        mock_instance.validate_dataframe.return_value = (True, [])

        yield mock_instance


@pytest.fixture
def sample_crime_data():
    """Sample crime data for testing."""
    return pd.DataFrame(
        {
            "incident_number": ["1", "2", "3", "4", "5"] * 20,
            "occurred_date": pd.date_range("2024-01-01", periods=100),
            "lat": [42.35] * 100,
            "lon": [-71.05] * 100,
            "description": ["Theft"] * 100,
        }
    )


def test_enforcer_initialization():
    """Test SchemaEnforcer initialization."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    assert enforcer.config == config
    assert enforcer.strict_mode == config.validation.quality_schema.strict_mode


def test_validate_raw_success(mock_schema_registry, sample_crime_data):
    """Test successful raw validation."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    result = enforcer.validate_raw(sample_crime_data, "crime")

    assert result.is_valid
    assert result.dataset == "crime"
    assert result.stage == ValidationStage.RAW
    assert result.row_count == len(sample_crime_data)


def test_validate_raw_below_min_rows(mock_schema_registry):
    """Test validation failure due to insufficient rows."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    # DataFrame with too few rows
    df = pd.DataFrame({"col1": [1, 2, 3]})

    result = enforcer.validate_raw(df, "crime")

    assert not result.is_valid
    assert any("row count" in error.lower() for error in result.errors)


def test_validate_raw_high_null_ratio(mock_schema_registry):
    """Test validation warning/error for high null ratio."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    # DataFrame with high null ratio
    df = pd.DataFrame(
        {
            "col1": [1] * 50 + [None] * 50,
            "col2": range(100),
        }
    )

    result = enforcer.validate_raw(df, "crime")

    # Should have issue for col1
    assert any(
        "col1" in issue.message and "null" in issue.message.lower() for issue in result.issues
    )


def test_validate_processed_geographic_bounds(mock_schema_registry):
    """Test geographic bounds validation."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    # DataFrame with coordinates outside Boston
    df = pd.DataFrame(
        {
            "incident_number": range(100),
            "lat": [40.0] * 50 + [42.35] * 50,  # First 50 outside bounds
            "lon": [-71.05] * 100,
        }
    )

    result = enforcer.validate_processed(df, "crime")

    # Should have warning about out-of-bounds coordinates
    assert any(
        "latitude" in issue.message.lower() or "bounds" in issue.message.lower()
        for issue in result.issues
    )


def test_validate_processed_temporal_bounds(mock_schema_registry):
    """Test temporal bounds validation."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    # DataFrame with future dates
    df = pd.DataFrame(
        {
            "incident_number": range(100),
            "occurred_date": pd.date_range("2030-01-01", periods=100),  # Future dates
        }
    )

    result = enforcer.validate_processed(df, "crime")

    # Should have warning about future dates
    assert any("future" in issue.message.lower() for issue in result.issues)


def test_validate_features_infinite_values(mock_schema_registry):
    """Test feature validation catches infinite values."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    # DataFrame with infinite values
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, float("inf"), 4.0, 5.0] * 20,
            "feature2": [1.0] * 100,
        }
    )

    result = enforcer.validate_features(df, "crime")

    assert not result.is_valid
    assert any("infinite" in error.lower() for error in result.errors)


def test_enforce_validation_raises_on_failure():
    """Test that enforce_validation raises ValidationError on failure."""
    config = get_config("dev")
    config.validation.quality_schema.strict_mode = True

    # DataFrame that will fail validation (too few rows)
    df = pd.DataFrame({"col1": [1, 2]})

    with patch("src.validation.schema_enforcer.SchemaRegistry") as mock_registry:
        mock_instance = MagicMock()
        mock_registry.return_value = mock_instance
        mock_instance.validate_dataframe.return_value = (True, [])
        with pytest.raises(ValidationError) as exc_info:
            enforce_validation(df, "crime", ValidationStage.RAW, config=config)

        assert "crime" in str(exc_info.value)


def test_validation_result_properties(mock_schema_registry, sample_crime_data):
    """Test ValidationResult properties."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)

    result = enforcer.validate_raw(sample_crime_data, "crime")

    # Test properties
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.has_errors, bool)
    assert isinstance(result.has_warnings, bool)
