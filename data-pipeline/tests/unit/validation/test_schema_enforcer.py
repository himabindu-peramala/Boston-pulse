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
    ValidationStage,
)


@pytest.fixture(autouse=True)
def mock_storage_client():
    """Mock storage.Client for all tests in this module."""
    with patch("src.validation.schema_registry.storage.Client"):
        yield


@pytest.fixture
def mock_gcs():
    """Mock GCS client for unit tests."""
    with patch("src.validation.schema_registry.storage.Client") as mock_client:
        mock_client.return_value = MagicMock()
        yield mock_client


@pytest.fixture
def mock_schema_registry(mock_gcs):
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


def test_enforcer_initialization(mock_gcs):
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
            "occurred_date": pd.date_range("2030-01-01", periods=100, tz="UTC"),  # Future dates
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


@pytest.fixture
def sample_geo_data():
    """Sample data with geographic coordinates."""
    return pd.DataFrame(
        {
            "latitude": [42.35, 42.40, 50.0],  # Last one is out of bounds
            "longitude": [-71.05, -71.10, -80.0],
            "incident_number": ["1", "2", "3"],
        }
    )


def test_validate_geographic_bounds(mock_storage_client, sample_geo_data):
    """Test geographic bounds validation."""
    config = get_config("dev")
    enforcer = SchemaEnforcer(config)
    result = ValidationResult(dataset="test", stage=ValidationStage.RAW, is_valid=True)

    enforcer._validate_geographic_bounds(sample_geo_data, result)

    assert result.has_warnings
    assert any("outside Boston bounds" in error for error in result.warnings)


def test_ensure_schema_exists_local_fallback(mock_storage_client):
    """Test that enforcer can fall back to local schema file."""
    config = get_config("dev")

    with patch("src.validation.schema_enforcer.SchemaRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        # Mock schema not in GCS
        mock_registry.schema_exists.return_value = False

        enforcer = SchemaEnforcer(config)

        # Mock local file existence and content
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value='{"properties": {"col1": {"type": "string"}}}',
            ),
        ):
            enforcer._ensure_schema_registered("crime", ValidationStage.RAW)

            assert mock_registry.register_schema.called
            args, kwargs = mock_registry.register_schema.call_args
            assert kwargs["dataset"] == "crime"


def test_enforce_validation_processed(mock_schema_registry, sample_crime_data):
    """Test enforcement at processed stage."""
    config = get_config("dev")
    df = sample_crime_data.copy()

    # Successful validation
    result = enforce_validation(df, "crime", ValidationStage.PROCESSED, config=config)
    assert result.is_valid
    assert result.stage == ValidationStage.PROCESSED


def test_enforce_validation_features(mock_schema_registry, sample_crime_data):
    """Test enforcement at features stage."""
    config = get_config("dev")
    df = sample_crime_data.copy()

    # Successful validation
    result = enforce_validation(df, "crime", ValidationStage.FEATURES, config=config)
    assert result.is_valid
    assert result.stage == ValidationStage.FEATURES
