"""
Tests for Schema Registry

Tests GCS-backed schema storage with versioning.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.schema_registry import SchemaRegistry, create_schema_from_dataframe


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client for testing."""
    with patch("src.validation.schema_registry.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_client, mock_bucket


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "incident_number": ["1", "2", "3"],
            "occurred_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "latitude": [42.35, 42.36, 42.37],
            "description": ["Test 1", "Test 2", "Test 3"],
        }
    )


def test_schema_registry_initialization(mock_gcs_client):
    """Test SchemaRegistry initialization."""
    config = get_config("dev")
    registry = SchemaRegistry(config)

    assert registry.config == config
    assert registry.bucket_name == config.storage.buckets.main


def test_register_schema_from_dataframe(mock_gcs_client, sample_dataframe):
    """Test registering schema from DataFrame."""
    _, mock_bucket = mock_gcs_client
    config = get_config("dev")
    registry = SchemaRegistry(config)

    # Register schema
    version = registry.register_schema(
        dataset="crime",
        layer="raw",
        schema=sample_dataframe,
        description="Test schema",
        primary_key="incident_number",
    )

    assert version.startswith("v")
    # Should upload versioned and latest
    assert mock_bucket.blob.call_count >= 2


def test_register_schema_from_dict(mock_gcs_client):
    """Test registering schema from dictionary."""
    _, mock_bucket = mock_gcs_client
    config = get_config("dev")
    registry = SchemaRegistry(config)

    schema = {
        "incident_number": {"type": "string", "nullable": False},
        "occurred_date": {"type": "datetime", "nullable": False},
    }

    version = registry.register_schema(
        dataset="crime",
        layer="raw",
        schema=schema,
        description="Test schema",
    )

    assert version.startswith("v")


def test_create_schema_from_dataframe(sample_dataframe):
    """Test creating schema from DataFrame."""
    schema = create_schema_from_dataframe(
        sample_dataframe,
        primary_key="incident_number",
        description="Test schema",
    )

    assert "description" in schema
    assert schema["primary_key"] == "incident_number"
    assert "schema" in schema
    assert "incident_number" in schema["schema"]


def test_validate_dataframe_success(mock_gcs_client, sample_dataframe):
    """Test successful DataFrame validation."""
    _, mock_bucket = mock_gcs_client
    config = get_config("dev")
    registry = SchemaRegistry(config)

    # Mock schema download
    schema_doc = {
        "metadata": {
            "dataset": "crime",
            "layer": "raw",
            "version": "v1",
            "created_at": datetime.utcnow().isoformat(),
            "created_by": "test",
            "num_columns": 4,
        },
        "schema": {
            "incident_number": {"type": "string", "nullable": False},
            "occurred_date": {"type": "datetime", "nullable": False},
            "latitude": {"type": "float", "nullable": True},
            "description": {"type": "string", "nullable": True},
        },
    }

    mock_blob = MagicMock()
    mock_blob.download_as_string.return_value = json.dumps(schema_doc)
    mock_bucket.blob.return_value = mock_blob

    is_valid, errors = registry.validate_dataframe(sample_dataframe, "crime", "raw")

    assert is_valid
    assert len(errors) == 0


def test_validate_dataframe_missing_columns(mock_gcs_client):
    """Test validation failure due to missing columns."""
    _, mock_bucket = mock_gcs_client
    config = get_config("dev")
    registry = SchemaRegistry(config)

    # DataFrame missing required column
    df = pd.DataFrame({"incident_number": ["1", "2"]})

    schema_doc = {
        "metadata": {"dataset": "crime", "layer": "raw", "version": "v1"},
        "schema": {
            "incident_number": {"type": "string", "nullable": False},
            "occurred_date": {"type": "datetime", "nullable": False},
        },
    }

    mock_blob = MagicMock()
    mock_blob.download_as_string.return_value = json.dumps(schema_doc)
    mock_bucket.blob.return_value = mock_blob

    is_valid, errors = registry.validate_dataframe(df, "crime", "raw")

    assert not is_valid
    assert any("occurred_date" in error for error in errors)


def test_validate_dataframe_null_values(mock_gcs_client):
    """Test validation failure due to null values in non-nullable column."""
    _, mock_bucket = mock_gcs_client
    config = get_config("dev")
    registry = SchemaRegistry(config)

    # DataFrame with null in non-nullable column
    df = pd.DataFrame(
        {
            "incident_number": ["1", None, "3"],
            "description": ["Test", "Test", "Test"],
        }
    )

    schema_doc = {
        "metadata": {"dataset": "crime", "layer": "raw", "version": "v1"},
        "schema": {
            "incident_number": {"type": "string", "nullable": False},
            "description": {"type": "string", "nullable": True},
        },
    }

    mock_blob = MagicMock()
    mock_blob.download_as_string.return_value = json.dumps(schema_doc)
    mock_bucket.blob.return_value = mock_blob

    is_valid, errors = registry.validate_dataframe(df, "crime", "raw")

    assert not is_valid
    assert any("incident_number" in error and "null" in error.lower() for error in errors)
