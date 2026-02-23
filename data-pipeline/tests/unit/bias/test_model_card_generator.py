"""
Tests for Model Card Generator

Tests model card generation in Markdown and JSON formats.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.bias.fairness_checker import FairnessResult
from src.bias.model_card_generator import ModelCardGenerator, create_model_card
from src.validation.schema_enforcer import ValidationResult, ValidationStage


@pytest.fixture
def sample_data():
    """Sample dataset for model card generation."""
    return pd.DataFrame(
        {
            "id": range(100),
            "value": range(100),
            "category": ["A"] * 50 + ["B"] * 50,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client for testing."""
    with patch("src.bias.model_card_generator.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_client, mock_bucket


def test_generator_initialization(mock_gcs_client):
    """Test ModelCardGenerator initialization."""
    generator = ModelCardGenerator()

    assert generator.config is not None
    assert generator.bucket_name == generator.config.storage.buckets.main


def test_generate_model_card_basic(mock_gcs_client, sample_data):
    """Test basic model card generation."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    assert card.dataset_name == "test"
    assert card.description == "Test dataset"
    assert card.row_count == len(sample_data)
    assert card.column_count == len(sample_data.columns)
    assert card.version.startswith("v")
    assert isinstance(card.created_at, datetime)


def test_generate_model_card_with_validation(mock_gcs_client, sample_data):
    """Test model card with validation results."""
    generator = ModelCardGenerator()

    validation_result = ValidationResult(
        dataset="test",
        stage=ValidationStage.RAW,
        is_valid=True,
        row_count=100,
        column_count=4,
    )

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        validation_result=validation_result,
    )

    assert card.validation_summary is not None
    assert card.validation_summary["stage"] == "raw"
    assert card.validation_summary["is_valid"] is True


def test_generate_model_card_with_fairness(mock_gcs_client, sample_data):
    """Test model card with fairness results."""
    generator = ModelCardGenerator()

    fairness_result = FairnessResult(
        dataset="test",
        evaluated_at=datetime.now(UTC),
        slices_evaluated=3,
    )

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        fairness_result=fairness_result,
    )

    assert card.fairness_summary is not None
    assert card.fairness_summary["slices_evaluated"] == 3


def test_generate_markdown(mock_gcs_client, sample_data):
    """Test Markdown generation."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset for Markdown",
    )

    markdown = generator._generate_markdown(card)

    assert isinstance(markdown, str)
    assert "# Model Card: test" in markdown
    assert "Test dataset for Markdown" in markdown
    assert "Dataset Characteristics" in markdown
    assert f"**Rows:** {len(sample_data):,}" in markdown


def test_model_card_to_dict(mock_gcs_client, sample_data):
    """Test model card dictionary conversion."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        tags=["test", "sample"],
    )

    card_dict = card.to_dict()

    assert isinstance(card_dict, dict)
    assert card_dict["dataset_name"] == "test"
    assert card_dict["row_count"] == len(sample_data)
    assert "test" in card_dict["tags"]


def test_save_model_card_json(mock_gcs_client, sample_data, tmp_path):
    """Test saving model card as JSON."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    # Save to local path for testing
    paths = generator.save_model_card(card, format="json", local_path=str(tmp_path))

    assert "json" in paths
    # Verify file was created
    json_path = paths["json"]
    assert str(tmp_path / card.version) in json_path or card.version in json_path


def test_save_model_card_markdown(mock_gcs_client, sample_data, tmp_path):
    """Test saving model card as Markdown."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    # Save to local path for testing
    paths = generator.save_model_card(card, format="markdown", local_path=str(tmp_path))

    assert "markdown" in paths


def test_save_model_card_both_formats(mock_gcs_client, sample_data, tmp_path):
    """Test saving model card in both formats."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    paths = generator.save_model_card(card, format="both", local_path=str(tmp_path))

    assert "json" in paths
    assert "markdown" in paths


def test_extract_time_range(mock_gcs_client):
    """Test time range extraction from DataFrame."""
    generator = ModelCardGenerator()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100),
        }
    )

    time_range = generator._extract_time_range(df)

    assert time_range is not None
    assert time_range["column"] == "date"
    assert "2024-01-01" in time_range["min"]


def test_create_model_card_convenience_function(mock_gcs_client, sample_data):
    """Test convenience create_model_card function."""
    card = create_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    assert card.dataset_name == "test"
    assert card.row_count == len(sample_data)


def test_model_card_with_tags(mock_gcs_client, sample_data):
    """Test model card with tags."""
    generator = ModelCardGenerator()

    tags = ["test", "crime", "boston"]
    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        tags=tags,
    )

    assert card.tags == tags


def test_model_card_with_primary_key(mock_gcs_client, sample_data):
    """Test model card with primary key."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        primary_key="id",
    )

    assert card.primary_key == "id"
