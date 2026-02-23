"""
Tests for Model Card Generator

Tests model card generation in Markdown and JSON formats.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.bias.model_card_generator import ModelCardGenerator, create_model_card


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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_generator_initialization(mock_gcs_client):
    """Test ModelCardGenerator initialization."""
    generator = ModelCardGenerator()

    assert generator.config is not None
    assert generator.bucket_name == generator.config.storage.buckets.main


# ---------------------------------------------------------------------------
# Basic card generation
# ---------------------------------------------------------------------------


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


def test_generate_model_card_custom_version(mock_gcs_client, sample_data):
    """Test that a custom version string is preserved."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        version="v1.2.3",
    )

    assert card.version == "v1.2.3"


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


def test_model_card_created_by(mock_gcs_client, sample_data):
    """Test model card stores custom creator."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        created_by="pipeline-bot",
    )

    assert card.created_by == "pipeline-bot"


def test_model_card_no_optional_summaries(mock_gcs_client, sample_data):
    """Test model card with no optional summaries is all-None."""
    generator = ModelCardGenerator()
    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Minimal",
    )

    assert card.validation_summary is None
    assert card.fairness_summary is None
    assert card.drift_summary is None
    assert card.anomaly_summary is None
    assert card.mitigation_summary is None


# ---------------------------------------------------------------------------
# Summary helpers — XCom-style dicts
# ---------------------------------------------------------------------------


def test_generate_model_card_with_validation(mock_gcs_client, sample_data):
    """Test model card with validation results (XCom dict)."""
    generator = ModelCardGenerator()

    validation_result = {
        "stage": "raw",
        "is_valid": True,
        "error_count": 0,
        "warning_count": 0,
        "errors": [],
        "warnings": [],
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        validation_result=validation_result,
    )

    assert card.validation_summary is not None
    assert card.validation_summary["stage"] == "raw"
    assert card.validation_summary["is_valid"] is True
    assert card.validation_summary["error_count"] == 0
    assert card.validation_summary["errors"] == []


def test_generate_model_card_with_validation_errors(mock_gcs_client, sample_data):
    """Test model card validation summary with errors."""
    generator = ModelCardGenerator()

    validation_result = {
        "stage": "processed",
        "is_valid": False,
        "error_count": 2,
        "warning_count": 1,
        "errors": ["Missing column X", "Type mismatch Y"],
        "warnings": ["High null ratio"],
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        validation_result=validation_result,
    )

    assert card.validation_summary["is_valid"] is False
    assert card.validation_summary["error_count"] == 2
    assert len(card.validation_summary["errors"]) == 2


def test_generate_model_card_with_fairness(mock_gcs_client, sample_data):
    """Test model card with fairness results (XCom dict)."""
    generator = ModelCardGenerator()

    fairness_result = {
        "slices_evaluated": 3,
        "violation_count": 1,
        "critical_count": 0,
        "warning_count": 1,
        "passes_fairness_gate": True,
        "violations": [{"metric": "representation", "message": "slight imbalance"}],
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        fairness_result=fairness_result,
    )

    assert card.fairness_summary is not None
    assert card.fairness_summary["slices_evaluated"] == 3
    assert card.fairness_summary["violation_count"] == 1
    assert card.fairness_summary["passes_gate"] is True


def test_generate_model_card_with_drift(mock_gcs_client, sample_data):
    """Test model card with drift detection results (XCom dict)."""
    generator = ModelCardGenerator()

    drift_result = {
        "features_analyzed": 10,
        "drifted_features": ["age", "income"],
        "warning_count": 1,
        "critical_count": 1,
        "warning_features": ["age"],
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        drift_result=drift_result,
    )

    assert card.drift_summary is not None
    assert card.drift_summary["features_analyzed"] == 10
    assert card.drift_summary["features_with_drift"] == 2
    assert "age" in card.drift_summary["critical_features"]


def test_generate_model_card_with_anomaly(mock_gcs_client, sample_data):
    """Test model card with anomaly detection results (XCom dict)."""
    generator = ModelCardGenerator()

    anomaly_result = {
        "anomaly_count": 5,
        "critical_count": 2,
        "warning_count": 3,
        "anomalies_by_type": {"outlier": 3, "duplicate": 2},
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        anomaly_result=anomaly_result,
    )

    assert card.anomaly_summary is not None
    assert card.anomaly_summary["total_anomalies"] == 5
    assert card.anomaly_summary["critical_count"] == 2
    assert card.anomaly_summary["anomalies_by_type"]["outlier"] == 3


def test_generate_model_card_with_mitigation(mock_gcs_client, sample_data):
    """Test model card with bias mitigation results (XCom dict)."""
    generator = ModelCardGenerator()

    mitigation_result = {
        "mitigation_applied": True,
        "strategy": "reweighting",
        "dimension": "district",
        "rows_before": 100,
        "rows_after": 100,
        "slices_improved": 3,
        "total_slices": 4,
        "weight_range": "0.5 – 1.5",
        "reason": None,
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        mitigation_result=mitigation_result,
    )

    assert card.mitigation_summary is not None
    assert card.mitigation_summary["applied"] is True
    assert card.mitigation_summary["strategy"] == "reweighting"
    assert card.mitigation_summary["slices_improved"] == 3


def test_generate_model_card_mitigation_not_applied(mock_gcs_client, sample_data):
    """Test model card with mitigation that was skipped."""
    generator = ModelCardGenerator()

    mitigation_result = {
        "mitigation_applied": False,
        "strategy": None,
        "dimension": None,
        "rows_before": None,
        "rows_after": None,
        "slices_improved": None,
        "total_slices": None,
        "weight_range": None,
        "reason": "No violations detected",
    }

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
        mitigation_result=mitigation_result,
    )

    assert card.mitigation_summary["applied"] is False
    assert card.mitigation_summary["reason"] == "No violations detected"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def test_generate_markdown_basic(mock_gcs_client, sample_data):
    """Test Markdown generation produces correct headers."""
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


def test_generate_markdown_with_all_sections(mock_gcs_client, sample_data):
    """Test Markdown includes all optional sections when summaries are present."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Full card",
        tags=["crime"],
        validation_result={
            "stage": "raw",
            "is_valid": False,
            "error_count": 1,
            "warning_count": 0,
            "errors": ["Missing required column"],
            "warnings": [],
        },
        fairness_result={
            "slices_evaluated": 2,
            "violation_count": 1,
            "critical_count": 1,
            "warning_count": 0,
            "passes_fairness_gate": False,
            "violations": [{"metric": "outcome_disparity", "message": "large gap"}],
        },
        drift_result={
            "features_analyzed": 5,
            "drifted_features": ["col1"],
            "warning_count": 0,
            "critical_count": 1,
            "warning_features": [],
        },
        anomaly_result={
            "anomaly_count": 2,
            "critical_count": 1,
            "warning_count": 1,
            "anomalies_by_type": {"outlier": 2},
        },
        mitigation_result={
            "mitigation_applied": False,
            "strategy": None,
            "dimension": None,
            "rows_before": None,
            "rows_after": None,
            "slices_improved": None,
            "total_slices": None,
            "weight_range": None,
            "reason": "No violations detected",
        },
    )

    markdown = generator._generate_markdown(card)

    assert "## Validation" in markdown
    assert "## Fairness Evaluation" in markdown
    assert "## Drift Detection" in markdown
    assert "## Anomaly Detection" in markdown
    assert "## Bias Mitigation" in markdown
    assert "FAIL" in markdown
    assert "Missing required column" in markdown


def test_generate_markdown_with_time_range(mock_gcs_client):
    """Test Markdown includes time range when datetime column present."""
    generator = ModelCardGenerator()
    df = pd.DataFrame(
        {
            "id": range(50),
            "date": pd.date_range("2023-01-01", periods=50),
        }
    )

    card = generator.generate_model_card(dataset="time_test", df=df, description="Time range test")
    markdown = generator._generate_markdown(card)

    assert "Time Range" in markdown


def test_generate_markdown_mitigation_applied(mock_gcs_client, sample_data):
    """Test Markdown shows applied mitigation details."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Mitigated",
        mitigation_result={
            "mitigation_applied": True,
            "strategy": "stratified_sampling",
            "dimension": "district",
            "rows_before": 100,
            "rows_after": 90,
            "slices_improved": 2,
            "total_slices": 3,
            "weight_range": None,
            "reason": None,
        },
    )

    markdown = generator._generate_markdown(card)
    assert "stratified_sampling" in markdown
    assert "district" in markdown


# ---------------------------------------------------------------------------
# Dictionary conversion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Save / persist
# ---------------------------------------------------------------------------


def test_save_model_card_json(mock_gcs_client, sample_data, tmp_path):
    """Test saving model card as JSON to local path."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    paths = generator.save_model_card(card, format="json", local_path=str(tmp_path))

    assert "json" in paths
    json_path = paths["json"]
    # Both operands are strings — no PosixPath on left side
    assert card.version in json_path
    assert Path(json_path).exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data["dataset_name"] == "test"


def test_save_model_card_markdown(mock_gcs_client, sample_data, tmp_path):
    """Test saving model card as Markdown to local path."""
    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    paths = generator.save_model_card(card, format="markdown", local_path=str(tmp_path))

    assert "markdown" in paths
    md_path = paths["markdown"]
    assert card.version in md_path
    assert Path(md_path).exists()
    content = Path(md_path).read_text()
    assert "# Model Card: test" in content


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
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


# ---------------------------------------------------------------------------
# Helper internals
# ---------------------------------------------------------------------------


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


def test_extract_time_range_no_dates(mock_gcs_client):
    """Test time range extraction returns None when no datetime columns exist."""
    generator = ModelCardGenerator()

    df = pd.DataFrame({"value": range(10), "label": ["a"] * 10})
    time_range = generator._extract_time_range(df)

    assert time_range is None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_create_model_card_convenience_function(mock_gcs_client, sample_data):
    """Test convenience create_model_card function."""
    card = create_model_card(
        dataset="test",
        df=sample_data,
        description="Test dataset",
    )

    assert card.dataset_name == "test"
    assert card.row_count == len(sample_data)
