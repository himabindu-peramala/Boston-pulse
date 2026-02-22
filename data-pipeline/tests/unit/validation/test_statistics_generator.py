"""
Tests for Statistics Generator

Tests data statistics generation, storage, and comparison.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.shared.config import get_config
from src.validation.statistics_generator import (
    FeatureStatistics,
    DataStatistics,
    StatisticsGenerator,
    generate_and_save_statistics,
    get_latest_statistics,
)


@pytest.fixture(autouse=True)
def mock_storage_client():
    """Mock storage.Client for all tests in this module."""
    with patch("src.validation.statistics_generator.storage.Client"):
        yield


@pytest.fixture
def sample_df():
    """Sample DataFrame for statistics generation."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric": np.random.normal(100, 10, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
            "with_nulls": [1.0, 2.0, None, 4.0, 5.0] * 20,
            "int_col": range(100),
        }
    )


def test_feature_statistics_to_dict():
    """Test FeatureStatistics conversion to dictionary."""
    stats = FeatureStatistics(
        name="test",
        dtype="float64",
        count=100,
        num_missing=5,
        missing_ratio=0.05,
        mean=10.0,
        std=2.0,
        min=5.0,
        max=15.0,
    )
    
    d = stats.to_dict()
    assert d["name"] == "test"
    assert d["mean"] == 10.0
    assert d["num_unique"] is None


def test_data_statistics_to_from_dict():
    """Test DataStatistics dictionary serialization/deserialization."""
    now = datetime.now(UTC)
    stats = DataStatistics(
        dataset="test",
        layer="raw",
        date=now,
        num_examples=100,
        num_features=1,
        feature_statistics=[
            FeatureStatistics(name="f1", dtype="int", count=100, num_missing=0, missing_ratio=0.0)
        ]
    )
    
    d = stats.to_dict()
    assert d["dataset"] == "test"
    
    new_stats = DataStatistics.from_dict(d)
    assert new_stats.dataset == "test"
    assert len(new_stats.feature_statistics) == 1
    assert new_stats.feature_statistics[0].name == "f1"


def test_generator_initialization():
    """Test StatisticsGenerator initialization."""
    config = get_config("dev")
    generator = StatisticsGenerator(config)
    
    assert generator.config == config
    assert generator.bucket_name == config.storage.buckets.main


def test_generate_statistics(sample_df):
    """Test generating statistics for a DataFrame."""
    generator = StatisticsGenerator()
    stats = generator.generate_statistics(sample_df, "test", "raw")
    
    assert stats.dataset == "test"
    assert stats.num_examples == 100
    assert stats.num_features == 4
    
    # Check numeric feature
    num_stats = stats.get_feature_stats("numeric")
    assert num_stats.mean is not None
    assert num_stats.std is not None
    
    # Check categorical feature
    cat_stats = stats.get_feature_stats("categorical")
    assert cat_stats.num_unique == 3
    assert "A" in cat_stats.value_counts
    
    # Check null handling
    null_stats = stats.get_feature_stats("with_nulls")
    assert null_stats.num_missing == 20
    assert null_stats.missing_ratio == 0.2


def test_save_load_statistics(sample_df):
    """Test saving and loading statistics from GCS."""
    with patch("src.validation.statistics_generator.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        generator = StatisticsGenerator()
        stats = generator.generate_statistics(sample_df, "test", "raw")
        
        # Save
        path = generator.save_statistics(stats)
        assert "gs://" in path
        assert mock_blob.upload_from_string.called
        
        # Load
        mock_blob.download_as_string.return_value = json.dumps(stats.to_dict())
        loaded_stats = generator.load_statistics("test", "raw")
        assert loaded_stats.dataset == "test"
        assert loaded_stats.num_examples == 100


def test_compare_statistics(sample_df):
    """Test comparing two sets of statistics."""
    generator = StatisticsGenerator()
    
    # Create reference and current
    ref_stats = generator.generate_statistics(sample_df, "test", "raw")
    curr_df = sample_df.copy()
    curr_df["numeric"] = curr_df["numeric"] + 10  # Shift mean
    curr_stats = generator.generate_statistics(curr_df, "test", "raw")
    
    comparison = generator.compare_statistics(curr_stats, ref_stats)
    
    assert "row_count_change" in comparison
    
    # Check feature comparison
    feat_comps = {f["feature"]: f for f in comparison["feature_comparisons"]}
    assert "numeric" in feat_comps
    assert feat_comps["numeric"]["mean_change"] == pytest.approx(10.0)


def test_visualize_statistics(sample_df, tmp_path):
    """Test HTML visualization generation."""
    generator = StatisticsGenerator()
    stats = generator.generate_statistics(sample_df, "test", "raw")
    
    output_file = str(tmp_path / "stats.html")
    path = generator.visualize_statistics(stats, output_path=output_file)
    
    assert path == output_file
    with open(output_file, "r") as f:
        content = f.read()
        assert "Statistics Report: test/raw" in content
        assert "numeric" in content


def test_convenience_functions(sample_df):
    """Test convenience functions."""
    with patch("src.validation.statistics_generator.StatisticsGenerator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen
        
        # generate_and_save
        generate_and_save_statistics(sample_df, "test", "raw")
        assert mock_gen.generate_statistics.called
        assert mock_gen.save_statistics.called
        
        # get_latest
        get_latest_statistics("test", "raw")
        assert mock_gen.load_statistics.called
