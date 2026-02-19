"""
Tests for Data Slicer

Tests data slicing for fairness evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.bias.data_slicer import DataSlicer, slice_data
from src.shared.config import get_config


@pytest.fixture
def sample_data():
    """Sample dataset for slicing tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(1000),
            "neighborhood": np.random.choice(["Downtown", "Roxbury", "Dorchester"], 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "age": np.random.randint(18, 80, 1000),
            "hour_of_day": np.random.randint(0, 24, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


def test_slicer_initialization():
    """Test DataSlicer initialization."""
    config = get_config("dev")
    slicer = DataSlicer(config)

    assert slicer.config == config


def test_slice_by_category(sample_data):
    """Test categorical slicing."""
    slicer = DataSlicer()

    slices = slicer.slice_by_category(sample_data, "neighborhood")

    assert len(slices) == 3  # 3 neighborhoods
    assert all(s.dimension == "neighborhood" for s in slices)
    assert all(s.size > 0 for s in slices)
    # Slices should be sorted by size (largest first)
    assert slices[0].size >= slices[1].size >= slices[2].size


def test_slice_by_category_min_size(sample_data):
    """Test categorical slicing with minimum slice size."""
    slicer = DataSlicer()

    slices = slicer.slice_by_category(sample_data, "neighborhood", min_slice_size=500)

    # Should filter out slices smaller than 500
    assert all(s.size >= 500 for s in slices)


def test_slice_by_quantiles(sample_data):
    """Test quantile-based slicing."""
    slicer = DataSlicer()

    slices = slicer.slice_by_quantiles(sample_data, "income", num_quantiles=4)

    assert len(slices) == 4  # 4 quartiles
    assert all(s.dimension == "income_quantile" for s in slices)
    assert all(s.size > 0 for s in slices)


def test_slice_by_quantiles_with_labels(sample_data):
    """Test quantile slicing with custom labels."""
    slicer = DataSlicer()

    labels = ["Low", "Medium-Low", "Medium-High", "High"]
    slices = slicer.slice_by_quantiles(sample_data, "income", num_quantiles=4, labels=labels)

    assert len(slices) == 4
    assert all(s.value in labels for s in slices)


def test_slice_by_ranges(sample_data):
    """Test range-based slicing."""
    slicer = DataSlicer()

    ranges = [
        (0, 6, "night"),
        (6, 12, "morning"),
        (12, 18, "afternoon"),
        (18, 24, "evening"),
    ]
    slices = slicer.slice_by_ranges(sample_data, "hour_of_day", ranges)

    assert len(slices) == 4
    assert all(s.dimension == "hour_of_day_range" for s in slices)
    expected_labels = ["night", "morning", "afternoon", "evening"]
    assert all(s.value in expected_labels for s in slices)


def test_slice_by_time_of_day(sample_data):
    """Test convenience method for time slicing."""
    slicer = DataSlicer()

    slices = slicer.slice_by_time_of_day(sample_data)

    assert len(slices) == 4
    time_periods = ["night", "morning", "afternoon", "evening"]
    assert all(s.value in time_periods for s in slices)


def test_get_default_slices(sample_data):
    """Test getting default slices based on config."""
    slicer = DataSlicer()

    all_slices = slicer.get_default_slices(sample_data, "test")

    assert isinstance(all_slices, dict)
    # Should include neighborhood (categorical)
    if "neighborhood" in all_slices:
        assert len(all_slices["neighborhood"]) > 0


def test_cross_slice(sample_data):
    """Test cross-dimensional slicing."""
    slicer = DataSlicer()

    slices = slicer.cross_slice(sample_data, ["neighborhood", "hour_of_day"], min_slice_size=5)

    # Should create slices for combinations
    assert len(slices) > 0
    assert all("_x_" in s.dimension for s in slices)


def test_get_slice_summary(sample_data):
    """Test slice summary generation."""
    slicer = DataSlicer()

    slices = slicer.slice_by_category(sample_data, "neighborhood")
    summary = slicer.get_slice_summary(slices)

    assert isinstance(summary, pd.DataFrame)
    assert "dimension" in summary.columns
    assert "value" in summary.columns
    assert "size" in summary.columns
    assert "percentage" in summary.columns
    assert len(summary) == len(slices)


def test_slice_data_percentage(sample_data):
    """Test that slice percentages sum to approximately 100%."""
    slicer = DataSlicer()

    slices = slicer.slice_by_category(sample_data, "neighborhood")
    total_percentage = sum(s.percentage for s in slices)

    assert abs(total_percentage - 100.0) < 0.1  # Should be close to 100%


def test_convenience_slice_data_function(sample_data):
    """Test convenience slice_data function."""
    slices = slice_data(sample_data, "neighborhood", slice_type="categorical")

    assert len(slices) > 0
    assert all(s.dimension == "neighborhood" for s in slices)


def test_slice_data_auto_type(sample_data):
    """Test automatic slice type detection."""
    # Should detect categorical
    slices_cat = slice_data(sample_data, "neighborhood", slice_type="auto")
    assert len(slices_cat) > 0

    # Should detect numerical and use quantiles
    slices_num = slice_data(sample_data, "income", slice_type="auto")
    assert len(slices_num) > 0
    assert "quantile" in slices_num[0].dimension
