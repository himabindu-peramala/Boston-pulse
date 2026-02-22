"""
Tests for Data Slicer

Tests data slicing for fairness evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.bias.data_slicer import DataSlice, DataSlicer, slice_data
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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_slicer_initialization():
    """Test DataSlicer initialization."""
    config = get_config("dev")
    slicer = DataSlicer(config)
    assert slicer.config == config


def test_slicer_default_config():
    """Test DataSlicer uses default config when none provided."""
    slicer = DataSlicer()
    assert slicer.config is not None


# ---------------------------------------------------------------------------
# DataSlice dataclass
# ---------------------------------------------------------------------------


def test_data_slice_repr():
    """Test DataSlice __repr__."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    sl = DataSlice(dimension="hood", value="Roxbury", data=df, size=3, percentage=30.0)
    r = repr(sl)
    assert "hood" in r
    assert "Roxbury" in r


def test_data_slice_fields():
    """Test DataSlice fields are stored correctly."""
    df = pd.DataFrame({"x": range(5)})
    sl = DataSlice(dimension="d", value="v", data=df, size=5, percentage=50.0)
    assert sl.size == 5
    assert sl.percentage == 50.0
    assert len(sl.data) == 5


# ---------------------------------------------------------------------------
# Categorical slicing
# ---------------------------------------------------------------------------


def test_slice_by_category(sample_data):
    """Test categorical slicing produces correct number of slices."""
    slicer = DataSlicer()
    slices = slicer.slice_by_category(sample_data, "neighborhood")

    assert len(slices) == 3
    assert all(s.dimension == "neighborhood" for s in slices)
    assert all(s.size > 0 for s in slices)
    assert slices[0].size >= slices[1].size >= slices[2].size


def test_slice_by_category_min_size(sample_data):
    """Test categorical slicing filters out small slices."""
    slicer = DataSlicer()
    slices = slicer.slice_by_category(sample_data, "neighborhood", min_slice_size=500)
    assert all(s.size >= 500 for s in slices)


def test_slice_by_category_missing_column(sample_data):
    """Test categorical slicing on a missing column returns empty list."""
    slicer = DataSlicer()
    slices = slicer.slice_by_category(sample_data, "nonexistent_column")
    assert slices == []


def test_slice_by_category_single_value():
    """Test slicing when column has a single unique value."""
    df = pd.DataFrame({"hood": ["Downtown"] * 100, "val": range(100)})
    slicer = DataSlicer()
    slices = slicer.slice_by_category(df, "hood")
    assert len(slices) == 1
    assert slices[0].value == "Downtown"


# ---------------------------------------------------------------------------
# Quantile slicing
# ---------------------------------------------------------------------------


def test_slice_by_quantiles(sample_data):
    """Test quantile-based slicing produces correct count."""
    slicer = DataSlicer()
    slices = slicer.slice_by_quantiles(sample_data, "income", num_quantiles=4)

    assert len(slices) == 4
    assert all(s.dimension == "income_quantile" for s in slices)
    assert all(s.size > 0 for s in slices)


def test_slice_by_quantiles_with_labels(sample_data):
    """Test quantile slicing with custom labels."""
    slicer = DataSlicer()
    labels = ["Low", "Medium-Low", "Medium-High", "High"]
    slices = slicer.slice_by_quantiles(sample_data, "income", num_quantiles=4, labels=labels)

    assert len(slices) == 4
    assert all(s.value in labels for s in slices)


def test_slice_by_quantiles_total_size(sample_data):
    """Test that quantile slices together cover the full dataset."""
    slicer = DataSlicer()
    slices = slicer.slice_by_quantiles(sample_data, "income", num_quantiles=4)
    total = sum(s.size for s in slices)
    assert total == len(sample_data)


# ---------------------------------------------------------------------------
# Range slicing
# ---------------------------------------------------------------------------


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


def test_slice_by_ranges_values_in_correct_bucket(sample_data):
    """Test that values fall in the correct range bucket."""
    slicer = DataSlicer()
    ranges = [(0, 12, "first_half"), (12, 24, "second_half")]
    slices = slicer.slice_by_ranges(sample_data, "hour_of_day", ranges)

    first_half = next(s for s in slices if s.value == "first_half")
    assert (first_half.data["hour_of_day"] < 12).all()


# ---------------------------------------------------------------------------
# Time-of-day convenience
# ---------------------------------------------------------------------------


def test_slice_by_time_of_day(sample_data):
    """Test convenience method for time slicing."""
    slicer = DataSlicer()
    slices = slicer.slice_by_time_of_day(sample_data)

    assert len(slices) == 4
    time_periods = ["night", "morning", "afternoon", "evening"]
    assert all(s.value in time_periods for s in slices)


def test_slice_by_time_of_day_custom_column():
    """Test time slicing with a non-default column name."""
    df = pd.DataFrame({"custom_hour": list(range(24)) * 10})
    slicer = DataSlicer()
    slices = slicer.slice_by_time_of_day(df, hour_column="custom_hour")
    assert len(slices) == 4


# ---------------------------------------------------------------------------
# Default slices
# ---------------------------------------------------------------------------


def test_get_default_slices(sample_data):
    """Test getting default slices based on config."""
    slicer = DataSlicer()
    all_slices = slicer.get_default_slices(sample_data, "test")

    assert isinstance(all_slices, dict)
    if "neighborhood" in all_slices:
        assert len(all_slices["neighborhood"]) > 0


# ---------------------------------------------------------------------------
# Cross slicing
# ---------------------------------------------------------------------------


def test_cross_slice(sample_data):
    """Test cross-dimensional slicing."""
    slicer = DataSlicer()
    slices = slicer.cross_slice(sample_data, ["neighborhood", "hour_of_day"], min_slice_size=5)

    assert len(slices) > 0
    assert all("_x_" in s.dimension for s in slices)


def test_cross_slice_min_size_filters(sample_data):
    """Test cross slice min_size properly filters small slices."""
    slicer = DataSlicer()
    slices_small = slicer.cross_slice(sample_data, ["neighborhood", "hour_of_day"], min_slice_size=1)
    slices_large = slicer.cross_slice(sample_data, ["neighborhood", "hour_of_day"], min_slice_size=200)
    assert len(slices_small) >= len(slices_large)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


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


def test_get_slice_summary_empty():
    """Test slice summary with empty list returns empty DataFrame."""
    slicer = DataSlicer()
    summary = slicer.get_slice_summary([])
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------


def test_slice_data_percentage(sample_data):
    """Test that slice percentages sum to approximately 100%."""
    slicer = DataSlicer()
    slices = slicer.slice_by_category(sample_data, "neighborhood")
    total_percentage = sum(s.percentage for s in slices)
    assert abs(total_percentage - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_convenience_slice_data_function(sample_data):
    """Test convenience slice_data function."""
    slices = slice_data(sample_data, "neighborhood", slice_type="categorical")
    assert len(slices) > 0
    assert all(s.dimension == "neighborhood" for s in slices)


def test_slice_data_auto_type(sample_data):
    """Test automatic slice type detection."""
    slices_cat = slice_data(sample_data, "neighborhood", slice_type="auto")
    assert len(slices_cat) > 0

    slices_num = slice_data(sample_data, "income", slice_type="auto")
    assert len(slices_num) > 0
    assert "quantile" in slices_num[0].dimension


def test_slice_data_quantile_type(sample_data):
    """Test quantile slice type via convenience function."""
    slices = slice_data(sample_data, "income", slice_type="quantile")
    assert len(slices) > 0


def test_slice_data_with_kwargs(sample_data):
    """Test convenience function passes kwargs to underlying slicer."""
    slices = slice_data(sample_data, "income", slice_type="quantile", num_quantiles=3)
    assert len(slices) == 3
