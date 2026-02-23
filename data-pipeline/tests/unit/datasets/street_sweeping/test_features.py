"""Unit tests for Street Sweeping feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.street_sweeping.features import StreetSweepingFeatureBuilder


@pytest.fixture
def builder():
    return StreetSweepingFeatureBuilder()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "_id": [1, 2, 3],
            "sam_street_id": ["12345", "67890", "11111"],
            "full_street_name": ["MAIN ST", "ELM ST", "OAK AVE"],
            "district": ["1A", "1B", "2A"],
            "side_of_street": ["LEFT", "RIGHT", "LEFT"],
            "season_start": ["APR", "APR", "MAY"],
            "season_end": ["NOV", "NOV", "OCT"],
            "week_type": ["EVERY WEEK", "EVERY OTHER WEEK", "EVERY WEEK"],
            "tow_zone": ["YES", "NO", "YES"],
            "lat": [42.3601, 42.3501, 42.3701],
            "long": [-71.0589, -71.0489, -71.0689],
        }
    )


def test_get_dataset_name(builder):
    assert builder.get_dataset_name() == "street_sweeping"


def test_build_features_empty_df(builder):
    df = pd.DataFrame()
    result = builder.build_features(df)
    assert isinstance(result, pd.DataFrame)


def test_tow_enforced_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "tow_enforced" in result.columns
    assert result.loc[0, "tow_enforced"] == 1
    assert result.loc[1, "tow_enforced"] == 0


def test_is_every_week_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "is_every_week" in result.columns
    assert result.loc[0, "is_every_week"] == 1
    assert result.loc[1, "is_every_week"] == 0


def test_season_months_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "season_start_month" in result.columns
    assert "season_end_month" in result.columns
    assert result.loc[0, "season_start_month"] == 4  # APR


def test_active_months_count(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "active_months_count" in result.columns
    assert result.loc[0, "active_months_count"] == 8  # APR to NOV


def test_district_code_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "district_code" in result.columns
    assert result["district_code"].dtype in ["int8", "int16", "int64"]
