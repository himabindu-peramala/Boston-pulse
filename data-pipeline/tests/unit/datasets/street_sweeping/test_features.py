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
            "district_name": ["DISTRICT 1A", "DISTRICT 1B", "DISTRICT 2A"],
            "side_of_street": ["LEFT", "RIGHT", "LEFT"],
            "from_street": ["ELM ST", "OAK ST", "PINE ST"],
            "to_street": ["OAK ST", "PINE ST", "MAPLE ST"],
            "start_time": ["8:00 AM", "9:00 AM", "10:00 AM"],
            "end_time": ["12:00 PM", "1:00 PM", "2:00 PM"],
            "week_1": ["Y", "Y", "Y"],
            "week_2": ["Y", "N", "Y"],
            "week_3": ["Y", "N", "Y"],
            "week_4": ["Y", "N", "Y"],
            "year_round": ["Y", "N", "Y"],
            "one_way": ["N", "Y", "N"],
            "miles": [0.5, 0.3, 0.7],
        }
    )


def test_get_dataset_name(builder):
    assert builder.get_dataset_name() == "street_sweeping"


def test_get_entity_key(builder):
    assert builder.get_entity_key() == "sam_street_id"


def test_build_features_empty_df(builder):
    df = pd.DataFrame()
    result = builder.build_features(df)
    assert isinstance(result, pd.DataFrame)


def test_is_year_round_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "is_year_round" in result.columns
    assert result.loc[0, "is_year_round"] == 1
    assert result.loc[1, "is_year_round"] == 0


def test_is_every_week_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "is_every_week" in result.columns
    assert result.loc[0, "is_every_week"] == 1
    assert result.loc[1, "is_every_week"] == 0


def test_district_code_feature(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "district_code" in result.columns
    assert result["district_code"].dtype in ["int8", "int16", "int64"]
