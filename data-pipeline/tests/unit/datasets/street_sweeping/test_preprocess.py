"""Unit tests for Street Sweeping data preprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.street_sweeping.preprocess import StreetSweepingPreprocessor


@pytest.fixture
def preprocessor():
    return StreetSweepingPreprocessor()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "_id": [1, 2, 3],
            "main_id": ["12345", "67890", "11111"],
            "st_name": ["main st", "elm st", "oak ave"],
            "dist": ["1A", "1B", "2A"],
            "dist_name": ["District 1A", "District 1B", "District 2A"],
            "side": ["left", "right", "left"],
            "from": ["elm st", "oak st", "pine st"],
            "to": ["oak st", "pine st", "maple st"],
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


def test_get_dataset_name(preprocessor):
    assert preprocessor.get_dataset_name() == "street_sweeping"


def test_get_required_columns(preprocessor):
    required = preprocessor.get_required_columns()
    assert "_id" in required
    assert "full_street_name" in required
    assert "district" in required


def test_transform_empty_df(preprocessor):
    df = pd.DataFrame()
    result = preprocessor.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_standardize_strings(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert result["full_street_name"].str.isupper().all()
    assert result["district"].str.isupper().all()


def test_drop_duplicates(preprocessor, sample_df):
    df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = preprocessor.transform(df_with_dupes)
    assert result["_id"].duplicated().sum() == 0


def test_handle_missing_values(preprocessor, sample_df):
    sample_df.loc[0, "dist"] = None
    result = preprocessor.transform(sample_df)
    assert result["district"].isna().sum() == 0


def test_column_mapping(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert "full_street_name" in result.columns
    assert "district" in result.columns
    assert "sam_street_id" in result.columns
    assert "side_of_street" in result.columns
