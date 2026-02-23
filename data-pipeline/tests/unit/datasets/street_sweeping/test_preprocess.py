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
    return pd.DataFrame({
        "_id": [1, 2, 3],
        "sam_street_id": ["12345", "67890", "11111"],
        "full_street_name": ["main st", "elm st", "oak ave"],
        "from_street": ["elm st", "oak st", "pine st"],
        "to_street": ["oak st", "pine st", "maple st"],
        "district": ["1A", "1B", "2A"],
        "side_of_street": ["left", "right", "left"],
        "season_start": ["APR", "APR", "APR"],
        "season_end": ["NOV", "NOV", "NOV"],
        "week_type": ["every week", "every other week", "every week"],
        "tow_zone": ["YES", "NO", "YES"],
        "lat": ["42.3601", "42.3501", "42.3701"],
        "long": ["-71.0589", "-71.0489", "-71.0689"],
    })


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
    sample_df.loc[0, "district"] = None
    result = preprocessor.transform(sample_df)
    assert result["district"].isna().sum() == 0


def test_validate_coordinates(preprocessor, sample_df):
    sample_df.loc[0, "lat"] = "999"  # out of bounds
    result = preprocessor.transform(sample_df)
    assert pd.isna(result.loc[0, "lat"])
