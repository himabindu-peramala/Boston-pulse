"""Unit tests for BERDO data preprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.berdo.preprocess import BerdoPreprocessor


@pytest.fixture
def preprocessor():
    return BerdoPreprocessor()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "_id": [1, 2, 3],
            "reporting_year": ["2022", "2022", "2021"],
            "property_name": ["test building", "city hall", "park plaza"],
            "address": ["100 main st", "1 city hall sq", "50 park plaza"],
            "zip": ["02101", "02108", "02116"],
            "property_type": ["Office", "Government", "Hotel"],
            "gross_floor_area": ["50000", "100000", "75000"],
            "site_energy_use_kbtu": ["1000000", "2000000", "1500000"],
            "total_ghg_emissions": ["500", "1000", "750"],
            "energy_star_score": ["75", "60", "45"],
            "electricity_use_grid_purchase": ["600000", "1200000", "900000"],
            "natural_gas_use": ["400000", "800000", "600000"],
            "lat": ["42.3601", "42.3601", "42.3490"],
            "long": ["-71.0589", "-71.0579", "-71.0700"],
        }
    )


def test_get_dataset_name(preprocessor):
    assert preprocessor.get_dataset_name() == "berdo"


def test_get_required_columns(preprocessor):
    required = preprocessor.get_required_columns()
    assert "_id" in required
    assert "reporting_year" in required
    assert "total_ghg_emissions" in required


def test_transform_empty_df(preprocessor):
    df = pd.DataFrame()
    result = preprocessor.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_standardize_strings(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert result["property_name"].str.isupper().all()
    assert result["property_type"].str.isupper().all()


def test_numeric_fields(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert pd.api.types.is_float_dtype(result["total_ghg_emissions"])
    assert pd.api.types.is_float_dtype(result["site_energy_use_kbtu"])


def test_drop_duplicates(preprocessor, sample_df):
    df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = preprocessor.transform(df_with_dupes)
    assert result["_id"].duplicated().sum() == 0


def test_validate_coordinates(preprocessor, sample_df):
    sample_df.loc[0, "lat"] = "999"
    result = preprocessor.transform(sample_df)
    assert pd.isna(result.loc[0, "lat"])
