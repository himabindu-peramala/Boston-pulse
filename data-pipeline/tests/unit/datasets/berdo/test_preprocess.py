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
            "berdo_id": ["B001", "B002", "B003"],
            "property_owner_name": ["test owner", "city hall", "park plaza"],
            "building_address": ["100 main st", "1 city hall sq", "50 park plaza"],
            "building_address_city": ["Boston", "Boston", "Boston"],
            "zip": ["02101", "02108", "02116"],
            "property_type": ["Office", "Government", "Hotel"],
            "gross_floor_area": [50000.0, 100000.0, 75000.0],
            "site_energy_use_kbtu": [1000000.0, 2000000.0, 1500000.0],
            "total_ghg_emissions": [500.0, 1000.0, 750.0],
            "energy_star_score": [75.0, 60.0, 45.0],
            "electricity_usage_kwh": [600000.0, 1200000.0, 900000.0],
            "natural_gas_use": [400000.0, 800000.0, 600000.0],
            "compliance_status": ["Compliant", "Non-Compliant", "Compliant"],
        }
    )


def test_get_dataset_name(preprocessor):
    assert preprocessor.get_dataset_name() == "berdo"


def test_get_required_columns(preprocessor):
    required = preprocessor.get_required_columns()
    assert "_id" in required
    assert "berdo_id" in required
    assert "total_ghg_emissions" in required


def test_transform_empty_df(preprocessor):
    df = pd.DataFrame()
    result = preprocessor.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_standardize_strings(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert result["property_owner_name"].str.isupper().all()
    assert result["property_type"].str.isupper().all()


def test_numeric_fields(preprocessor, sample_df):
    result = preprocessor.transform(sample_df)
    assert pd.api.types.is_float_dtype(result["total_ghg_emissions"])
    assert pd.api.types.is_float_dtype(result["site_energy_use_kbtu"])


def test_drop_duplicates(preprocessor, sample_df):
    df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = preprocessor.transform(df_with_dupes)
    assert result["_id"].duplicated().sum() == 0


def test_handle_missing_values(preprocessor, sample_df):
    sample_df.loc[0, "property_type"] = None
    result = preprocessor.transform(sample_df)
    assert result["property_type"].isna().sum() == 0
