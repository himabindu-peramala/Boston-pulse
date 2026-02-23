"""Unit tests for BERDO feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.berdo.features import BerdoFeatureBuilder


@pytest.fixture
def builder():
    return BerdoFeatureBuilder()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "_id": [1, 2, 3],
            "reporting_year": [2022, 2022, 2021],
            "property_name": ["TEST BUILDING", "CITY HALL", "PARK PLAZA"],
            "property_type": ["OFFICE", "GOVERNMENT", "HOTEL"],
            "gross_floor_area": [50000.0, 100000.0, 75000.0],
            "site_energy_use_kbtu": [1000000.0, 2000000.0, 1500000.0],
            "total_ghg_emissions": [500.0, 1000.0, 750.0],
            "energy_star_score": [80.0, 60.0, 40.0],
            "electricity_use_grid_purchase": [600000.0, 1200000.0, 900000.0],
            "natural_gas_use": [400000.0, 800000.0, 600000.0],
            "lat": [42.3601, 42.3601, 42.3490],
            "long": [-71.0589, -71.0579, -71.0700],
        }
    )


def test_get_dataset_name(builder):
    assert builder.get_dataset_name() == "berdo"


def test_get_entity_key(builder):
    assert builder.get_entity_key() == "_id"


def test_build_features_empty_df(builder):
    df = pd.DataFrame()
    result = builder.build_features(df)
    assert isinstance(result, pd.DataFrame)


def test_emissions_per_sqft(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "emissions_per_sqft" in result.columns
    assert result.loc[0, "emissions_per_sqft"] == pytest.approx(500.0 / 50000.0)


def test_energy_per_sqft(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "energy_per_sqft" in result.columns
    assert result.loc[0, "energy_per_sqft"] == pytest.approx(1000000.0 / 50000.0)


def test_high_emitter_flag(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "high_emitter" in result.columns
    assert result["high_emitter"].isin([0, 1]).all()


def test_energy_star_category(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "energy_star_category" in result.columns
    assert result.loc[0, "energy_star_category"] == "High"
    assert result.loc[1, "energy_star_category"] == "Medium"
    assert result.loc[2, "energy_star_category"] == "Low"


def test_electricity_ratio(builder, sample_df):
    result = builder.build_features(sample_df)
    assert "electricity_ratio" in result.columns
    assert result.loc[0, "electricity_ratio"] == pytest.approx(600000.0 / 1000000.0)