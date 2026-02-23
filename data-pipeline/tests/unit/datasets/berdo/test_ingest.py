"""Unit tests for BERDO data ingester."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.berdo.ingest import BerdoIngester


@pytest.fixture
def ingester():
    return BerdoIngester()


@pytest.fixture
def sample_records():
    return [
        {
            "_id": 1,
            "reporting_year": "2022",
            "property_name": "Test Building",
            "address": "100 MAIN ST",
            "zip": "02101",
            "property_type": "Office",
            "gross_floor_area": "50000",
            "site_energy_use_kbtu": "1000000",
            "total_ghg_emissions": "500",
            "energy_star_score": "75",
            "electricity_use_grid_purchase": "600000",
            "natural_gas_use": "400000",
            "lat": "42.3601",
            "long": "-71.0589",
        }
    ]


def test_get_dataset_name(ingester):
    assert ingester.get_dataset_name() == "berdo"


def test_get_watermark_field(ingester):
    assert ingester.get_watermark_field() == "reporting_year"


def test_get_primary_key(ingester):
    assert ingester.get_primary_key() == "_id"


@patch("src.datasets.berdo.ingest.requests.get")
def test_fetch_data_success(mock_get, ingester, sample_records):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "result": {"records": sample_records},
    }
    mock_get.return_value = mock_resp

    df = ingester.fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "property_name" in df.columns


@patch("src.datasets.berdo.ingest.requests.get")
def test_fetch_data_empty(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "result": {"records": []}}
    mock_get.return_value = mock_resp

    df = ingester.fetch_data()
    assert df.empty


@patch("src.datasets.berdo.ingest.requests.get")
def test_fetch_data_api_error(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match="HTTP 500"):
        ingester.fetch_data()