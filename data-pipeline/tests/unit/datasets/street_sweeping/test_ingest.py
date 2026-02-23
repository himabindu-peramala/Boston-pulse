"""Unit tests for Street Sweeping data ingester."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.street_sweeping.ingest import StreetSweepingIngester


@pytest.fixture
def ingester():
    return StreetSweepingIngester()


@pytest.fixture
def sample_records():
    return [
        {
            "_id": 1,
            "sam_street_id": "12345",
            "full_street_name": "MAIN ST",
            "from_street": "ELM ST",
            "to_street": "OAK ST",
            "district": "1A",
            "side_of_street": "LEFT",
            "season_start": "APR",
            "season_end": "NOV",
            "week_type": "EVERY WEEK",
            "tow_zone": "YES",
            "lat": "42.3601",
            "long": "-71.0589",
        }
    ]


def test_get_dataset_name(ingester):
    assert ingester.get_dataset_name() == "street_sweeping"


def test_get_watermark_field(ingester):
    assert ingester.get_watermark_field() == "sam_street_id"


def test_get_primary_key(ingester):
    assert ingester.get_primary_key() == "_id"


@patch("src.datasets.street_sweeping.ingest.requests.get")
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
    assert "full_street_name" in df.columns


@patch("src.datasets.street_sweeping.ingest.requests.get")
def test_fetch_data_empty(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "result": {"records": []}}
    mock_get.return_value = mock_resp

    df = ingester.fetch_data()
    assert df.empty


@patch("src.datasets.street_sweeping.ingest.requests.get")
def test_fetch_data_api_error(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match="HTTP 500"):
        ingester.fetch_data()