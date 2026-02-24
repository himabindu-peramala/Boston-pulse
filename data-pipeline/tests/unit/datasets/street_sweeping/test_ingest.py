"""Unit tests for Street Sweeping data ingester."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.street_sweeping.ingest import StreetSweepingIngester


@pytest.fixture
def ingester():
    return StreetSweepingIngester()


SAMPLE_CSV = """_id,main_id,st_name,dist,dist_name,side,from,to,start_time,end_time,week_1,week_2,week_3,week_4,year_round,one_way,miles
1,12345,MAIN ST,1A,DISTRICT 1A,LEFT,ELM ST,OAK ST,8:00 AM,12:00 PM,Y,Y,Y,Y,Y,N,0.5
"""


def test_get_dataset_name(ingester):
    assert ingester.get_dataset_name() == "street_sweeping"


def test_get_watermark_field(ingester):
    assert ingester.get_watermark_field() == "sam_street_id"


def test_get_primary_key(ingester):
    assert ingester.get_primary_key() == "_id"


@patch("src.datasets.street_sweeping.ingest.requests.get")
def test_fetch_data_success(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CSV
    mock_get.return_value = mock_resp

    df = ingester.fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "st_name" in df.columns or "full_street_name" in df.columns


@patch("src.datasets.street_sweeping.ingest.requests.get")
def test_fetch_data_api_error(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match="HTTP 500"):
        ingester.fetch_data()
