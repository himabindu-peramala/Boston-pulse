"""Unit tests for BERDO data ingester."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.berdo.ingest import BerdoIngester


@pytest.fixture
def ingester():
    return BerdoIngester()


def _make_excel_bytes():
    """Create a minimal Excel file in memory for mocking."""
    df = pd.DataFrame(
        {
            "BERDO ID": ["B001"],
            "Property Owner Name": ["Test Owner"],
            "Building Address": ["100 Main St"],
            "Building Address City": ["Boston"],
            "Building Address Zip  Code": ["02101"],
            "Largest Property Type": ["Office"],
            "Reported Gross Floor Area (sq ft)": [50000],
            "Total Site Energy Usage (kBtu)": [1000000],
            "Estimated Total GHG Emissions (kgCO2e)": [500],
            "Energy Star Score": [75],
            "Electricity Usage (kWh)": [600000],
            "Natural Gas Usage (kBtu)": [400000],
            "Compliance Status": ["Compliant"],
            "Site EUI (Energy Use Intensity kBtu/ft²)": [20.0],
        }
    )
    buf = BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def test_get_dataset_name(ingester):
    assert ingester.get_dataset_name() == "berdo"


def test_get_watermark_field(ingester):
    assert ingester.get_watermark_field() == "reporting_year"


def test_get_primary_key(ingester):
    assert ingester.get_primary_key() == "_id"


@patch("src.datasets.berdo.ingest.requests.get")
def test_fetch_data_success(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = _make_excel_bytes()
    mock_get.return_value = mock_resp

    df = ingester.fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


@patch("src.datasets.berdo.ingest.requests.get")
def test_fetch_data_api_error(mock_get, ingester):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match="HTTP 500"):
        ingester.fetch_data()
