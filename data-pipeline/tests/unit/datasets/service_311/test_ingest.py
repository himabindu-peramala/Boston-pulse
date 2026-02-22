"""
Unit tests for Service311Ingester.

Tests the 311 data ingestion from Analyze Boston API.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.service_311.ingest import Service311Ingester, ingest_311_data


class TestService311Ingester:
    """Test cases for Service311Ingester class."""

    @pytest.fixture
    def ingester(self):
        """Create a Service311Ingester instance."""
        return Service311Ingester()

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response matching Analyze Boston format."""
        return {
            "success": True,
            "result": {
                "records": [
                    {
                        "case_id": "101001",
                        "open_date": "2024-01-15T14:30:00",
                        "case_topic": "Sanitation",
                        "service_name": "Trash Pickup",
                        "assigned_department": "Public Works",
                        "case_status": "Open",
                        "neighborhood": "South End",
                        "latitude": 42.3456,
                        "longitude": -71.0789,
                    },
                    {
                        "case_id": "101002",
                        "open_date": "2024-01-15T16:45:00",
                        "case_topic": "Highway",
                        "service_name": "Pothole",
                        "assigned_department": "Public Works",
                        "case_status": "Closed",
                        "neighborhood": "Dorchester",
                        "latitude": 42.3123,
                        "longitude": -71.0567,
                    },
                ],
                "total": 2,
            },
        }

    def test_get_dataset_name(self, ingester):
        """Test dataset name is correct."""
        assert ingester.get_dataset_name() == "service_311"

    def test_get_watermark_field(self, ingester):
        """Test watermark field is correct."""
        assert ingester.get_watermark_field() == "open_date"

    def test_get_primary_key(self, ingester):
        """Test primary key is correct."""
        assert ingester.get_primary_key() == "case_id"

    def test_get_api_endpoint(self, ingester):
        """Test API endpoint is constructed correctly."""
        endpoint = ingester.get_api_endpoint()
        assert "datastore_search_sql" in endpoint
        assert "data.boston.gov" in endpoint

    @patch("src.datasets.service_311.ingest.requests.get")
    def test_fetch_data_success(self, mock_get, ingester, sample_api_response):
        """Test successful data fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        df = ingester.fetch_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "case_id" in df.columns
        assert "open_date" in df.columns
        mock_get.assert_called_once()

    @patch("src.datasets.service_311.ingest.requests.get")
    def test_fetch_data_with_watermark(self, mock_get, ingester, sample_api_response):
        """Test data fetching with watermark filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        since = datetime(2024, 1, 1)
        df = ingester.fetch_data(since=since)

        assert isinstance(df, pd.DataFrame)
        assert mock_get.called
        # Check if sql param contains the date
        args, kwargs = mock_get.call_args
        sql = kwargs.get("params", {}).get("sql", "")
        assert "2024-01-01" in sql

    @patch("src.datasets.service_311.ingest.requests.get")
    def test_fetch_data_api_error(self, mock_get, ingester):
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "error": {"message": "Resource not found"},
        }
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="API error"):
            ingester.fetch_data()

    @patch("src.datasets.service_311.ingest.requests.get")
    def test_run_success(self, mock_get, ingester, sample_api_response):
        """Test successful ingestion run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        result = ingester.run(execution_date="2024-01-15")

        assert result.success
        assert result.dataset == "service_311"
        assert result.rows_fetched == 2
        assert result.execution_date == "2024-01-15"

    def test_validate_schema_valid(self, ingester):
        """Test schema validation with valid data."""
        df = pd.DataFrame(
            {
                "case_id": ["101", "102"],
                "open_date": ["2024-01-15", "2024-01-16"],
            }
        )

        is_valid, errors = ingester.validate_schema(df)

        assert is_valid
        assert len(errors) == 0

    def test_validate_schema_missing_pk(self, ingester):
        """Test schema validation with missing primary key."""
        df = pd.DataFrame({"SOME_COLUMN": ["A", "B"]})

        is_valid, errors = ingester.validate_schema(df)

        assert not is_valid
        assert any("Primary key" in e for e in errors)

    @patch("src.datasets.service_311.ingest.Service311Ingester")
    def test_ingest_311_data(self, mock_ingester_class):
        """Test ingest_311_data convenience function."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(to_dict=lambda: {"success": True})
        mock_ingester_class.return_value = mock_instance

        result = ingest_311_data(execution_date="2024-01-15")

        assert result["success"]
        mock_instance.run.assert_called_once()
