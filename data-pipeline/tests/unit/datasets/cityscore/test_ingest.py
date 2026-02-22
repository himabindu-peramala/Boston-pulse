"""
Unit tests for CityScoreIngester.

Tests the CityScore data ingestion from Analyze Boston API.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.cityscore.ingest import CityScoreIngester, ingest_cityscore_data


class TestCityScoreIngester:
    """Test cases for CityScoreIngester class."""

    @pytest.fixture
    def ingester(self):
        """Create a CityScoreIngester instance."""
        return CityScoreIngester()

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response matching Analyze Boston format."""
        return {
            "success": True,
            "result": {
                "records": [
                    {
                        "_id": 1,
                        "score_calculated_ts": "2024-01-15T00:00:00",
                        "metric_name": "Trash On-Time %",
                        "day_score": 0.85,
                        "target": 0.80,
                    },
                    {
                        "_id": 2,
                        "score_calculated_ts": "2024-01-15T00:00:00",
                        "metric_name": "Pothole On-Time %",
                        "day_score": 0.92,
                        "target": 0.90,
                    },
                ],
                "total": 2,
            },
        }

    def test_get_dataset_name(self, ingester):
        """Test dataset name is correct."""
        assert ingester.get_dataset_name() == "cityscore"

    def test_get_watermark_field(self, ingester):
        """Test watermark field is correct."""
        assert ingester.get_watermark_field() == "score_calculated_ts"

    def test_get_primary_key(self, ingester):
        """Test primary key is correct."""
        assert ingester.get_primary_key() == "_id"

    def test_get_api_endpoint(self, ingester):
        """Test API endpoint is constructed correctly."""
        endpoint = ingester.get_api_endpoint()
        assert "datastore_search" in endpoint
        assert "data.boston.gov" in endpoint

    @patch("src.datasets.cityscore.ingest.requests.get")
    def test_fetch_data_success(self, mock_get, ingester, sample_api_response):
        """Test successful data fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        df = ingester.fetch_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "_id" in df.columns
        assert "score_calculated_ts" in df.columns
        mock_get.assert_called_once()

    @patch("src.datasets.cityscore.ingest.requests.get")
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

    @patch("src.datasets.cityscore.ingest.requests.get")
    def test_run_success(self, mock_get, ingester, sample_api_response):
        """Test successful ingestion run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        result = ingester.run(execution_date="2024-01-15")

        assert result.success
        assert result.dataset == "cityscore"
        assert result.rows_fetched == 2

    @patch("src.datasets.cityscore.ingest.CityScoreIngester")
    def test_ingest_cityscore_data(self, mock_ingester_class):
        """Test ingest_cityscore_data convenience function."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(to_dict=lambda: {"success": True})
        mock_ingester_class.return_value = mock_instance

        result = ingest_cityscore_data(execution_date="2024-01-15")

        assert result["success"]
        mock_instance.run.assert_called_once()
