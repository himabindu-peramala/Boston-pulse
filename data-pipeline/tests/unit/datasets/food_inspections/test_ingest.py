"""
Unit tests for FoodInspectionsIngester.

Tests the food inspections data ingestion from Analyze Boston API.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.food_inspections.ingest import FoodInspectionsIngester, ingest_food_inspections_data


class TestFoodInspectionsIngester:
    """Test cases for FoodInspectionsIngester class."""

    @pytest.fixture
    def ingester(self):
        """Create a FoodInspectionsIngester instance."""
        return FoodInspectionsIngester()

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response matching Analyze Boston format."""
        return {
            "success": True,
            "result": {
                "records": [
                    {
                        "_id": 1,
                        "businessname": "Test Cafe",
                        "licenseno": "12345",
                        "result": "Pass",
                        "resultdttm": "2024-01-15T10:00:00",
                        "location": "(42.3456, -71.0789)",
                    },
                    {
                        "_id": 2,
                        "businessname": "Burger Joint",
                        "licenseno": "67890",
                        "result": "Fail",
                        "resultdttm": "2024-01-15T11:00:00",
                        "location": "(42.3123, -71.0567)",
                    },
                ],
                "total": 2,
            },
        }

    def test_get_dataset_name(self, ingester):
        """Test dataset name is correct."""
        assert ingester.get_dataset_name() == "food_inspections"

    def test_get_watermark_field(self, ingester):
        """Test watermark field is correct."""
        assert ingester.get_watermark_field() == "resultdttm"

    def test_get_primary_key(self, ingester):
        """Test primary key is correct."""
        assert ingester.get_primary_key() == "_id"

    def test_get_api_endpoint(self, ingester):
        """Test API endpoint is constructed correctly."""
        endpoint = ingester.get_api_endpoint()
        assert "datastore_search" in endpoint
        assert "data.boston.gov" in endpoint

    @patch("src.datasets.food_inspections.ingest.requests.get")
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
        assert "resultdttm" in df.columns
        mock_get.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.datasets.food_inspections.ingest.FoodInspectionsIngester")
    def test_ingest_food_inspections_data(self, mock_ingester_class):
        """Test ingest_food_inspections_data convenience function."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(to_dict=lambda: {"success": True})
        mock_ingester_class.return_value = mock_instance

        result = ingest_food_inspections_data(execution_date="2024-01-15")

        assert result["success"]
        mock_instance.run.assert_called_once()
