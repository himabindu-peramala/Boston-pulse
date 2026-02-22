"""
Unit tests for CrimeIngester.

Tests the crime data ingestion from Analyze Boston API.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.crime.ingest import CrimeIngester, get_crime_sample, ingest_crime_data


class TestCrimeIngester:
    """Test cases for CrimeIngester class."""

    @pytest.fixture
    def ingester(self):
        """Create a CrimeIngester instance."""
        return CrimeIngester()

    @pytest.fixture
    def mock_successful_response(self, sample_api_response):
        """Reusable mock for a successful API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        return mock_response

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response matching Analyze Boston format."""
        return {
            "success": True,
            "result": {
                "records": [
                    {
                        "INCIDENT_NUMBER": "I2024001",
                        "OFFENSE_CODE": "3115",
                        "OFFENSE_CODE_GROUP": "Investigate Property",
                        "OFFENSE_DESCRIPTION": "INVESTIGATE PROPERTY",
                        "DISTRICT": "A1",
                        "REPORTING_AREA": "101",
                        "SHOOTING": "N",
                        "OCCURRED_ON_DATE": "2024-01-15T14:30:00",
                        "YEAR": 2024,
                        "MONTH": 1,
                        "DAY_OF_WEEK": "Monday",
                        "HOUR": 14,
                        "UCR_PART": "Part Three",
                        "STREET": "MAIN ST",
                        "Lat": 42.3601,
                        "Long": -71.0589,
                        "Location": "(42.3601, -71.0589)",
                    },
                    {
                        "INCIDENT_NUMBER": "I2024002",
                        "OFFENSE_CODE": "619",
                        "OFFENSE_CODE_GROUP": "Larceny",
                        "OFFENSE_DESCRIPTION": "LARCENY THEFT FROM MV - NON-ACCESSORY",
                        "DISTRICT": "B2",
                        "REPORTING_AREA": "202",
                        "SHOOTING": "N",
                        "OCCURRED_ON_DATE": "2024-01-15T16:45:00",
                        "YEAR": 2024,
                        "MONTH": 1,
                        "DAY_OF_WEEK": "Monday",
                        "HOUR": 16,
                        "UCR_PART": "Part One",
                        "STREET": "WASHINGTON ST",
                        "Lat": 42.3489,
                        "Long": -71.0765,
                        "Location": "(42.3489, -71.0765)",
                    },
                ],
                "total": 2,
            },
        }

    def test_get_dataset_name(self, ingester):
        """Test dataset name is correct."""
        assert ingester.get_dataset_name() == "crime"

    def test_get_watermark_field(self, ingester):
        """Test watermark field is correct."""
        assert ingester.get_watermark_field() == "OCCURRED_ON_DATE"

    def test_get_primary_key(self, ingester):
        """Test primary key is correct."""
        assert ingester.get_primary_key() == "INCIDENT_NUMBER"

    def test_get_api_endpoint(self, ingester):
        """Test API endpoint is constructed correctly."""
        endpoint = ingester.get_api_endpoint()
        assert "datastore_search" in endpoint
        assert "data.boston.gov" in endpoint

    @patch("src.datasets.crime.ingest.requests.get")
    def test_fetch_data_success(self, mock_get, ingester, sample_api_response):
        """Test successful data fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = ingester.fetch_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "INCIDENT_NUMBER" in df.columns
        assert "OCCURRED_ON_DATE" in df.columns
        mock_get.assert_called_once()

    @patch("src.datasets.crime.ingest.requests.get")
    def test_fetch_data_with_watermark(self, mock_get, ingester, sample_api_response):
        """Test data fetching with watermark filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        since = datetime(2024, 1, 1)
        df = ingester.fetch_data(since=since)

        assert isinstance(df, pd.DataFrame)
        call_args = mock_get.call_args
        assert "params" in call_args.kwargs or len(call_args.args) > 1

    @patch("src.datasets.crime.ingest.requests.get")
    def test_fetch_data_api_error(self, mock_get, ingester):
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "error": {"message": "Resource not found"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="API returned error"):
            ingester.fetch_data()

    @patch("src.datasets.crime.ingest.requests.get")
    def test_run_success(self, mock_get, ingester, sample_api_response):
        """Test successful ingestion run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = ingester.run(execution_date="2024-01-15")

        assert result.success
        assert result.dataset == "crime"
        assert result.rows_fetched == 2
        assert result.execution_date == "2024-01-15"

    @patch("src.datasets.crime.ingest.requests.get")
    def test_run_with_watermark(self, mock_get, ingester, sample_api_response):
        """Test ingestion run with explicit watermark."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        watermark_start = datetime(2024, 1, 1)
        result = ingester.run(
            execution_date="2024-01-15",
            watermark_start=watermark_start,
        )

        assert result.success
        assert result.watermark_start == watermark_start

    @patch("src.datasets.crime.ingest.requests.get")
    def test_run_failure(self, mock_get, ingester):
        """Test handling of failed ingestion."""
        mock_get.side_effect = Exception("Network error")

        result = ingester.run(execution_date="2024-01-15")

        assert not result.success
        assert "Network error" in result.error_message

    @patch("src.datasets.crime.ingest.requests.get")
    def test_get_data_after_run(self, mock_get, ingester, sample_api_response):
        """Test getting data after successful run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        ingester.run(execution_date="2024-01-15")
        df = ingester.get_data()

        assert df is not None
        assert len(df) == 2

    def test_validate_schema_valid(self, ingester):
        """Test schema validation with valid data."""
        df = pd.DataFrame(
            {
                "INCIDENT_NUMBER": ["I001", "I002"],
                "OCCURRED_ON_DATE": ["2024-01-15", "2024-01-16"],
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

    def test_validate_schema_missing_watermark(self, ingester):
        """Test schema validation with missing watermark field."""
        df = pd.DataFrame({"INCIDENT_NUMBER": ["I001", "I002"]})

        is_valid, errors = ingester.validate_schema(df)

        assert not is_valid
        assert any("Watermark" in e for e in errors)

    def test_validate_schema_empty_df(self, ingester):
        """Test schema validation with empty DataFrame."""
        df = pd.DataFrame()

        is_valid, errors = ingester.validate_schema(df)

        assert not is_valid
        assert any("empty" in e.lower() for e in errors)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.datasets.crime.ingest.CrimeIngester")
    def test_ingest_crime_data(self, mock_ingester_class):
        """Test ingest_crime_data convenience function."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(to_dict=lambda: {"success": True})
        mock_ingester_class.return_value = mock_instance

        result = ingest_crime_data(execution_date="2024-01-15")

        assert result["success"]
        mock_instance.run.assert_called_once()

    @patch("src.datasets.crime.ingest.CrimeIngester")
    def test_get_crime_sample(self, mock_ingester_class):
        """Test get_crime_sample convenience function."""
        mock_instance = MagicMock()
        mock_instance.fetch_sample_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_ingester_class.return_value = mock_instance

        df = get_crime_sample(n=100)

        assert isinstance(df, pd.DataFrame)
        mock_instance.fetch_sample_data.assert_called_once_with(100)
