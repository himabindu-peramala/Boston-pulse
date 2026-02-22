"""
Unit tests for CityScorePreprocessor.

Tests the CityScore data preprocessing and validation.
"""

import pandas as pd
import pytest

from src.datasets.cityscore.preprocess import CityScorePreprocessor, preprocess_cityscore_data


class TestCityScorePreprocessor:
    """Test cases for CityScorePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a CityScorePreprocessor instance."""
        return CityScorePreprocessor()

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data matching API format."""
        return pd.DataFrame(
            {
                "_id": [1, 2, 3, 1],  # Duplicate
                "score_calculated_ts": [
                    "2024-01-15T00:00:00",
                    "2024-01-15T00:00:00",
                    "2024-01-16T00:00:00",
                    "2024-01-15T00:00:00",
                ],
                "metric_name": [
                    "Trash On-Time %",
                    "Pothole On-Time %",
                    "Trash On-Time %",
                    "Trash On-Time %",
                ],
                "day_score": [0.85, 0.92, 0.88, 0.85],
                "target": [0.80, 0.90, 0.80, 0.80],
            }
        )

    def test_get_dataset_name(self, preprocessor):
        """Test dataset name is correct."""
        assert preprocessor.get_dataset_name() == "cityscore"

    def test_run_success(self, preprocessor, sample_raw_data):
        """Test successful preprocessing run."""
        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        assert result.success
        df = preprocessor.get_data()
        assert len(df) == 3  # Duplicate dropped
        assert "timestamp" in df.columns
        assert "year" in df.columns

    def test_datetime_parsing(self, preprocessor, sample_raw_data):
        """Test that dates are parsed correctly."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["year"].iloc[0] == 2024

    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = preprocessor.run(df, execution_date="2024-01-15")
        assert result.success
        assert len(preprocessor.get_data()) == 0

    def test_preprocess_cityscore_data_convenience(self, sample_raw_data):
        """Test convenience function."""
        result = preprocess_cityscore_data(sample_raw_data, execution_date="2024-01-15")
        assert result["success"]
        assert result["rows_output"] == 3
