"""
Unit tests for Service311Preprocessor.

Tests the 311 data preprocessing and validation.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.datasets.service_311.preprocess import Service311Preprocessor, preprocess_311_data


class TestService311Preprocessor:
    """Test cases for Service311Preprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a Service311Preprocessor instance."""
        return Service311Preprocessor()

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data matching API format."""
        return pd.DataFrame(
            {
                "case_id": ["101", "102", "101"], # Duplicate
                "open_date": ["2024-01-15T14:30:00", "2024-01-15T16:45:00", "2024-01-15T14:30:00"],
                "close_date": ["2024-01-16T10:00:00", None, "2024-01-16T10:00:00"],
                "case_topic": ["Sanitation", "Highway", "Sanitation"],
                "service_name": ["Trash Pickup", "Pothole", "Trash Pickup"],
                "assigned_department": ["PWD", "PWD", "PWD"],
                "case_status": ["Closed", "Open", "Closed"],
                "neighborhood": ["South End", "Dorchester", "South End"],
                "latitude": [42.3456, 42.3123, 42.3456],
                "longitude": [-71.0789, -71.0567, -71.0789],
                "on_time": ["ON TIME", "OVERDUE", "ON TIME"],
            }
        )

    def test_get_dataset_name(self, preprocessor):
        """Test dataset name is correct."""
        assert preprocessor.get_dataset_name() == "311"

    def test_run_success(self, preprocessor, sample_raw_data):
        """Test successful preprocessing run."""
        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        assert result.success
        df = preprocessor.get_data()
        assert len(df) == 2  # One duplicate dropped
        assert "lat" in df.columns
        assert "year" in df.columns

    def test_coordinate_standardization(self, preprocessor, sample_raw_data):
        """Test that coordinates are renamed and typed correctly."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()
        
        assert "lat" in df.columns
        assert "long" in df.columns
        assert df["lat"].dtype == "float64"

    def test_datetime_parsing(self, preprocessor, sample_raw_data):
        """Test that dates are parsed correctly."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()
        
        assert pd.api.types.is_datetime64_any_dtype(df["open_date"])
        assert df["year"].iloc[0] == 2024

    def test_invalid_coordinates(self, preprocessor):
        """Test filtering of invalid coordinates."""
        df = pd.DataFrame({
            "case_id": ["1", "2"],
            "open_date": ["2024-01-15", "2024-01-15"],
            "neighborhood": ["A", "B"],
            "latitude": [42.3, 0.0],  # 0.0 is out of bounds for Boston
            "longitude": [-71.0, 0.0],
            "case_topic": ["T1", "T2"]
        })
        
        preprocessor.run(df, execution_date="2024-01-15")
        processed_df = preprocessor.get_data()
        
        # Row 2 should have NaN for lat/long
        assert np.isnan(processed_df.loc[processed_df["case_id"] == "2", "lat"].iloc[0])

    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = preprocessor.run(df, execution_date="2024-01-15")
        assert result.success
        assert len(preprocessor.get_data()) == 0


def test_preprocess_311_data_convenience(sample_raw_data):
    """Test convenience function."""
    result = preprocess_311_data(sample_raw_data, execution_date="2024-01-15")
    assert result["success"]
    assert result["rows_processed"] == 2
