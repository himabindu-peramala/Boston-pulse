"""
Unit tests for FoodInspectionsPreprocessor.

Tests the food inspections data preprocessing and validation.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.datasets.food_inspections.preprocess import FoodInspectionsPreprocessor, preprocess_food_inspections_data


class TestFoodInspectionsPreprocessor:
    """Test cases for FoodInspectionsPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a FoodInspectionsPreprocessor instance."""
        return FoodInspectionsPreprocessor()

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data matching API format."""
        return pd.DataFrame(
            {
                "_id": [1, 2, 3, 1], # Duplicate
                "businessname": ["Cafe A", "Cafe B", "Cafe C", "Cafe A"],
                "licenseno": ["L1", "L2", "L3", "L1"],
                "result": ["Pass", "Fail", "Pass", "Pass"],
                "resultdttm": ["2024-01-15T10:00:00", "2024-01-15T11:00:00", "2024-01-16T12:00:00", "2024-01-15T10:00:00"],
                "location": ["(42.3456, -71.0789)", "(42.3123, -71.0567)", "(42.3789, -71.0123)", "(42.3456, -71.0789)"],
            }
        )

    def test_get_dataset_name(self, preprocessor):
        """Test dataset name is correct."""
        assert preprocessor.get_dataset_name() == "food_inspections"

    def test_run_success(self, preprocessor, sample_raw_data):
        """Test successful preprocessing run."""
        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        assert result.success
        df = preprocessor.get_data()
        assert len(df) == 3  # Duplicate dropped
        assert "lat" in df.columns
        assert "year" in df.columns

    def test_coordinate_parsing(self, preprocessor, sample_raw_data):
        """Test that coordinates are parsed correctly from location string."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()
        
        assert "lat" in df.columns
        assert "long" in df.columns
        assert df["lat"].iloc[0] == 42.3456
        assert df["long"].iloc[0] == -71.0789

    def test_datetime_parsing(self, preprocessor, sample_raw_data):
        """Test that dates are parsed correctly."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()
        
        assert pd.api.types.is_datetime64_any_dtype(df["resultdttm"])
        assert df["year"].iloc[0] == 2024

    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = preprocessor.run(df, execution_date="2024-01-15")
        assert result.success
        assert len(preprocessor.get_data()) == 0


def test_preprocess_food_inspections_data_convenience(sample_raw_data):
    """Test convenience function."""
    result = preprocess_food_inspections_data(sample_raw_data, execution_date="2024-01-15")
    assert result["success"]
    assert result["rows_processed"] == 3
