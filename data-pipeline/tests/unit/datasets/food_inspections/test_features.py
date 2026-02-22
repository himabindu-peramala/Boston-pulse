"""
Unit tests for FoodInspectionsFeatureBuilder.

Tests the food inspections feature generation.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.datasets.food_inspections.features import (
    FoodInspectionsFeatureBuilder,
    build_food_inspections_features,
)


class TestFoodInspectionsFeatureBuilder:
    """Test cases for FoodInspectionsFeatureBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a FoodInspectionsFeatureBuilder instance."""
        return FoodInspectionsFeatureBuilder()

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed data matching FoodInspectionsPreprocessor output."""
        base_date = datetime(2024, 1, 15, tzinfo=UTC)
        return pd.DataFrame(
            {
                "_id": [1, 2, 3],
                "resultdttm": [
                    base_date,
                    base_date,
                    base_date,
                ],
                "result": ["Pass", "Fail", "Pass"],
                "lat": [42.3456, 42.3457, 42.3456],  # 1 and 3 in same cell
                "long": [-71.0789, -71.0788, -71.0789],
            }
        )

    def test_get_dataset_name(self, builder):
        """Test dataset name is correct."""
        assert builder.get_dataset_name() == "food_inspections"

    def test_build_features_success(self, builder, sample_processed_data):
        """Test successful feature building."""
        features_df = builder.build_features(sample_processed_data)

        assert not features_df.empty
        assert "grid_cell" in features_df.columns
        assert "inspection_count_180d" in features_df.columns
        assert "failure_count_180d" in features_df.columns
        assert "failure_ratio_180d" in features_df.columns

    def test_grid_cell_aggregation(self, builder):
        """Test that records in the same grid cell are aggregated."""
        df = pd.DataFrame(
            {
                "_id": [1, 2],
                "resultdttm": [datetime(2024, 1, 15), datetime(2024, 1, 15)],
                "result": ["Pass", "Pass"],
                "lat": [42.3501, 42.3502],  # Same grid cell (0.001)
                "long": [-71.0601, -71.0602],
            }
        )

        features_df = builder.build_features(df)
        assert len(features_df) == 1
        assert features_df["inspection_count_180d"].iloc[0] == 2

    def test_failure_ratio_calculation(self, builder):
        """Test failure ratio calculation."""
        df = pd.DataFrame(
            {
                "_id": [1, 2],
                "resultdttm": [datetime(2024, 1, 15), datetime(2024, 1, 15)],
                "result": ["Pass", "Fail"],
                "lat": [42.3501, 42.3501],
                "long": [-71.0601, -71.0601],
            }
        )

        features_df = builder.build_features(df)
        assert features_df["failure_ratio_180d"].iloc[0] == 0.5

    def test_empty_dataframe(self, builder):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        features_df = builder.build_features(df)
        assert features_df.empty

    def test_build_food_inspections_features_convenience(self, sample_processed_data):
        """Test convenience function."""
        result = build_food_inspections_features(sample_processed_data, execution_date="2024-01-15")
        assert result["success"]
        assert "rows_output" in result
