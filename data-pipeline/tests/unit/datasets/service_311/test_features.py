"""
Unit tests for Service311FeatureBuilder.

Tests the 311 feature generation.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.datasets.service_311.features import Service311FeatureBuilder, build_311_features


class TestService311FeatureBuilder:
    """Test cases for Service311FeatureBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a Service311FeatureBuilder instance."""
        return Service311FeatureBuilder()

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed data matching Service311Preprocessor output."""
        return pd.DataFrame(
            {
                "case_id": ["1", "2", "3"],
                "open_date": [
                    datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 0, tzinfo=UTC),  # Older record
                ],
                "neighborhood": ["South End", "South End", "South End"],
                "lat": [42.3456, 42.3457, 42.3458],
                "long": [-71.0789, -71.0788, -71.0787],
                "on_time": ["ON TIME", "OVERDUE", "ON TIME"],
            }
        )

    def test_get_dataset_name(self, builder):
        """Test dataset name is correct."""
        assert builder.get_dataset_name() == "311"

    def test_build_features_success(self, builder, sample_processed_data):
        """Test successful feature building."""
        features_df = builder.build_features(sample_processed_data)

        assert not features_df.empty
        assert "grid_cell" in features_df.columns
        assert "request_count_7d" in features_df.columns
        assert "overdue_ratio_30d" in features_df.columns

    def test_grid_cell_aggregation(self, builder):
        """Test that records in the same grid cell are aggregated."""
        df = pd.DataFrame(
            {
                "case_id": ["1", "2"],
                "open_date": [datetime(2024, 1, 15), datetime(2024, 1, 15)],
                "neighborhood": ["Downtown", "Downtown"],
                "lat": [42.3501, 42.3502],  # Same grid cell (0.001)
                "long": [-71.0601, -71.0602],
                "on_time": ["ON TIME", "ON TIME"],
            }
        )

        features_df = builder.build_features(df)
        assert len(features_df) == 1
        assert features_df["request_count_7d"].iloc[0] == 2

    def test_overdue_ratio_calculation(self, builder):
        """Test overdue ratio calculation."""
        df = pd.DataFrame(
            {
                "case_id": ["1", "2"],
                "open_date": [datetime(2024, 1, 15), datetime(2024, 1, 15)],
                "neighborhood": ["Downtown", "Downtown"],
                "lat": [42.3501, 42.3501],
                "long": [-71.0601, -71.0601],
                "on_time": ["ON TIME", "OVERDUE"],
            }
        )

        features_df = builder.build_features(df)
        assert features_df["overdue_ratio_30d"].iloc[0] == 0.5

    def test_empty_dataframe(self, builder):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        features_df = builder.build_features(df)
        assert features_df.empty

    def test_build_311_features_convenience(self, sample_processed_data):
        """Test convenience function."""
        result = build_311_features(sample_processed_data, execution_date="2024-01-15")
        assert result["success"]
        assert "rows_output" in result
