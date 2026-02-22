"""
Unit tests for CityScoreFeatureBuilder.

Tests the CityScore feature generation.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.datasets.cityscore.features import CityScoreFeatureBuilder, build_cityscore_features


class TestCityScoreFeatureBuilder:
    """Test cases for CityScoreFeatureBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a CityScoreFeatureBuilder instance."""
        return CityScoreFeatureBuilder()

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed data matching CityScorePreprocessor output."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2024, 1, 15, tzinfo=UTC),
                    datetime(2024, 1, 14, tzinfo=UTC),
                    datetime(2024, 1, 13, tzinfo=UTC),
                ],
                "date": [
                    datetime(2024, 1, 15, tzinfo=UTC).date(),
                    datetime(2024, 1, 14, tzinfo=UTC).date(),
                    datetime(2024, 1, 13, tzinfo=UTC).date(),
                ],
                "metric_name": ["Trash On-Time %", "Trash On-Time %", "Trash On-Time %"],
                "day_score": [0.85, 0.80, 0.75],
                "target": [0.80, 0.80, 0.80],
            }
        )

    def test_get_dataset_name(self, builder):
        """Test dataset name is correct."""
        assert builder.get_dataset_name() == "cityscore"

    def test_build_features_success(self, builder, sample_processed_data):
        """Test successful feature building."""
        features_df = builder.build_features(sample_processed_data)

        assert not features_df.empty
        assert "date" in features_df.columns
        assert "avg_day_score" in features_df.columns

    def test_score_vs_target_calculation(self, builder):
        """Test that metric-specific columns are created."""
        df = pd.DataFrame(
            {
                "id": [1],
                "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                "date": [datetime(2024, 1, 15, tzinfo=UTC).date()],
                "metric_name": ["Test Metric"],
                "day_score": [0.90],
                "target": [0.80],
            }
        )

        features_df = builder.build_features(df)
        assert "test_metric" in features_df.columns
        assert features_df["test_metric"].iloc[0] == 0.90

    def test_empty_dataframe(self, builder):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        features_df = builder.build_features(df)
        assert features_df.empty

    def test_build_cityscore_features_convenience(self, sample_processed_data):
        """Test convenience function."""
        result = build_cityscore_features(sample_processed_data, execution_date="2024-01-15")
        assert result["success"]
        assert "rows_output" in result
