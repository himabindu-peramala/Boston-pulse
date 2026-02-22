"""
Unit tests for CityScoreFeatureBuilder.

Tests the CityScore feature generation.
"""

from datetime import datetime

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
                "score_date": [
                    datetime(2024, 1, 15),
                    datetime(2024, 1, 14),
                    datetime(2024, 1, 13),
                ],
                "metric": ["Trash On-Time %", "Trash On-Time %", "Trash On-Time %"],
                "score": [0.85, 0.80, 0.75],
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
        assert "metric" in features_df.columns
        assert "avg_score_7d" in features_df.columns
        assert "score_vs_target" in features_df.columns

    def test_score_vs_target_calculation(self, builder):
        """Test score_vs_target calculation."""
        df = pd.DataFrame({
            "id": [1],
            "score_date": [datetime(2024, 1, 15)],
            "metric": ["Test Metric"],
            "score": [0.90],
            "target": [0.80]
        })
        
        features_df = builder.build_features(df)
        assert features_df["score_vs_target"].iloc[0] == 1.125 # 0.9 / 0.8

    def test_empty_dataframe(self, builder):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        features_df = builder.build_features(df)
        assert features_df.empty


def test_build_cityscore_features_convenience(sample_processed_data):
    """Test convenience function."""
    result = build_cityscore_features(sample_processed_data, execution_date="2024-01-15")
    assert result["success"]
    assert result["rows_processed"] > 0
