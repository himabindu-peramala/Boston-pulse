"""
Boston Pulse ML - Feature Loader Tests.

Tests for ml/models/crime_navigate/feature_loader.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestLoadFeatures:
    """Tests for load_features function."""

    def test_returns_dataframe_and_result(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """load_features returns a DataFrame and FeatureLoadResult."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = sample_features_df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        df, result = load_features("2024-01-15", sample_cfg, "test-bucket")

        assert isinstance(df, pd.DataFrame)
        assert result.success is True
        assert result.rows > 0
        assert result.h3_cells > 0

    def test_deduplicates_to_latest_per_cell_bucket(
        self, sample_cfg: dict[str, Any], mocker: Any
    ) -> None:
        """load_features keeps only the latest row per (h3_index, hour_bucket)."""
        # Create data with duplicates
        df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001", "h3_002"],
                "hour_bucket": [0, 0, 1],
                "computed_date": ["2024-01-10", "2024-01-15", "2024-01-12"],
                "weighted_score_3d": [1.0, 2.0, 3.0],
                "weighted_score_30d": [10.0, 20.0, 30.0],
                "weighted_score_90d": [100.0, 200.0, 300.0],
                "incident_count_30d": [5, 10, 15],
                "trend_3v10": [0.1, 0.2, 0.3],
                "trend_10v30": [0.1, 0.2, 0.3],
                "trend_30v90": [0.1, 0.2, 0.3],
                "gun_incident_count_30d": [1, 2, 3],
                "high_severity_ratio_30d": [0.1, 0.2, 0.3],
                "night_score_ratio": [0.1, 0.2, 0.3],
                "evening_score_ratio": [0.1, 0.2, 0.3],
                "weekend_score_ratio": [0.1, 0.2, 0.3],
                "neighbor_weighted_score_30d": [5.0, 10.0, 15.0],
                "neighbor_trend_3v10": [0.1, 0.2, 0.3],
                "neighbor_gun_count_30d": [1, 2, 3],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        result_df, result = load_features("2024-01-20", sample_cfg, "test-bucket")

        # Should have 2 rows (one per unique h3_index + hour_bucket)
        assert len(result_df) == 2
        # h3_001 + bucket 0 should have the latest value (2024-01-15)
        h3_001_row = result_df[
            (result_df["h3_index"] == "h3_001") & (result_df["hour_bucket"] == 0)
        ]
        assert h3_001_row["weighted_score_3d"].values[0] == 2.0

    def test_raises_on_missing_columns(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """load_features raises ValueError if required columns are missing."""
        df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "computed_date": ["2024-01-15"],
                # Missing most feature columns
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        with pytest.raises(ValueError, match="Feature columns missing"):
            load_features("2024-01-15", sample_cfg, "test-bucket")

    def test_raises_on_empty_partitions(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """load_features raises RuntimeError if no partitions found."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = pd.DataFrame()
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        with pytest.raises(RuntimeError, match="No feature partitions found"):
            load_features("2024-01-15", sample_cfg, "test-bucket")

    def test_uses_lookback_days_from_config(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """load_features uses active_cell_lookback_days from config."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = sample_features_df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        load_features("2024-01-15", sample_cfg, "test-bucket")

        # Check that read_all_partitions was called with correct date range
        call_args = mock_loader.read_all_partitions.call_args
        assert "after" in call_args.kwargs
        assert "before" in call_args.kwargs

    def test_result_contains_correct_metadata(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """FeatureLoadResult contains correct metadata."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = sample_features_df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        df, result = load_features("2024-01-15", sample_cfg, "test-bucket")

        assert result.rows == len(df)
        assert result.h3_cells == df["h3_index"].nunique()
        assert "h3_index" in result.columns
        assert "hour_bucket" in result.columns

    def test_handles_date_column_fallback(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """load_features falls back to 'date' column if 'computed_date' missing."""
        df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001"],
                "hour_bucket": [0, 0],
                "date": ["2024-01-10", "2024-01-15"],  # 'date' instead of 'computed_date'
                "weighted_score_3d": [1.0, 2.0],
                "weighted_score_30d": [10.0, 20.0],
                "weighted_score_90d": [100.0, 200.0],
                "incident_count_30d": [5, 10],
                "trend_3v10": [0.1, 0.2],
                "trend_10v30": [0.1, 0.2],
                "trend_30v90": [0.1, 0.2],
                "gun_incident_count_30d": [1, 2],
                "high_severity_ratio_30d": [0.1, 0.2],
                "night_score_ratio": [0.1, 0.2],
                "evening_score_ratio": [0.1, 0.2],
                "weekend_score_ratio": [0.1, 0.2],
                "neighbor_weighted_score_30d": [5.0, 10.0],
                "neighbor_trend_3v10": [0.1, 0.2],
                "neighbor_gun_count_30d": [1, 2],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = df
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        result_df, result = load_features("2024-01-20", sample_cfg, "test-bucket")

        # Should still deduplicate correctly using 'date' column
        assert len(result_df) == 1
        assert result_df["weighted_score_3d"].values[0] == 2.0  # Latest

    def test_returns_failure_result_on_file_not_found(
        self, sample_cfg: dict[str, Any], mocker: Any
    ) -> None:
        """load_features returns failure result when FileNotFoundError raised."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.side_effect = FileNotFoundError("No partitions")
        mocker.patch("models.crime_navigate.feature_loader.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.feature_loader import load_features

        df, result = load_features("2024-01-15", sample_cfg, "test-bucket")

        assert df.empty
        assert result.success is False
        assert result.error is not None
