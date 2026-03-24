"""
Boston Pulse ML - Target Builder Tests.

Tests for ml/models/crime_navigate/target_builder.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestBuildTargets:
    """Tests for build_targets function."""

    def test_returns_dataframe_with_danger_rate(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """build_targets returns DataFrame with danger_rate column."""
        # Create mock processed data
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001", "h3_002"],
                "hour_bucket": [0, 0, 1],
                "occurred_on_date": ["2024-01-10", "2024-01-11", "2024-01-12"],
                "severity_weight": [2.0, 3.0, 1.5],
                "incident_number": ["INC001", "INC002", "INC003"],
                "district": ["A1", "A1", "B2"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(sample_features_df, "2024-01-20", sample_cfg, "test-bucket")

        assert "danger_rate" in df.columns
        assert result.success is True

    def test_danger_rate_computation(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """danger_rate = total_severity / days_active."""
        features_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
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

        # h3_001, bucket 0: 2 incidents on 2 days, total_severity = 5.0
        # danger_rate = 5.0 / 2 = 2.5
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001"],
                "hour_bucket": [0, 0],
                "occurred_on_date": ["2024-01-10", "2024-01-11"],
                "severity_weight": [2.0, 3.0],
                "incident_number": ["INC001", "INC002"],
                "district": ["A1", "A1"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(features_df, "2024-01-20", sample_cfg, "test-bucket")

        h3_001_row = df[(df["h3_index"] == "h3_001") & (df["hour_bucket"] == 0)]
        assert h3_001_row["danger_rate"].values[0] == pytest.approx(2.5, rel=0.01)

    def test_zero_rate_for_cells_without_incidents(
        self, sample_cfg: dict[str, Any], mocker: Any
    ) -> None:
        """Cells with no incidents get danger_rate = 0.0."""
        features_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
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

        # Only h3_001 has incidents
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "occurred_on_date": ["2024-01-10"],
                "severity_weight": [2.0],
                "incident_number": ["INC001"],
                "district": ["A1"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(features_df, "2024-01-20", sample_cfg, "test-bucket")

        # h3_002 should have danger_rate = 0.0
        h3_002_row = df[(df["h3_index"] == "h3_002") & (df["hour_bucket"] == 1)]
        assert h3_002_row["danger_rate"].values[0] == 0.0
        assert result.zero_rate_cells > 0

    def test_includes_district_column(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """build_targets includes district column for bias analysis."""
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
                "occurred_on_date": ["2024-01-10", "2024-01-11"],
                "severity_weight": [2.0, 3.0],
                "incident_number": ["INC001", "INC002"],
                "district": ["A1", "B2"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(sample_features_df, "2024-01-20", sample_cfg, "test-bucket")

        assert "district" in df.columns

    def test_raises_on_empty_processed_data(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """build_targets raises RuntimeError if no processed data found."""
        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = pd.DataFrame()
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        with pytest.raises(RuntimeError, match="No processed data found"):
            build_targets(sample_features_df, "2024-01-20", sample_cfg, "test-bucket")

    def test_result_contains_correct_statistics(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """TargetBuildResult contains correct statistics."""
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
                "occurred_on_date": ["2024-01-10", "2024-01-11"],
                "severity_weight": [2.0, 3.0],
                "incident_number": ["INC001", "INC002"],
                "district": ["A1", "B2"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(sample_features_df, "2024-01-20", sample_cfg, "test-bucket")

        assert result.rows == len(df)
        assert result.h3_cells == df["h3_index"].nunique()
        assert result.mean_danger_rate >= 0
        assert result.zero_rate_cells >= 0

    def test_normalizes_by_days_active(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """danger_rate is normalized by days_active, not incident count."""
        features_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "weighted_score_3d": [1.0],
                "weighted_score_30d": [10.0],
                "weighted_score_90d": [100.0],
                "incident_count_30d": [5],
                "trend_3v10": [0.1],
                "trend_10v30": [0.1],
                "trend_30v90": [0.1],
                "gun_incident_count_30d": [1],
                "high_severity_ratio_30d": [0.1],
                "night_score_ratio": [0.1],
                "evening_score_ratio": [0.1],
                "weekend_score_ratio": [0.1],
                "neighbor_weighted_score_30d": [5.0],
                "neighbor_trend_3v10": [0.1],
                "neighbor_gun_count_30d": [1],
            }
        )

        # 3 incidents on the same day = 1 day_active
        # total_severity = 6.0, days_active = 1
        # danger_rate = 6.0 / 1 = 6.0
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001", "h3_001"],
                "hour_bucket": [0, 0, 0],
                "occurred_on_date": ["2024-01-10", "2024-01-10", "2024-01-10"],
                "severity_weight": [2.0, 2.0, 2.0],
                "incident_number": ["INC001", "INC002", "INC003"],
                "district": ["A1", "A1", "A1"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(features_df, "2024-01-20", sample_cfg, "test-bucket")

        assert df["danger_rate"].values[0] == pytest.approx(6.0, rel=0.01)

    def test_handles_missing_district(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """Cells without district info get 'UNKNOWN'."""
        features_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
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

        # Only h3_001 has incidents with district
        processed_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "occurred_on_date": ["2024-01-10"],
                "severity_weight": [2.0],
                "incident_number": ["INC001"],
                "district": ["A1"],
            }
        )

        mock_loader = MagicMock()
        mock_loader.read_all_partitions.return_value = processed_df
        mocker.patch("models.crime_navigate.target_builder.GCSLoader", return_value=mock_loader)

        from models.crime_navigate.target_builder import build_targets

        df, result = build_targets(features_df, "2024-01-20", sample_cfg, "test-bucket")

        # h3_002 should have district = 'UNKNOWN'
        h3_002_row = df[(df["h3_index"] == "h3_002") & (df["hour_bucket"] == 1)]
        assert h3_002_row["district"].values[0] == "UNKNOWN"
