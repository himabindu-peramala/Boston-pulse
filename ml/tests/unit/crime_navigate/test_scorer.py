"""
Boston Pulse ML - Scorer Tests.

Tests for ml/models/crime_navigate/scorer.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestScoreAllCells:
    """Tests for score_all_cells function."""

    def test_returns_dataframe_and_result(
        self,
        sample_cfg: dict[str, Any],
        sample_features_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """score_all_cells returns a DataFrame and ScoringResult."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.uniform(0, 5, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        assert isinstance(df, pd.DataFrame)
        assert result.success is True
        assert result.rows_scored > 0

    def test_output_has_required_columns(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """score_all_cells output has all required columns."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.uniform(0, 5, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        required_cols = [
            "h3_index",
            "hour_bucket",
            "predicted_danger",
            "risk_score",
            "risk_tier",
            "model_version",
            "scored_at",
        ]
        for col in required_cols:
            assert col in df.columns

    def test_risk_scores_in_valid_range(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """risk_score values are in [0, 100] range."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.uniform(0, 5, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        assert df["risk_score"].min() >= 0
        assert df["risk_score"].max() <= 100

    def test_tier_assignment_correct(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """risk_tier is correctly assigned based on score."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        # Create predictions that will span all tiers
        mock_model.predict.return_value = np.linspace(0, 10, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        # Check tier assignment logic
        for _, row in df.iterrows():
            score = row["risk_score"]
            tier = row["risk_tier"]

            if score >= 66:
                assert tier == "HIGH"
            elif score >= 33:
                assert tier == "MEDIUM"
            else:
                assert tier == "LOW"

    def test_deduplicates_by_cell_bucket(self, sample_cfg: dict[str, Any], mocker: Any) -> None:
        """score_all_cells handles duplicate cell-bucket pairs."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        # Create features with duplicates
        features_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_001", "h3_002"],
                "hour_bucket": [0, 0, 1],  # Duplicate h3_001 + bucket 0
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

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

        df, result = score_all_cells(
            mock_model, features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        # Should have 3 rows (no deduplication in scorer — that's feature_loader's job)
        assert len(df) == 3

    def test_writes_to_gcs(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """score_all_cells writes output to GCS."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.uniform(0, 5, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        mock_loader.write_parquet.assert_called_once()
        assert result.output_gcs_path is not None

    def test_includes_model_version(
        self, sample_cfg: dict[str, Any], sample_features_df: pd.DataFrame, mocker: Any
    ) -> None:
        """score_all_cells includes model_version in output."""
        from models.crime_navigate.scorer import score_all_cells

        mock_loader = MagicMock()
        mock_loader.write_parquet.return_value = "gs://test-bucket/scores/scores.parquet"
        mocker.patch("models.crime_navigate.scorer.GCSLoader", return_value=mock_loader)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.uniform(0, 5, len(sample_features_df))

        df, result = score_all_cells(
            mock_model, sample_features_df, "2024-01-15", sample_cfg, "test-bucket", "20240115"
        )

        assert (df["model_version"] == "20240115").all()
        assert result.model_version == "20240115"


class TestScalePerBucket:
    """Tests for _scale_per_bucket function."""

    def test_scales_independently_per_bucket(self, mocker: Any) -> None:
        """_scale_per_bucket scales each bucket independently."""
        from models.crime_navigate.scorer import _scale_per_bucket

        df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002", "h3_003", "h3_004"],
                "hour_bucket": [0, 0, 1, 1],
                "predicted_danger": [1.0, 10.0, 5.0, 15.0],
            }
        )

        scaled = _scale_per_bucket(df)

        # Bucket 0: min=1, max=10 → h3_001=0, h3_002=100
        # Bucket 1: min=5, max=15 → h3_003=0, h3_004=100
        assert scaled.iloc[0] == pytest.approx(0.0, abs=0.1)
        assert scaled.iloc[1] == pytest.approx(100.0, abs=0.1)
        assert scaled.iloc[2] == pytest.approx(0.0, abs=0.1)
        assert scaled.iloc[3] == pytest.approx(100.0, abs=0.1)

    def test_handles_constant_values_in_bucket(self, mocker: Any) -> None:
        """_scale_per_bucket handles buckets where all values are the same."""
        from models.crime_navigate.scorer import _scale_per_bucket

        df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 0],
                "predicted_danger": [5.0, 5.0],  # Same value
            }
        )

        scaled = _scale_per_bucket(df)

        # When all values are the same, should return 50.0
        assert scaled.iloc[0] == pytest.approx(50.0, abs=0.1)
        assert scaled.iloc[1] == pytest.approx(50.0, abs=0.1)


class TestGetScoreStatistics:
    """Tests for get_score_statistics function."""

    def test_returns_correct_statistics(self) -> None:
        """get_score_statistics returns correct statistics."""
        from models.crime_navigate.scorer import get_score_statistics

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002", "h3_003"],
                "hour_bucket": [0, 1, 2],
                "predicted_danger": [1.0, 2.0, 3.0],
                "risk_score": [20.0, 50.0, 80.0],
                "risk_tier": ["LOW", "MEDIUM", "HIGH"],
            }
        )

        stats = get_score_statistics(scores_df)

        assert stats["total_cells"] == 3
        assert stats["unique_h3"] == 3
        assert stats["hour_buckets"] == [0, 1, 2]
        assert stats["risk_distribution"]["LOW"] == 1
        assert stats["risk_distribution"]["MEDIUM"] == 1
        assert stats["risk_distribution"]["HIGH"] == 1
        assert stats["mean_risk_score"] == pytest.approx(50.0, abs=0.1)
