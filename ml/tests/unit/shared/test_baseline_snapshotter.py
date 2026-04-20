"""
Unit tests for the baseline snapshotter.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestSnapshotTrainingBaseline:
    """Tests for snapshot_training_baseline function."""

    @pytest.fixture
    def sample_training_df(self) -> pd.DataFrame:
        """Sample training DataFrame for tests."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "h3_index": [f"h3_{i:03d}" for i in range(n)],
                "hour_bucket": np.random.choice([0, 1, 2, 3, 4, 5], n),
                "weighted_score_3d": np.random.uniform(0, 10, n),
                "weighted_score_30d": np.random.uniform(0, 20, n),
                "incident_count_30d": np.random.randint(0, 30, n),
                "trend_3v10": np.random.uniform(-1, 1, n),
                "danger_rate": np.random.exponential(0.5, n),
            }
        )

    @pytest.fixture
    def sample_cfg(self) -> dict[str, Any]:
        """Sample config for tests."""
        return {
            "registry": {
                "artifact_bucket": "test-bucket",
                "package": "navigate/crime-risk",
            },
            "features": {
                "input_columns": [
                    "weighted_score_3d",
                    "weighted_score_30d",
                    "incident_count_30d",
                    "trend_3v10",
                    "hour_bucket",
                ],
                "target_column": "danger_rate",
            },
        }

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_creates_sample_and_stats(
        self,
        mock_storage_client: MagicMock,
        sample_training_df: pd.DataFrame,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that snapshot creates both sample parquet and stats JSON."""
        from shared.baseline_snapshotter import snapshot_training_baseline

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        paths = snapshot_training_baseline(
            training_df=sample_training_df,
            feature_cols=sample_cfg["features"]["input_columns"],
            target_col=sample_cfg["features"]["target_column"],
            version="20240115",
            cfg=sample_cfg,
        )

        # Verify paths returned
        assert "sample_uri" in paths
        assert "stats_uri" in paths
        assert "latest_sample_uri" in paths
        assert "20240115" in paths["sample_uri"]
        assert "crime-risk" in paths["sample_uri"]

        # Verify uploads were called
        assert mock_blob.upload_from_filename.called
        assert mock_blob.upload_from_string.called

        # Verify copy to latest was called
        assert mock_bucket.copy_blob.call_count == 2

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_computes_numeric_stats(
        self,
        mock_storage_client: MagicMock,
        sample_training_df: pd.DataFrame,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that numeric feature stats are computed correctly."""
        from shared.baseline_snapshotter import snapshot_training_baseline

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        uploaded_stats: dict[str, Any] = {}

        def capture_stats(data: str, content_type: str) -> None:
            nonlocal uploaded_stats
            if content_type == "application/json":
                uploaded_stats = json.loads(data)

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = capture_stats
        mock_bucket.blob.return_value = mock_blob

        snapshot_training_baseline(
            training_df=sample_training_df,
            feature_cols=sample_cfg["features"]["input_columns"],
            target_col=sample_cfg["features"]["target_column"],
            version="20240115",
            cfg=sample_cfg,
        )

        # Verify stats structure
        assert "feature_stats" in uploaded_stats
        assert "weighted_score_3d" in uploaded_stats["feature_stats"]

        ws_stats = uploaded_stats["feature_stats"]["weighted_score_3d"]
        assert ws_stats["dtype"] == "numeric"
        assert "mean" in ws_stats
        assert "std" in ws_stats
        assert "min" in ws_stats
        assert "max" in ws_stats
        assert "p25" in ws_stats
        assert "p50" in ws_stats
        assert "p75" in ws_stats
        assert "null_pct" in ws_stats

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_computes_categorical_stats(
        self,
        mock_storage_client: MagicMock,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that categorical feature stats are computed correctly."""
        from shared.baseline_snapshotter import snapshot_training_baseline

        # Create df with a categorical column
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "C", "B", "A", "A", "B", "C", "A"],
                "value": range(10),
            }
        )

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        uploaded_stats: dict[str, Any] = {}

        def capture_stats(data: str, content_type: str) -> None:
            nonlocal uploaded_stats
            if content_type == "application/json":
                uploaded_stats = json.loads(data)

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = capture_stats
        mock_bucket.blob.return_value = mock_blob

        snapshot_training_baseline(
            training_df=df,
            feature_cols=["category"],
            target_col="value",
            version="20240115",
            cfg=sample_cfg,
        )

        cat_stats = uploaded_stats["feature_stats"]["category"]
        assert cat_stats["dtype"] == "categorical"
        assert "n_unique" in cat_stats
        assert "top_values" in cat_stats
        assert cat_stats["n_unique"] == 3

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_samples_large_dataframes(
        self,
        mock_storage_client: MagicMock,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that large DataFrames are sampled down."""
        from shared.baseline_snapshotter import BASELINE_SAMPLE_SIZE, snapshot_training_baseline

        # Create large df
        n = BASELINE_SAMPLE_SIZE + 5000
        df = pd.DataFrame(
            {
                "feature": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        uploaded_stats: dict[str, Any] = {}

        def capture_stats(data: str, content_type: str) -> None:
            nonlocal uploaded_stats
            if content_type == "application/json":
                uploaded_stats = json.loads(data)

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = capture_stats
        mock_bucket.blob.return_value = mock_blob

        snapshot_training_baseline(
            training_df=df,
            feature_cols=["feature"],
            target_col="target",
            version="20240115",
            cfg=sample_cfg,
        )

        assert uploaded_stats["n_rows_total"] == n
        assert uploaded_stats["n_rows_sampled"] == BASELINE_SAMPLE_SIZE

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_handles_missing_columns(
        self,
        mock_storage_client: MagicMock,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that missing columns are gracefully skipped."""
        from shared.baseline_snapshotter import snapshot_training_baseline

        df = pd.DataFrame(
            {
                "weighted_score_3d": [1.0, 2.0, 3.0],
                "danger_rate": [0.1, 0.2, 0.3],
            }
        )

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        # Should not raise even though most input_columns are missing
        paths = snapshot_training_baseline(
            training_df=df,
            feature_cols=sample_cfg["features"]["input_columns"],
            target_col=sample_cfg["features"]["target_column"],
            version="20240115",
            cfg=sample_cfg,
        )

        assert paths is not None
        assert "sample_uri" in paths

    @patch("shared.baseline_snapshotter.storage.Client")
    def test_snapshot_includes_version_in_stats(
        self,
        mock_storage_client: MagicMock,
        sample_training_df: pd.DataFrame,
        sample_cfg: dict[str, Any],
    ) -> None:
        """Test that version is included in the stats JSON."""
        from shared.baseline_snapshotter import snapshot_training_baseline

        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        uploaded_stats: dict[str, Any] = {}

        def capture_stats(data: str, content_type: str) -> None:
            nonlocal uploaded_stats
            if content_type == "application/json":
                uploaded_stats = json.loads(data)

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = capture_stats
        mock_bucket.blob.return_value = mock_blob

        snapshot_training_baseline(
            training_df=sample_training_df,
            feature_cols=sample_cfg["features"]["input_columns"],
            target_col=sample_cfg["features"]["target_column"],
            version="20240115",
            cfg=sample_cfg,
        )

        assert uploaded_stats["version"] == "20240115"
