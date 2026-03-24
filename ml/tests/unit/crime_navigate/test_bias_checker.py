"""
Boston Pulse ML - Bias Checker Tests.

Tests for ml/models/crime_navigate/bias_checker.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestCheckBias:
    """Tests for check_bias function."""

    def test_passes_when_slices_are_similar(
        self,
        sample_cfg: dict[str, Any],
        sample_training_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """check_bias passes when slice RMSEs are similar to overall."""
        from models.crime_navigate.bias_checker import check_bias
        from models.crime_navigate.trainer import random_split, train_model

        # Mock GCS loader
        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, _ = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        # Set high thresholds so it passes
        sample_cfg["bias"]["max_slice_rmse_deviation"] = 0.9
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 10.0

        result = check_bias(model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

        assert result.passed is True
        assert result.overall_rmse > 0

    def test_fails_when_slice_deviation_exceeds_threshold(
        self, sample_cfg: dict[str, Any], mock_mlflow: Any, mocker: Any
    ) -> None:
        """check_bias raises BiasGateError when slice RMSE deviates too much."""
        from models.crime_navigate.bias_checker import BiasGateError, check_bias

        # Mock GCS loader
        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        # Create a mock model that predicts very differently for different districts
        mock_model = MagicMock()

        # Create val_df with distinct districts
        val_df = pd.DataFrame(
            {
                "h3_index": [f"h3_{i:03d}" for i in range(100)],
                "hour_bucket": [i % 6 for i in range(100)],
                "weighted_score_3d": np.random.uniform(0, 10, 100),
                "weighted_score_30d": np.random.uniform(0, 20, 100),
                "weighted_score_90d": np.random.uniform(0, 50, 100),
                "incident_count_30d": np.random.randint(0, 30, 100),
                "trend_3v10": np.random.uniform(-1, 1, 100),
                "trend_10v30": np.random.uniform(-1, 1, 100),
                "trend_30v90": np.random.uniform(-1, 1, 100),
                "gun_incident_count_30d": np.random.randint(0, 5, 100),
                "high_severity_ratio_30d": np.random.uniform(0, 1, 100),
                "night_score_ratio": np.random.uniform(0, 1, 100),
                "evening_score_ratio": np.random.uniform(0, 1, 100),
                "weekend_score_ratio": np.random.uniform(0, 1, 100),
                "neighbor_weighted_score_30d": np.random.uniform(0, 15, 100),
                "neighbor_trend_3v10": np.random.uniform(-1, 1, 100),
                "neighbor_gun_count_30d": np.random.randint(0, 10, 100),
                "danger_rate": np.concatenate([np.ones(50) * 1.0, np.ones(50) * 10.0]),
                "district": ["A1"] * 50 + ["B2"] * 50,
            }
        )

        # Model predicts 1.0 for all — will have very different RMSE for A1 vs B2
        mock_model.predict.return_value = np.ones(100) * 1.0

        # Set low thresholds so it fails
        sample_cfg["bias"]["max_slice_rmse_deviation"] = 0.01
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 1.1
        sample_cfg["bias"]["min_slice_size"] = 10

        with pytest.raises(BiasGateError, match="Bias gate FAILED"):
            check_bias(mock_model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

    def test_warns_but_passes_for_small_slices(
        self, sample_cfg: dict[str, Any], mock_mlflow: Any, mocker: Any
    ) -> None:
        """check_bias warns but doesn't fail for small slices."""
        from models.crime_navigate.bias_checker import check_bias

        # Mock GCS loader
        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        # Create a mock model
        mock_model = MagicMock()

        # Create val_df with one small slice
        val_df = pd.DataFrame(
            {
                "h3_index": [f"h3_{i:03d}" for i in range(100)],
                "hour_bucket": [i % 6 for i in range(100)],
                "weighted_score_3d": np.random.uniform(0, 10, 100),
                "weighted_score_30d": np.random.uniform(0, 20, 100),
                "weighted_score_90d": np.random.uniform(0, 50, 100),
                "incident_count_30d": np.random.randint(0, 30, 100),
                "trend_3v10": np.random.uniform(-1, 1, 100),
                "trend_10v30": np.random.uniform(-1, 1, 100),
                "trend_30v90": np.random.uniform(-1, 1, 100),
                "gun_incident_count_30d": np.random.randint(0, 5, 100),
                "high_severity_ratio_30d": np.random.uniform(0, 1, 100),
                "night_score_ratio": np.random.uniform(0, 1, 100),
                "evening_score_ratio": np.random.uniform(0, 1, 100),
                "weekend_score_ratio": np.random.uniform(0, 1, 100),
                "neighbor_weighted_score_30d": np.random.uniform(0, 15, 100),
                "neighbor_trend_3v10": np.random.uniform(-1, 1, 100),
                "neighbor_gun_count_30d": np.random.randint(0, 10, 100),
                "danger_rate": np.concatenate([np.ones(97) * 1.0, np.ones(3) * 100.0]),
                "district": ["A1"] * 97 + ["B2"] * 3,  # B2 is small slice
            }
        )

        mock_model.predict.return_value = np.ones(100) * 1.0

        # Only evaluate district: hour_bucket slices (~17 rows each) would also be checked
        # and fail strict deviation vs overall RMSE on this synthetic frame.
        sample_cfg["bias"]["slice_dimensions"] = ["district"]

        # A1 has perfect preds (rmse=0) while overall RMSE is driven by the 3 B2 outliers,
        # so relative deviation for A1 is ~100%. Allow that so the gate tests *small-slice*
        # exemption only; B2 (count=3 < min_slice_size) still must not fail the gate.
        sample_cfg["bias"]["max_slice_rmse_deviation"] = 1.0
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 1.1
        sample_cfg["bias"]["min_slice_size"] = 10  # B2 has only 3

        # Should pass because B2 is too small to fail the gate
        result = check_bias(
            mock_model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id"
        )

        assert result.passed is True

    def test_saves_report_to_gcs(
        self,
        sample_cfg: dict[str, Any],
        sample_training_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """check_bias saves bias report to GCS."""
        from models.crime_navigate.bias_checker import check_bias
        from models.crime_navigate.trainer import random_split, train_model

        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, _ = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        sample_cfg["bias"]["max_slice_rmse_deviation"] = 0.9
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 10.0

        result = check_bias(model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

        mock_loader.write_json.assert_called_once()
        assert result.report_gcs_path is not None

    def test_handles_missing_slice_dimension(
        self,
        sample_cfg: dict[str, Any],
        sample_training_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """check_bias skips missing slice dimensions gracefully."""
        from models.crime_navigate.bias_checker import check_bias
        from models.crime_navigate.trainer import random_split, train_model

        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        # Remove district column
        val_df = val_df.drop(columns=["district"])

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, _ = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        sample_cfg["bias"]["max_slice_rmse_deviation"] = 0.9
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 10.0

        # Should not raise, just skip the missing dimension
        result = check_bias(model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

        assert result.passed is True

    def test_records_worst_slice_info(
        self,
        sample_cfg: dict[str, Any],
        sample_training_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """check_bias records worst slice information."""
        from models.crime_navigate.bias_checker import check_bias
        from models.crime_navigate.trainer import random_split, train_model

        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, _ = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        sample_cfg["bias"]["max_slice_rmse_deviation"] = 0.9
        sample_cfg["bias"]["max_slice_rmse_multiplier"] = 10.0

        result = check_bias(model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

        assert result.worst_deviation_pct >= 0

    def test_worst_slice_excludes_small_slices(
        self, sample_cfg: dict[str, Any], mock_mlflow: Any, mocker: Any
    ) -> None:
        """Headline worst_slice is max deviation among non-small slices only."""
        from models.crime_navigate.bias_checker import BiasGateError, check_bias

        mock_loader = MagicMock()
        mock_loader.write_json.return_value = "gs://test-bucket/bias_reports/report.json"
        mocker.patch("models.crime_navigate.bias_checker.GCSLoader", return_value=mock_loader)

        # 95 rows A1, 5 rows External — External has huge error but is small
        val_df = pd.DataFrame(
            {
                "h3_index": [f"h3_{i:03d}" for i in range(100)],
                "hour_bucket": [i % 6 for i in range(100)],
                "weighted_score_3d": np.random.uniform(0, 10, 100),
                "weighted_score_30d": np.random.uniform(0, 20, 100),
                "weighted_score_90d": np.random.uniform(0, 50, 100),
                "incident_count_30d": np.random.randint(0, 30, 100),
                "trend_3v10": np.random.uniform(-1, 1, 100),
                "trend_10v30": np.random.uniform(-1, 1, 100),
                "trend_30v90": np.random.uniform(-1, 1, 100),
                "gun_incident_count_30d": np.random.randint(0, 5, 100),
                "high_severity_ratio_30d": np.random.uniform(0, 1, 100),
                "night_score_ratio": np.random.uniform(0, 1, 100),
                "evening_score_ratio": np.random.uniform(0, 1, 100),
                "weekend_score_ratio": np.random.uniform(0, 1, 100),
                "neighbor_weighted_score_30d": np.random.uniform(0, 15, 100),
                "neighbor_trend_3v10": np.random.uniform(-1, 1, 100),
                "neighbor_gun_count_30d": np.random.randint(0, 10, 100),
                "danger_rate": np.concatenate([np.ones(95) * 1.0, np.ones(5) * 50.0]),
                "district": ["A1"] * 95 + ["External"] * 5,
            }
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones(100) * 1.0

        sample_cfg["bias"]["slice_dimensions"] = ["district"]
        sample_cfg["bias"]["min_slice_size"] = 10

        with pytest.raises(BiasGateError, match="among non-small slices"):
            check_bias(mock_model, val_df, "2024-01-15", sample_cfg, "test-bucket", "test-run-id")

        report = mock_loader.write_json.call_args[0][0]
        assert report["worst_slice"] == "district=A1"


class TestGetSliceSummary:
    """Tests for get_slice_summary function."""

    def test_returns_correct_summary(self, mocker: Any) -> None:
        """get_slice_summary returns correct counts."""
        from models.crime_navigate.bias_checker import get_slice_summary
        from shared.schemas import BiasResult

        result = BiasResult(
            passed=True,
            overall_rmse=0.5,
            slice_results={
                "district=A1": {"rmse": 0.4, "passed": True, "count": 50},
                "district=B2": {"rmse": 0.6, "passed": False, "count": 30},
                "hour_bucket=0": {"rmse": 0.5, "passed": True, "count": 20},
            },
            worst_slice="district=B2",
            worst_deviation_pct=20.0,
            report_gcs_path="gs://test/report.json",
        )

        summary = get_slice_summary(result)

        assert summary["total_slices"] == 3
        assert summary["passed_slices"] == 2
        assert summary["failed_slices"] == 1
        assert summary["worst_slice"] == "district=B2"
