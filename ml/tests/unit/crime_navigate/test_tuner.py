"""
Boston Pulse ML - Tuner Tests.

Tests for ml/models/crime_navigate/tuner.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class TestTuneHyperparams:
    """Tests for tune_hyperparams function."""

    def test_returns_best_params_and_result(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """tune_hyperparams returns best params and TuningResult."""
        from models.crime_navigate.trainer import random_split
        from models.crime_navigate.tuner import tune_hyperparams

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        # Reduce trials for faster test
        sample_cfg["tuning"]["n_trials"] = 2
        sample_cfg["tuning"]["timeout_seconds"] = 30

        best_params, result = tune_hyperparams(train_df, val_df, sample_cfg, "test-run-id")

        assert isinstance(best_params, dict)
        assert result.success is True
        assert result.n_trials >= 1
        assert result.best_val_rmse > 0

    def test_best_params_contain_required_keys(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """tune_hyperparams returns params with objective and metric."""
        from models.crime_navigate.trainer import random_split
        from models.crime_navigate.tuner import tune_hyperparams

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        sample_cfg["tuning"]["n_trials"] = 2
        sample_cfg["tuning"]["timeout_seconds"] = 30

        best_params, result = tune_hyperparams(train_df, val_df, sample_cfg, "test-run-id")

        assert "objective" in best_params
        assert "metric" in best_params
        assert best_params["objective"] == "regression_l1"
        assert best_params["metric"] == "rmse"

    def test_params_within_search_space(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """tune_hyperparams returns params within configured search space."""
        from models.crime_navigate.trainer import random_split
        from models.crime_navigate.tuner import tune_hyperparams

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        sample_cfg["tuning"]["n_trials"] = 2
        sample_cfg["tuning"]["timeout_seconds"] = 30

        best_params, result = tune_hyperparams(train_df, val_df, sample_cfg, "test-run-id")

        ss = sample_cfg["tuning"]["search_space"]

        assert ss["num_leaves"][0] <= best_params["num_leaves"] <= ss["num_leaves"][1]
        assert ss["learning_rate"][0] <= best_params["learning_rate"] <= ss["learning_rate"][1]
        assert ss["n_estimators"][0] <= best_params["n_estimators"] <= ss["n_estimators"][1]

    def test_logs_trials_to_mlflow(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mocker: Any
    ) -> None:
        """tune_hyperparams logs each trial to MLflow."""

        mock_start_run = mocker.patch("mlflow.start_run")
        mock_start_run.return_value.__enter__ = mocker.Mock(
            return_value=mocker.Mock(info=mocker.Mock(run_id="test-run-id"))
        )
        mock_start_run.return_value.__exit__ = mocker.Mock(return_value=False)

        mocker.patch("mlflow.log_params")
        mocker.patch("mlflow.log_metric")

        from models.crime_navigate.trainer import random_split
        from models.crime_navigate.tuner import tune_hyperparams

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        sample_cfg["tuning"]["n_trials"] = 2
        sample_cfg["tuning"]["timeout_seconds"] = 30

        tune_hyperparams(train_df, val_df, sample_cfg, "test-run-id")

        # Should have called start_run for each trial (nested runs)
        assert mock_start_run.call_count >= 2

    def test_result_contains_mlflow_run_id(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """TuningResult contains the parent MLflow run ID."""
        from models.crime_navigate.trainer import random_split
        from models.crime_navigate.tuner import tune_hyperparams

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        sample_cfg["tuning"]["n_trials"] = 2
        sample_cfg["tuning"]["timeout_seconds"] = 30

        best_params, result = tune_hyperparams(train_df, val_df, sample_cfg, "parent-run-123")

        assert result.mlflow_parent_run_id == "parent-run-123"


class TestGetDefaultParams:
    """Tests for get_default_params function."""

    def test_returns_valid_params(self, sample_cfg: dict[str, Any]) -> None:
        """get_default_params returns valid LightGBM parameters."""
        from models.crime_navigate.tuner import get_default_params

        params = get_default_params(sample_cfg)

        assert "objective" in params
        assert "metric" in params
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert "n_estimators" in params

    def test_default_params_can_train_model(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """get_default_params returns params that can train a model."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.tuner import get_default_params

        train_df, val_df = random_split(sample_training_df, sample_cfg)
        default_params = get_default_params(sample_cfg)

        # Should not raise
        model, path, result = train_model(
            train_df, val_df, default_params, sample_cfg, "test-run-id"
        )

        assert result.success is True
