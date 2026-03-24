"""
Boston Pulse ML - Trainer Tests.

Tests for ml/models/crime_navigate/trainer.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


class TestRandomSplit:
    """Tests for random_split function."""

    def test_splits_data_correctly(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame
    ) -> None:
        """random_split returns train and val DataFrames with correct sizes."""
        from models.crime_navigate.trainer import random_split

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        total = len(sample_training_df)
        expected_val = int(total * sample_cfg["training"]["val_fraction"])

        assert len(val_df) == pytest.approx(expected_val, abs=5)
        assert len(train_df) + len(val_df) == total

    def test_split_is_reproducible(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame
    ) -> None:
        """random_split produces same split with same seed."""
        from models.crime_navigate.trainer import random_split

        train1, val1 = random_split(sample_training_df, sample_cfg)
        train2, val2 = random_split(sample_training_df, sample_cfg)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.reset_index(drop=True), val2.reset_index(drop=True))


class TestTrainModel:
    """Tests for train_model function."""

    def test_returns_model_and_result(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model returns a model, path, and TrainingResult."""
        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert model is not None
        assert path.endswith(".lgb")
        assert result.success is True
        assert result.train_rmse >= 0
        assert result.val_rmse >= 0

    def test_computes_rmse_correctly(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model computes train and val RMSE."""
        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert result.train_rmse > 0
        assert result.val_rmse > 0
        # Train RMSE should generally be lower than val RMSE
        # (not always true but usually)

    def test_saves_model_to_path(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model saves model to the returned path."""
        import os

        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert os.path.exists(path)

    def test_handles_categorical_features(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model handles categorical features correctly."""
        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        # Should not raise even with categorical columns
        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert result.success is True

    def test_records_best_iteration(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model records best iteration from early stopping."""
        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }

        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert result.best_iteration > 0
        assert result.best_iteration <= 100

    def test_records_feature_count(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """train_model records number of features."""
        from models.crime_navigate.trainer import random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, path, result = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        assert result.n_features == len(sample_cfg["features"]["input_columns"])


class TestLoadModel:
    """Tests for load_model function."""

    def test_loads_saved_model(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """load_model loads a previously saved model."""
        from models.crime_navigate.trainer import load_model, random_split, train_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        _, path, _ = train_model(train_df, val_df, best_params, sample_cfg, "test-run-id")

        loaded_model = load_model(path)
        assert loaded_model is not None


class TestPredict:
    """Tests for predict function."""

    def test_returns_non_negative_predictions(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """predict returns non-negative values (danger_rate cannot be negative)."""
        from models.crime_navigate.trainer import predict, random_split, train_model

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

        feature_cols = sample_cfg["features"]["input_columns"]
        preds = predict(model, val_df, feature_cols)

        assert np.all(preds >= 0)
