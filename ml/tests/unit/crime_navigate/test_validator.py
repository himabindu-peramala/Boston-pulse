"""
Boston Pulse ML - Validator Tests.

Tests for ml/models/crime_navigate/validator.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


class TestValidateModel:
    """Tests for validate_model function."""

    def test_passes_when_rmse_within_gate(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model passes when RMSE is within gate threshold."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        # Set a high RMSE gate so it passes
        sample_cfg["validation"]["rmse_gate"] = 100.0

        result = validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

        assert result.passed is True
        assert result.rmse_val > 0

    def test_fails_when_rmse_exceeds_gate(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model raises ValidationGateError when RMSE exceeds gate."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import ValidationGateError, validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        # Set a very low RMSE gate so it fails
        sample_cfg["validation"]["rmse_gate"] = 0.001

        with pytest.raises(ValidationGateError, match="RMSE gate FAILED"):
            validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

    def test_fails_when_overfit_ratio_exceeds_gate(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model raises ValidationGateError when overfit ratio exceeds gate."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import ValidationGateError, validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        # Set a very low overfit gate so it fails
        sample_cfg["validation"]["rmse_gate"] = 100.0
        sample_cfg["validation"]["overfit_ratio_gate"] = 0.1

        with pytest.raises(ValidationGateError, match="Overfit gate FAILED"):
            validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

    def test_fails_when_val_set_too_small(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model raises ValidationGateError when val set has too few cells."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import ValidationGateError, validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        # Set a very high min_val_cells so it fails
        sample_cfg["validation"]["min_val_cells"] = 10000

        with pytest.raises(ValidationGateError, match="cells, minimum required"):
            validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

    def test_computes_overfit_ratio_correctly(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model computes overfit_ratio = train_rmse / val_rmse."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        sample_cfg["validation"]["rmse_gate"] = 100.0

        result = validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

        expected_ratio = training_result.train_rmse / result.rmse_val
        assert result.overfit_ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_returns_complete_validation_result(
        self, sample_cfg: dict[str, Any], sample_training_df: pd.DataFrame, mock_mlflow: Any
    ) -> None:
        """validate_model returns ValidationResult with all fields."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import validate_model

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        sample_cfg["validation"]["rmse_gate"] = 100.0

        result = validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

        assert result.rmse_val > 0
        assert result.rmse_train > 0
        assert result.overfit_ratio > 0
        assert result.passed is True

    def test_shap_analysis_runs(
        self,
        sample_cfg: dict[str, Any],
        sample_training_df: pd.DataFrame,
        mock_mlflow: Any,
        mocker: Any,
    ) -> None:
        """validate_model runs SHAP analysis and returns path."""
        from models.crime_navigate.trainer import random_split, train_model
        from models.crime_navigate.validator import validate_model

        # Mock matplotlib to avoid display issues
        mocker.patch("matplotlib.pyplot.savefig")
        mocker.patch("matplotlib.pyplot.close")

        train_df, val_df = random_split(sample_training_df, sample_cfg)

        best_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 20,
            "learning_rate": 0.1,
            "n_estimators": 50,
        }

        model, _, training_result = train_model(
            train_df, val_df, best_params, sample_cfg, "test-run-id"
        )

        sample_cfg["validation"]["rmse_gate"] = 100.0

        result = validate_model(model, val_df, training_result, sample_cfg, "test-run-id")

        # SHAP should have run (path may be None if SHAP fails, but shouldn't raise)
        assert result.passed is True
