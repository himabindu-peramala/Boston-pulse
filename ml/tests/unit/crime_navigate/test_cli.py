"""Unit tests for models/crime_navigate/cli.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from models.crime_navigate import cli as cli_module
from shared.schemas import (
    BiasResult,
    FeatureLoadResult,
    PublishResult,
    ScoringResult,
    TargetBuildResult,
    TrainingResult,
    TuningResult,
    ValidationResult,
)


def _feature_result() -> FeatureLoadResult:
    return FeatureLoadResult(rows=100, h3_cells=10, columns=["a"], success=True, error=None)


def _target_result() -> TargetBuildResult:
    return TargetBuildResult(
        rows=100,
        h3_cells=10,
        mean_danger_rate=0.5,
        zero_rate_cells=0,
        success=True,
        error=None,
    )


def _training_result() -> TrainingResult:
    return TrainingResult(
        model_path="/tmp/model.lgb",
        train_rmse=0.1,
        val_rmse=0.12,
        best_iteration=10,
        n_features=5,
        n_train_rows=80,
        n_val_rows=20,
        mlflow_run_id="mlflow-run-1",
        success=True,
        error=None,
    )


def _validation_result() -> ValidationResult:
    return ValidationResult(
        rmse_val=0.12,
        rmse_train=0.1,
        overfit_ratio=1.0,
        passed=True,
        shap_artifact_path=None,
        feature_importance={},
        error=None,
    )


def _bias_result() -> BiasResult:
    return BiasResult(
        passed=True,
        overall_rmse=0.12,
        slice_results={},
        worst_slice=None,
        worst_deviation_pct=1.0,
        report_gcs_path=None,
        error=None,
    )


def _scoring_result() -> ScoringResult:
    return ScoringResult(
        rows_scored=100,
        h3_cells=10,
        output_gcs_path="gs://b/scores",
        score_distribution={"low": 10},
        model_version="20240101",
        success=True,
        error=None,
    )


def _publish_result() -> PublishResult:
    return PublishResult(
        rows_upserted=50,
        firestore_collection="c",
        duration_seconds=1.0,
        model_version="20240101",
        success=True,
        error=None,
    )


def _tuning_result() -> TuningResult:
    return TuningResult(
        best_params={"num_leaves": 16},
        best_val_rmse=0.11,
        n_trials=2,
        mlflow_parent_run_id="mlflow-run-1",
        success=True,
        error=None,
    )


@pytest.fixture
def cli_mlflow_mocks(mocker: Any) -> None:
    mocker.patch.object(cli_module.mlflow, "set_experiment")
    mocker.patch.object(cli_module.mlflow, "end_run")
    run = MagicMock()
    run.info.run_id = "mlflow-run-1"
    mocker.patch.object(cli_module.mlflow, "start_run", return_value=run)


@pytest.fixture
def cli_alert_mocks(mocker: Any) -> None:
    mocker.patch("shared.alerting.alert_training_start")
    mocker.patch("shared.alerting.alert_gate_failure")
    mocker.patch("shared.alerting.alert_model_pushed")
    mocker.patch("shared.alerting.alert_scores_published")
    mocker.patch("shared.alerting.alert_training_complete")


@pytest.fixture
def pipeline_step_mocks(
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_features_df: Any,
    sample_training_df: Any,
    cli_alert_mocks: None,
) -> MagicMock:
    """Patch all heavy pipeline steps; returns mock ModelRegistry instance."""
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)

    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(sample_features_df, _feature_result()),
    )
    mocker.patch(
        "models.crime_navigate.target_builder.build_targets",
        return_value=(sample_training_df, _target_result()),
    )
    train_df = sample_training_df.iloc[:80]
    val_df = sample_training_df.iloc[80:]
    mocker.patch(
        "models.crime_navigate.trainer.random_split",
        return_value=(train_df, val_df),
    )
    model_mock = MagicMock()
    mocker.patch(
        "models.crime_navigate.trainer.train_model",
        return_value=(model_mock, "/tmp/m.lgb", _training_result()),
    )
    mocker.patch(
        "models.crime_navigate.tuner.tune_hyperparams",
        return_value=({"num_leaves": 16}, _tuning_result()),
    )
    mocker.patch(
        "models.crime_navigate.validator.validate_model",
        return_value=_validation_result(),
    )
    mocker.patch(
        "models.crime_navigate.bias_checker.check_bias",
        return_value=_bias_result(),
    )
    mocker.patch(
        "models.crime_navigate.scorer.score_all_cells",
        return_value=(sample_features_df, _scoring_result()),
    )
    mocker.patch(
        "models.crime_navigate.publisher.publish_scores",
        return_value=_publish_result(),
    )

    reg = MagicMock()
    reg.push.return_value = "gs://artifacts/model"
    reg.promote_to_production = MagicMock()
    mocker.patch("shared.registry.ModelRegistry", return_value=reg)
    return reg


def test_load_config_reads_yaml() -> None:
    """load_config returns a dict from the repo YAML."""
    cfg = cli_module.load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg
    assert cfg["data"]["bucket"]


def test_load_config_missing_raises(mocker: Any) -> None:
    """load_config raises FileNotFoundError when config path does not exist."""
    leaf = MagicMock()
    leaf.exists.return_value = False
    inter = MagicMock()
    inter.__truediv__ = MagicMock(return_value=leaf)
    level3 = MagicMock()
    level3.__truediv__ = MagicMock(return_value=inter)
    p2 = MagicMock()
    p2.parent = level3
    p1 = MagicMock()
    p1.parent = p2
    p = MagicMock()
    p.parent = p1
    mocker.patch.object(cli_module, "Path", MagicMock(return_value=p))

    with pytest.raises(FileNotFoundError, match="Config not found"):
        cli_module.load_config()


def test_run_training_pipeline_success(
    cli_mlflow_mocks: None,
    pipeline_step_mocks: MagicMock,
) -> None:
    out = cli_module.run_training_pipeline(
        execution_date="2024-01-15",
        stage="production",
        skip_tuning=False,
        skip_publish=False,
    )
    assert out["status"] == "success"
    assert out["mlflow_run_id"] == "mlflow-run-1"
    assert out["steps"]["tune"]["best_params"] == {"num_leaves": 16}
    pipeline_step_mocks.push.assert_called_once()
    pipeline_step_mocks.promote_to_production.assert_not_called()


def test_run_training_pipeline_staging_promotes(
    cli_mlflow_mocks: None,
    pipeline_step_mocks: MagicMock,
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_training_df: Any,
    cli_alert_mocks: None,
) -> None:
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)
    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(mocker.Mock(), _feature_result()),
    )
    mocker.patch(
        "models.crime_navigate.target_builder.build_targets",
        return_value=(sample_training_df, _target_result()),
    )
    tr, va = sample_training_df.iloc[:80], sample_training_df.iloc[80:]
    mocker.patch(
        "models.crime_navigate.trainer.random_split",
        return_value=(tr, va),
    )
    mocker.patch(
        "models.crime_navigate.trainer.train_model",
        return_value=(MagicMock(), "/x", _training_result()),
    )
    mocker.patch(
        "models.crime_navigate.tuner.tune_hyperparams",
        return_value=({}, _tuning_result()),
    )
    mocker.patch(
        "models.crime_navigate.validator.validate_model",
        return_value=_validation_result(),
    )
    mocker.patch(
        "models.crime_navigate.bias_checker.check_bias",
        return_value=_bias_result(),
    )
    mocker.patch(
        "models.crime_navigate.scorer.score_all_cells",
        return_value=(mocker.Mock(), _scoring_result()),
    )
    mocker.patch(
        "models.crime_navigate.publisher.publish_scores",
        return_value=_publish_result(),
    )
    mocker.patch("shared.registry.ModelRegistry", return_value=pipeline_step_mocks)
    cli_module.run_training_pipeline(
        execution_date="2024-01-15",
        stage="staging",
    )
    pipeline_step_mocks.promote_to_production.assert_called_once_with("20240115")


def test_run_training_pipeline_skip_tuning(
    cli_mlflow_mocks: None,
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_features_df: Any,
    sample_training_df: Any,
) -> None:
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)
    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(sample_features_df, _feature_result()),
    )
    mocker.patch(
        "models.crime_navigate.target_builder.build_targets",
        return_value=(sample_training_df, _target_result()),
    )
    mocker.patch(
        "models.crime_navigate.trainer.random_split",
        return_value=(sample_training_df, sample_training_df),
    )
    mocker.patch(
        "models.crime_navigate.trainer.train_model",
        return_value=(MagicMock(), "/m", _training_result()),
    )
    default_params = {"num_leaves": 31}
    mocker.patch(
        "models.crime_navigate.tuner.get_default_params",
        return_value=default_params,
    )
    mocker.patch(
        "models.crime_navigate.validator.validate_model",
        return_value=_validation_result(),
    )
    mocker.patch(
        "models.crime_navigate.bias_checker.check_bias",
        return_value=_bias_result(),
    )
    mocker.patch(
        "models.crime_navigate.scorer.score_all_cells",
        return_value=(sample_features_df, _scoring_result()),
    )
    mocker.patch(
        "models.crime_navigate.publisher.publish_scores",
        return_value=_publish_result(),
    )
    reg = MagicMock()
    reg.push.return_value = "gs://x"
    mocker.patch("shared.registry.ModelRegistry", return_value=reg)
    mocker.patch("shared.alerting.alert_training_start")
    mocker.patch("shared.alerting.alert_model_pushed")
    mocker.patch("shared.alerting.alert_scores_published")
    mocker.patch("shared.alerting.alert_training_complete")

    out = cli_module.run_training_pipeline(
        execution_date="2024-01-15",
        stage="production",
        skip_tuning=True,
    )
    assert out["steps"]["tune"]["skipped"] is True
    assert out["steps"]["tune"]["best_params"] == default_params


def test_run_training_pipeline_skip_publish(
    cli_mlflow_mocks: None,
    pipeline_step_mocks: MagicMock,
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_features_df: Any,
    sample_training_df: Any,
) -> None:
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)
    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(sample_features_df, _feature_result()),
    )
    mocker.patch(
        "models.crime_navigate.target_builder.build_targets",
        return_value=(sample_training_df, _target_result()),
    )
    mocker.patch(
        "models.crime_navigate.trainer.random_split",
        return_value=(sample_training_df.iloc[:10], sample_training_df.iloc[10:]),
    )
    mocker.patch(
        "models.crime_navigate.trainer.train_model",
        return_value=(MagicMock(), "/m", _training_result()),
    )
    mocker.patch(
        "models.crime_navigate.tuner.tune_hyperparams",
        return_value=({}, _tuning_result()),
    )
    mocker.patch(
        "models.crime_navigate.validator.validate_model",
        return_value=_validation_result(),
    )
    mocker.patch(
        "models.crime_navigate.bias_checker.check_bias",
        return_value=_bias_result(),
    )
    mocker.patch(
        "models.crime_navigate.scorer.score_all_cells",
        return_value=(sample_features_df, _scoring_result()),
    )
    pub = mocker.patch("models.crime_navigate.publisher.publish_scores")
    mocker.patch("shared.alerting.alert_training_start")
    mocker.patch("shared.alerting.alert_model_pushed")
    mocker.patch("shared.alerting.alert_scores_published")
    mocker.patch("shared.alerting.alert_training_complete")
    mocker.patch("shared.registry.ModelRegistry", return_value=pipeline_step_mocks)

    out = cli_module.run_training_pipeline(
        execution_date="2024-01-15",
        stage="production",
        skip_publish=True,
    )
    assert out["steps"]["publish"]["skipped"] is True
    pub.assert_not_called()


def test_run_training_pipeline_feature_load_fails(
    cli_mlflow_mocks: None,
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_features_df: Any,
) -> None:
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)
    bad = FeatureLoadResult(rows=0, h3_cells=0, columns=[], success=False, error="missing")
    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(sample_features_df, bad),
    )
    mocker.patch("shared.alerting.alert_training_start")
    with pytest.raises(RuntimeError, match="Feature loading failed"):
        cli_module.run_training_pipeline("2024-01-15", "production")


def test_run_training_pipeline_validation_gate_alerts(
    cli_mlflow_mocks: None,
    mocker: Any,
    sample_cfg: dict[str, Any],
    sample_features_df: Any,
    sample_training_df: Any,
) -> None:
    mocker.patch.object(cli_module, "load_config", return_value=sample_cfg)
    mocker.patch(
        "models.crime_navigate.feature_loader.load_features",
        return_value=(sample_features_df, _feature_result()),
    )
    mocker.patch(
        "models.crime_navigate.target_builder.build_targets",
        return_value=(sample_training_df, _target_result()),
    )
    mocker.patch(
        "models.crime_navigate.trainer.random_split",
        return_value=(sample_training_df, sample_training_df),
    )
    mocker.patch(
        "models.crime_navigate.trainer.train_model",
        return_value=(MagicMock(), "/m", _training_result()),
    )
    mocker.patch(
        "models.crime_navigate.tuner.tune_hyperparams",
        return_value=({}, _tuning_result()),
    )
    mocker.patch(
        "models.crime_navigate.validator.validate_model",
        side_effect=ValueError("gate"),
    )
    alert = mocker.patch("shared.alerting.alert_gate_failure")
    mocker.patch("shared.alerting.alert_training_start")

    with pytest.raises(ValueError, match="gate"):
        cli_module.run_training_pipeline("2024-01-15", "production")

    alert.assert_called_once()


def test_main_train_success(
    mocker: Any,
    tmp_path: Path,
    cli_mlflow_mocks: None,
    pipeline_step_mocks: MagicMock,
) -> None:
    out_path = tmp_path / "out.json"
    mocker.patch.object(
        sys,
        "argv",
        ["cli", "train", "--execution-date", "2024-01-15", "--output-json", str(out_path)],
    )
    rc = cli_module.main()
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["status"] == "success"


def test_main_train_failure_writes_json(
    mocker: Any,
    tmp_path: Path,
    cli_mlflow_mocks: None,
) -> None:
    mocker.patch.object(cli_module, "load_config", side_effect=RuntimeError("boom"))
    mocker.patch("shared.alerting.alert_training_start")
    out_path = tmp_path / "err.json"
    mocker.patch.object(
        sys,
        "argv",
        [
            "cli",
            "train",
            "--execution-date",
            "2024-01-15",
            "--output-json",
            str(out_path),
        ],
    )
    rc = cli_module.main()
    assert rc == 1
    err = json.loads(out_path.read_text())
    assert err["status"] == "failed"
