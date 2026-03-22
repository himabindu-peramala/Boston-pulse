"""Tests for shared/mlflow_utils.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.mlflow_utils import (
    get_or_create_run,
    get_run_artifact_uri,
    log_metrics_safe,
    log_model_info,
    log_params_safe,
    setup_mlflow,
)


@pytest.fixture
def sample_cfg() -> dict:
    return {
        "mlflow": {"experiment_name": "exp-a"},
        "model": {"name": "m1", "version_prefix": "vpre"},
    }


def test_setup_mlflow(
    sample_cfg: dict, monkeypatch: pytest.MonkeyPatch, mocker: pytest.Mock
) -> None:
    mocker.patch("shared.mlflow_utils.mlflow.set_tracking_uri")
    mocker.patch("shared.mlflow_utils.mlflow.set_experiment")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///t.db")
    setup_mlflow(sample_cfg)


def test_get_or_create_run(sample_cfg: dict, mocker: pytest.Mock) -> None:
    mocker.patch("shared.mlflow_utils.setup_mlflow")
    mocker.patch("shared.mlflow_utils.mlflow.end_run")
    run = MagicMock()
    run.info.run_id = "run-123"
    mocker.patch("shared.mlflow_utils.mlflow.start_run", return_value=run)
    rid = get_or_create_run(sample_cfg, "2024-06-01")
    assert rid == "run-123"


def test_log_params_safe_nested_and_list(mocker: pytest.Mock) -> None:
    mlog = mocker.patch("shared.mlflow_utils.mlflow.log_param")
    log_params_safe({"a": {"b": 1}, "c": [1, 2, 3], "d": "ok"})
    assert mlog.call_count >= 3


def test_log_params_safe_failure_logged(mocker: pytest.Mock) -> None:
    mocker.patch(
        "shared.mlflow_utils.mlflow.log_param",
        side_effect=[None, ValueError("x")],
    )
    log_params_safe({"ok": 1, "bad": 2})


def test_log_metrics_safe_nan_skipped(mocker: pytest.Mock) -> None:
    mlm = mocker.patch("shared.mlflow_utils.mlflow.log_metric")
    log_metrics_safe({"a": float("nan"), "b": 1.0, "c": None}, step=1)
    mlm.assert_called_once()


def test_log_metrics_safe_exception_path(mocker: pytest.Mock) -> None:
    mocker.patch(
        "shared.mlflow_utils.mlflow.log_metric",
        side_effect=RuntimeError("mlflow down"),
    )
    log_metrics_safe({"b": 1.0})


def test_get_run_artifact_uri(mocker: pytest.Mock) -> None:
    run = MagicMock()
    run.info.artifact_uri = "file:///art"
    mocker.patch("shared.mlflow_utils.mlflow.get_run", return_value=run)
    assert get_run_artifact_uri("rid") == "file:///art"


def test_log_model_info(mocker: pytest.Mock) -> None:
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    mocker.patch("shared.mlflow_utils.mlflow.start_run", return_value=cm)
    mocker.patch("shared.mlflow_utils.mlflow.log_param")
    mocker.patch("shared.mlflow_utils.mlflow.log_metric")
    log_model_info("rid", "/m.lgb", ["f1"], 0.1, 0.2)
