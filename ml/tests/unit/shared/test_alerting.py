"""Tests for shared/alerting.py."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
import requests

alerting = importlib.import_module("shared.alerting")


class TestSlackHelpers:
    def test_get_webhook_url_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        assert alerting._get_webhook_url() is None

    def test_get_webhook_url_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        assert alerting._get_webhook_url() == "https://hooks.slack.com/test"

    def test_send_slack_no_webhook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        assert alerting._send_slack_message({"text": "hi"}) is False

    def test_send_slack_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_post = MagicMock()
        mock_post.return_value.raise_for_status = MagicMock()
        monkeypatch.setattr(requests, "post", mock_post)
        assert alerting._send_slack_message({"text": "x"}) is True
        mock_post.assert_called_once()

    def test_send_slack_request_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        monkeypatch.setattr(
            requests,
            "post",
            MagicMock(side_effect=RuntimeError("network")),
        )
        assert alerting._send_slack_message({"text": "x"}) is False


class TestAlertFunctions:
    def test_alert_training_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        alerting.alert_training_start("crime", "2024-01-15", "dag_x")
        assert "ML Training Started" in str(called[0])

    def test_alert_training_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: True)
        alerting.alert_training_complete(
            dataset="crime",
            execution_date="2024-01-15",
            train_result={},
            val_result={"rmse_val": 1.2345},
            bias_result={"passed": True},
            score_result={"h3_cells": 100},
            publish_result={"rows_upserted": 50},
            dag_id="d",
        )

    def test_alert_training_complete_rmse_not_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: True)
        alerting.alert_training_complete(
            "crime",
            "2024-01-15",
            {},
            {"rmse_val": "N/A"},
            {"passed": False},
            {"h3_cells": 0},
            {"rows_upserted": 0},
            "d",
        )

    def test_alert_gate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: True)
        alerting.alert_gate_failure("crime", "2024-01-15", "val", "err" * 200, "d")

    def test_alert_model_pushed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: True)
        alerting.alert_model_pushed("crime", "2024-01-15", "v1", "gs://m", 0.5, "d")

    def test_alert_scores_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: True)
        alerting.alert_scores_published("crime", "2024-01-15", 10, 5, 1.2, "d")

    def test_alert_model_promoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        comparison = {
            "should_promote": True,
            "reason": "candidate better than production",
            "production_version": "20240101",
            "production_rmse": 0.6,
            "candidate_rmse": 0.5,
            "delta_pct": -16.67,
        }
        alerting.alert_model_promoted("crime", "2024-01-15", "v2", comparison, "dag_x")
        assert "Model Promoted" in str(called[0])

    def test_alert_promotion_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        comparison = {
            "should_promote": False,
            "reason": "candidate worse than production by more than tolerance",
            "production_version": "20240101",
            "production_rmse": 0.5,
            "candidate_rmse": 0.6,
            "delta_pct": 20.0,
        }
        alerting.alert_promotion_skipped("crime", "2024-01-15", "v2", comparison, "dag_x")
        assert "Promotion Skipped" in str(called[0])

    def test_alert_drift_report_low_severity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        summary = {
            "execution_date": "2024-01-15",
            "model": "crime_navigate",
            "reference_rows": 10000,
            "current_rows": 5000,
            "n_features_total": 16,
            "n_features_drifted": 1,
            "drift_share": 0.0625,
            "dataset_drift_detected": False,
            "per_feature": {
                "weighted_score_3d": {"drift_detected": True, "drift_score": 0.15},
            },
            "html_gcs_uri": "gs://bucket/report.html",
        }
        alerting.alert_drift_report(summary)
        assert "LOW" in str(called[0])
        assert "✅" in str(called[0])

    def test_alert_drift_report_high_severity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        summary = {
            "execution_date": "2024-01-15",
            "model": "crime_navigate",
            "reference_rows": 10000,
            "current_rows": 5000,
            "n_features_total": 16,
            "n_features_drifted": 8,
            "drift_share": 0.5,
            "dataset_drift_detected": True,
            "per_feature": {
                "weighted_score_3d": {"drift_detected": True, "drift_score": 0.85},
                "trend_3v10": {"drift_detected": True, "drift_score": 0.72},
            },
            "html_gcs_uri": "gs://bucket/report.html",
        }
        alerting.alert_drift_report(summary)
        assert "HIGH" in str(called[0])
        assert "🚨" in str(called[0])

    def test_alert_retrain_triggered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[dict] = []
        monkeypatch.setattr(alerting, "_send_slack_message", lambda m: called.append(m) or True)
        summary = {
            "execution_date": "2024-01-15",
            "model": "crime_navigate",
            "n_features_total": 16,
            "n_features_drifted": 6,
            "drift_share": 0.375,
            "per_feature": {
                "weighted_score_3d": {"drift_detected": True, "drift_score": 0.42},
                "trend_3v10": {"drift_detected": True, "drift_score": 0.35},
            },
        }
        reasons = [
            "max feature drift 0.420 > 0.25",
            "drift share 37.5% > 30%",
        ]
        alerting.alert_retrain_triggered("crime_navigate", reasons, summary)
        assert "Retrain Triggered" in str(called[0])
        assert "🔄" in str(called[0])
        assert "crime_navigate" in str(called[0])
