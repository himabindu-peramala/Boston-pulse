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
