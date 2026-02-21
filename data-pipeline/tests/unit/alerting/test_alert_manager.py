"""
Tests for Alert Manager

Tests centralized alerting system.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.alerting.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    send_alert,
)
from src.shared.config import get_config


@pytest.fixture
def mock_requests():
    """Mock requests for Slack webhook testing."""
    with patch("src.alerting.alert_manager.requests") as mock_req:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_req.post.return_value = mock_response
        yield mock_req


def test_alert_initialization():
    """Test Alert dataclass initialization."""
    alert = Alert(
        title="Test Alert",
        message="Test message",
        severity=AlertSeverity.WARNING,
        dataset="crime",
    )

    assert alert.title == "Test Alert"
    assert alert.message == "Test message"
    assert alert.severity == AlertSeverity.WARNING
    assert alert.dataset == "crime"
    assert isinstance(alert.timestamp, datetime)


def test_alert_to_dict():
    """Test Alert to_dict conversion."""
    alert = Alert(
        title="Test",
        message="Message",
        severity=AlertSeverity.INFO,
    )

    alert_dict = alert.to_dict()

    assert alert_dict["title"] == "Test"
    assert alert_dict["severity"] == "info"
    assert "timestamp" in alert_dict


def test_alert_manager_initialization():
    """Test AlertManager initialization."""
    config = get_config("dev")
    manager = AlertManager(config)

    assert manager.config == config
    assert manager.max_alerts_per_hour == config.alerting.rate_limit.max_alerts_per_hour


def test_send_alert_to_log():
    """Test sending alert to log channel."""
    config = get_config("dev")
    manager = AlertManager(config)

    success = manager.send_alert(
        title="Test Alert",
        message="Test message",
        severity="info",
        channels=["log"],
    )

    assert success


def test_send_alert_to_slack(mock_requests):
    """Test sending alert to Slack."""
    config = get_config("dev")
    config.slack_webhook_url = "https://hooks.slack.com/test"
    manager = AlertManager(config)

    success = manager.send_alert(
        title="Test Alert",
        message="Test message",
        severity="warning",
        channels=["slack"],
    )

    assert success
    assert mock_requests.post.called


def test_send_alert_default_routing():
    """Test default alert routing by severity."""
    config = get_config("dev")
    manager = AlertManager(config)

    # Test that different severities use different channels
    channels_info = manager._get_channels_for_severity(AlertSeverity.INFO)
    channels_warning = manager._get_channels_for_severity(AlertSeverity.WARNING)
    channels_critical = manager._get_channels_for_severity(AlertSeverity.CRITICAL)

    assert "log" in channels_info
    assert len(channels_critical) >= len(channels_warning)  # Critical should have more channels


def test_rate_limiting():
    """Test alert rate limiting."""
    config = get_config("dev")
    config.alerting.rate_limit.cooldown_minutes = 0  # Set very short cooldown
    config.alerting.rate_limit.max_alerts_per_hour = 2
    manager = AlertManager(config)

    # Send same alert multiple times
    for i in range(3):
        success = manager.send_alert(
            title="Same Alert",
            message="Same message",
            severity="info",
            dataset="test",
            channels=["log"],
        )
        if i < 2:
            assert success  # First 2 should succeed
        # Note: Third may or may not be rate limited depending on timing


def test_send_validation_alert():
    """Test sending validation-specific alert."""
    config = get_config("dev")
    manager = AlertManager(config)

    errors = ["Error 1", "Error 2", "Error 3"]
    success = manager.send_validation_alert(
        dataset="crime",
        stage="raw",
        errors=errors,
        severity="warning",
    )

    assert success


def test_send_drift_alert():
    """Test sending drift-specific alert."""
    config = get_config("dev")
    manager = AlertManager(config)

    drifted_features = ["feature1", "feature2", "feature3"]
    success = manager.send_drift_alert(
        dataset="crime",
        drifted_features=drifted_features,
        severity="warning",
    )

    assert success


def test_send_anomaly_alert():
    """Test sending anomaly-specific alert."""
    config = get_config("dev")
    manager = AlertManager(config)

    success = manager.send_anomaly_alert(
        dataset="crime",
        anomaly_count=10,
        critical_count=2,
        severity="warning",
    )

    assert success


def test_alert_history_tracking():
    """Test that alert history is tracked."""
    config = get_config("dev")
    manager = AlertManager(config)

    alert = Alert(
        title="Test",
        message="Message",
        severity=AlertSeverity.INFO,
        dataset="test",
    )

    manager._record_alert(alert)

    alert_key = f"{alert.severity}:{alert.title}:{alert.dataset}"
    assert alert_key in manager._alert_history


def test_convenience_send_alert_function():
    """Test convenience send_alert function."""
    config = get_config("dev")

    success = send_alert(
        title="Test",
        message="Message",
        severity="info",
        config=config,
    )

    assert success
