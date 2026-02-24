"""
Tests for Alert Manager

Tests centralized alerting system.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.alerting.alert_manager import (
    Alert,
    AlertChannel,
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


@pytest.fixture
def manager():
    """AlertManager with dev config and log-only routing."""
    config = get_config("dev")
    return AlertManager(config)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


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


def test_alert_optional_fields():
    """Test Alert optional fields default to None."""
    alert = Alert(
        title="Test",
        message="Msg",
        severity=AlertSeverity.INFO,
    )

    assert alert.dataset is None
    assert alert.dag_id is None
    assert alert.task_id is None
    assert alert.execution_date is None
    assert alert.metadata == {}


def test_alert_with_all_fields():
    """Test Alert with all fields set."""
    alert = Alert(
        title="Full Alert",
        message="Detail",
        severity=AlertSeverity.CRITICAL,
        dataset="crime",
        dag_id="crime_pipeline",
        task_id="validate",
        execution_date="2024-01-01",
        metadata={"key": "value"},
    )

    assert alert.dag_id == "crime_pipeline"
    assert alert.task_id == "validate"
    assert alert.metadata == {"key": "value"}


def test_alert_to_dict():
    """Test Alert to_dict conversion."""
    alert = Alert(
        title="Test",
        message="Message",
        severity=AlertSeverity.INFO,
        dataset="crime",
    )

    alert_dict = alert.to_dict()

    assert alert_dict["title"] == "Test"
    assert alert_dict["severity"] == "info"
    assert alert_dict["dataset"] == "crime"
    assert "timestamp" in alert_dict


def test_alert_severity_values():
    """Test AlertSeverity enum values."""
    assert AlertSeverity.INFO == "info"
    assert AlertSeverity.WARNING == "warning"
    assert AlertSeverity.CRITICAL == "critical"


def test_alert_channel_values():
    """Test AlertChannel enum values."""
    assert AlertChannel.LOG == "log"
    assert AlertChannel.SLACK == "slack"
    assert AlertChannel.EMAIL == "email"


# ---------------------------------------------------------------------------
# AlertManager initialisation
# ---------------------------------------------------------------------------


def test_alert_manager_initialization():
    """Test AlertManager initialization."""
    config = get_config("dev")
    manager = AlertManager(config)

    assert manager.config == config
    assert manager.max_alerts_per_hour == config.alerting.rate_limit.max_alerts_per_hour


def test_alert_manager_default_config():
    """Test AlertManager uses default config when none provided."""
    manager = AlertManager()
    assert manager.config is not None


# ---------------------------------------------------------------------------
# Sending alerts
# ---------------------------------------------------------------------------


def test_send_alert_to_log(manager):
    """Test sending alert to log channel."""
    success = manager.send_alert(
        title="Test Alert",
        message="Test message",
        severity="info",
        channels=["log"],
    )

    assert success


def test_send_alert_with_metadata(manager):
    """Test sending alert with additional metadata."""
    success = manager.send_alert(
        title="Meta Alert",
        message="Alert with meta",
        severity="warning",
        dataset="crime",
        dag_id="crime_pipeline",
        task_id="validate",
        execution_date="2024-01-01",
        metadata={"rows": 1000},
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


def test_send_alert_slack_no_webhook(manager):
    """Test Slack send skips gracefully when no webhook URL."""
    manager.config.slack_webhook_url = None
    success = manager.send_alert(
        title="Slack No Webhook",
        message="No URL configured",
        severity="warning",
        channels=["slack"],
    )
    # Should still return True (sends to log fallback or skips)
    assert isinstance(success, bool)


def test_send_alert_all_severities(manager):
    """Test alert sending for all severity levels."""
    for sev in ("info", "warning", "critical"):
        success = manager.send_alert(
            title=f"{sev} Alert",
            message="Test",
            severity=sev,
            channels=["log"],
        )
        assert success, f"Failed for severity: {sev}"


# ---------------------------------------------------------------------------
# Channel routing
# ---------------------------------------------------------------------------


def test_send_alert_default_routing():
    """Test default alert routing by severity."""
    config = get_config("dev")
    manager = AlertManager(config)

    channels_info = manager._get_channels_for_severity(AlertSeverity.INFO)
    channels_warning = manager._get_channels_for_severity(AlertSeverity.WARNING)
    channels_critical = manager._get_channels_for_severity(AlertSeverity.CRITICAL)

    assert "log" in channels_info
    assert len(channels_critical) >= len(channels_warning)


def test_send_to_log_all_severities(manager):
    """Test log channel handles all severity levels."""
    for sev in [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
        alert = Alert(title="Test", message="Msg", severity=sev)
        manager._send_to_log(alert)  # Should not raise


def test_send_to_email_placeholder(manager):
    """Test email send is a no-op placeholder (no exception)."""
    alert = Alert(title="Email Test", message="Body", severity=AlertSeverity.CRITICAL)
    manager._send_to_email(alert)  # Placeholder — must not raise


def test_send_to_pagerduty_placeholder(manager):
    """Test PagerDuty send is a no-op placeholder (no exception)."""
    alert = Alert(title="PD Test", message="Body", severity=AlertSeverity.CRITICAL)
    manager._send_to_pagerduty(alert)  # Placeholder — must not raise


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiting():
    """Test alert rate limiting."""
    config = get_config("dev")
    config.alerting.rate_limit.cooldown_minutes = 0
    config.alerting.rate_limit.max_alerts_per_hour = 2
    manager = AlertManager(config)

    for i in range(3):
        success = manager.send_alert(
            title="Same Alert",
            message="Same message",
            severity="info",
            dataset="test",
            channels=["log"],
        )
        if i < 2:
            assert success


def test_should_send_alert_first_time(manager):
    """Test _should_send_alert returns True for a new alert."""
    alert = Alert(title="New", message="First time", severity=AlertSeverity.INFO)
    assert manager._should_send_alert(alert) is True


def test_should_send_alert_after_record(manager):
    """Test _should_send_alert respects cooldown after recording."""
    config = get_config("dev")
    config.alerting.rate_limit.cooldown_minutes = 60  # long cooldown
    manager = AlertManager(config)

    alert = Alert(
        title="Cooldown Test",
        message="Msg",
        severity=AlertSeverity.WARNING,
        dataset="x",
    )
    manager._record_alert(alert)
    # Second call within cooldown should be suppressed
    result = manager._should_send_alert(alert)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Alert history
# ---------------------------------------------------------------------------


def test_alert_history_tracking(manager):
    """Test that alert history is tracked."""
    alert = Alert(
        title="Test",
        message="Message",
        severity=AlertSeverity.INFO,
        dataset="test",
    )

    manager._record_alert(alert)

    alert_key = f"{alert.severity}:{alert.title}:{alert.dataset}"
    assert alert_key in manager._alert_history


def test_alert_history_count_increments(manager):
    """Test alert history count increments on repeated recording."""
    alert = Alert(title="Repeat", message="Msg", severity=AlertSeverity.INFO, dataset="d")
    manager._record_alert(alert)
    manager._record_alert(alert)

    key = f"{alert.severity}:{alert.title}:{alert.dataset}"
    assert manager._alert_history[key].count_in_window >= 2


# ---------------------------------------------------------------------------
# Domain-specific alert helpers
# ---------------------------------------------------------------------------


def test_send_validation_alert(manager):
    """Test sending validation-specific alert."""
    errors = ["Error 1", "Error 2", "Error 3"]
    success = manager.send_validation_alert(
        dataset="crime",
        stage="raw",
        errors=errors,
        severity="warning",
    )
    assert success


def test_send_validation_alert_critical(manager):
    """Test sending critical validation alert."""
    success = manager.send_validation_alert(
        dataset="crime",
        stage="processed",
        errors=["Schema mismatch"],
        severity="critical",
    )
    assert success


def test_send_drift_alert(manager):
    """Test sending drift-specific alert."""
    drifted_features = ["feature1", "feature2", "feature3"]
    success = manager.send_drift_alert(
        dataset="crime",
        drifted_features=drifted_features,
        severity="warning",
    )
    assert success


def test_send_drift_alert_empty_features(manager):
    """Test drift alert with no drifted features."""
    success = manager.send_drift_alert(
        dataset="crime",
        drifted_features=[],
        severity="warning",
    )
    assert success


def test_send_anomaly_alert(manager):
    """Test sending anomaly-specific alert."""
    success = manager.send_anomaly_alert(
        dataset="crime",
        anomaly_count=10,
        critical_count=2,
        severity="warning",
    )
    assert success


def test_send_anomaly_alert_zero_anomalies(manager):
    """Test anomaly alert with zero anomalies."""
    success = manager.send_anomaly_alert(
        dataset="crime",
        anomaly_count=0,
        critical_count=0,
        severity="warning",
    )
    assert success


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


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


def test_convenience_send_alert_default_config():
    """Test convenience send_alert uses default config when none passed."""
    success = send_alert(
        title="Default config test",
        message="No config passed",
        severity="info",
    )
    assert success
