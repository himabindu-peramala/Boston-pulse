"""
Boston Pulse - Alert Manager

Centralized alerting system with severity-based routing:
- Slack webhooks for team notifications
- Email via SendGrid for critical alerts
- Rate limiting to prevent alert fatigue
- Alert history tracking

Usage:
    alert_manager = AlertManager(config)

    # Send an alert
    alert_manager.send_alert(
        title="High null ratio detected",
        message="Crime dataset has 15% null values in latitude column",
        severity="warning",
        dataset="crime",
        metadata={"null_ratio": 0.15, "column": "latitude"}
    )

    # Alert with custom routing
    alert_manager.send_alert(
        title="Pipeline failed",
        message="Crime ingestion failed after 3 retries",
        severity="critical",
        channels=["slack", "email"]
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

import requests

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(StrEnum):
    """Alert delivery channels."""

    LOG = "log"
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"


@dataclass
class Alert:
    """Alert message."""

    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    dataset: str | None = None
    dag_id: str | None = None
    task_id: str | None = None
    execution_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "dataset": self.dataset,
            "dag_id": self.dag_id,
            "task_id": self.task_id,
            "execution_date": self.execution_date,
            "metadata": self.metadata,}


@dataclass
class AlertHistory:
    """Track alert history for rate limiting."""

    alert_key: str
    last_sent: datetime
    count_in_window: int = 1


class AlertManager:
    """
    Centralized alert management with severity-based routing.

    Routes alerts to appropriate channels based on severity:
    - INFO: Log only
    - WARNING: Log + Slack
    - CRITICAL: Log + Slack + Email + PagerDuty
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize alert manager.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()

        # Alert history for rate limiting
        self._alert_history: dict[str, AlertHistory] = {}

        # Rate limit settings
        self.max_alerts_per_hour = self.config.alerting.rate_limit.max_alerts_per_hour
        self.cooldown_minutes = self.config.alerting.rate_limit.cooldown_minutes

    def send_alert(
        self,
        title: str,
        message: str,
        severity: Literal["info", "warning", "critical"] = "info",
        dataset: str | None = None,
        dag_id: str | None = None,
        task_id: str | None = None,
        execution_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> bool:
        """
        Send an alert through configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Severity level (info, warning, critical)
            dataset: Associated dataset
            dag_id: Associated DAG ID
            task_id: Associated task ID
            execution_date: Execution date
            metadata: Additional metadata
            channels: Override default channel routing

        Returns:
            True if alert was sent, False if rate limited
        """
        # Create alert object
        alert = Alert(
            title=title,
            message=message,
            severity=AlertSeverity(severity),
            dataset=dataset,
            dag_id=dag_id,
            task_id=task_id,
            execution_date=execution_date,
            metadata=metadata or {},
        )

        # Check rate limiting
        if not self._should_send_alert(alert):
            logger.warning(
                f"Alert rate limited: {title}",
                extra={"title": title, "severity": severity},
            )
            return False

        # Determine channels
        if channels is None:
            channels = self._get_channels_for_severity(alert.severity)

        # Send to each channel
        success = True
        for channel in channels:
            try:
                self._send_to_channel(alert, AlertChannel(channel))
            except Exception as e:
                logger.error(
                    f"Failed to send alert to {channel}: {e}",
                    extra={"channel": channel, "alert": alert.title},
                    exc_info=True,
                )
                success = False

        # Update alert history
        self._record_alert(alert)

        return success

    def send_validation_alert(
        self,
        dataset: str,
        stage: str,
        errors: list[str],
        severity: Literal["warning", "critical"] = "warning",
    ) -> bool:
        """
        Send alert for validation failures.

        Args:
            dataset: Dataset name
            stage: Validation stage (raw, processed, features)
            errors: List of validation errors
            severity: Alert severity

        Returns:
            True if sent successfully
        """
        title = f"Validation Failed: {dataset} ({stage})"
        message = f"Validation errors in {dataset} {stage} data:\n" + "\n".join(
            f"  - {error}" for error in errors[:5]  # Show first 5 errors
        )

        if len(errors) > 5:
            message += f"\n  ... and {len(errors) - 5} more errors"

        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            dataset=dataset,
            metadata={"stage": stage, "error_count": len(errors), "errors": errors},
        )

    def send_drift_alert(
        self,
        dataset: str,
        drifted_features: list[str],
        severity: Literal["warning", "critical"] = "warning",
    ) -> bool:
        """
        Send alert for drift detection.

        Args:
            dataset: Dataset name
            drifted_features: List of features with drift
            severity: Alert severity

        Returns:
            True if sent successfully
        """
        title = f"Data Drift Detected: {dataset}"
        message = f"Drift detected in {len(drifted_features)} features:\n" + "\n".join(
            f"  - {feature}" for feature in drifted_features[:10]
        )

        if len(drifted_features) > 10:
            message += f"\n  ... and {len(drifted_features) - 10} more features"

        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            dataset=dataset,
            metadata={"drifted_features": drifted_features},
        )

    def send_anomaly_alert(
        self,
        dataset: str,
        anomaly_count: int,
        critical_count: int,
        severity: Literal["warning", "critical"] = "warning",
    ) -> bool:
        """
        Send alert for anomaly detection.

        Args:
            dataset: Dataset name
            anomaly_count: Total anomaly count
            critical_count: Critical anomaly count
            severity: Alert severity

        Returns:
            True if sent successfully
        """
        title = f"Anomalies Detected: {dataset}"
        message = f"Found {anomaly_count} anomalies ({critical_count} critical) in {dataset}"

        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            dataset=dataset,
            metadata={"anomaly_count": anomaly_count, "critical_count": critical_count},
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on rate limiting."""
        # Generate alert key for deduplication
        alert_key = f"{alert.severity}:{alert.title}:{alert.dataset}"

        now = datetime.now(UTC)

        # Check if we have history for this alert
        if alert_key in self._alert_history:
            history = self._alert_history[alert_key]

            # Check cooldown period
            if now - history.last_sent < timedelta(minutes=self.cooldown_minutes):
                return False

            # Check rate limit (alerts per hour)
            window_start = now - timedelta(hours=1)
            if (
                history.last_sent > window_start
                and history.count_in_window >= self.max_alerts_per_hour
            ):
                return False

        return True

    def _record_alert(self, alert: Alert) -> None:
        """Record alert in history for rate limiting."""
        alert_key = f"{alert.severity}:{alert.title}:{alert.dataset}"
        now = datetime.now(UTC)

        if alert_key in self._alert_history:
            history = self._alert_history[alert_key]

            # Reset count if outside window
            window_start = now - timedelta(hours=1)
            if history.last_sent < window_start:
                history.count_in_window = 1
            else:
                history.count_in_window += 1

            history.last_sent = now
        else:
            self._alert_history[alert_key] = AlertHistory(
                alert_key=alert_key,
                last_sent=now,
                count_in_window=1,
            )

    def _get_channels_for_severity(self, severity: AlertSeverity) -> list[str]:
        """Get default channels for a severity level."""
        routing = self.config.alerting.routing

        if severity == AlertSeverity.CRITICAL:
            return routing.critical
        elif severity == AlertSeverity.WARNING:
            return routing.warning
        else:
            return routing.info

    def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> None:
        """Send alert to a specific channel."""
        if channel == AlertChannel.LOG:
            self._send_to_log(alert)
        elif channel == AlertChannel.SLACK:
            self._send_to_slack(alert)
        elif channel == AlertChannel.EMAIL:
            self._send_to_email(alert)
        elif channel == AlertChannel.PAGERDUTY:
            self._send_to_pagerduty(alert)
        else:
            logger.warning(f"Unknown alert channel: {channel}")

    def _send_to_log(self, alert: Alert) -> None:
        """Send alert to logs."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,}[alert.severity]

        logger.log(
            log_level,
            f"ALERT: {alert.title} - {alert.message}",
            extra={
                "alert_severity": alert.severity.value,
                "dataset": alert.dataset,
                "dag_id": alert.dag_id,
                "task_id": alert.task_id,
                "metadata": alert.metadata,},
        )

    def _send_to_slack(self, alert: Alert) -> None:
        """Send alert to Slack via webhook."""
        webhook_url = self.config.slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")

        if not webhook_url:
            logger.warning("Slack webhook URL not configured, skipping Slack alert")
            return

        # Format alert for Slack
        color = {
            AlertSeverity.INFO: "#36a64f",  # Green
            AlertSeverity.WARNING: "#ff9900",  # Orange
            AlertSeverity.CRITICAL: "#ff0000",  # Red}[alert.severity]

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [{"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,},
                    ],
                    "footer": "Boston Pulse Data Pipeline",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }

        # Add optional fields
        if alert.dataset:
            payload["attachments"][0]["fields"].append(
                {"title": "Dataset", "value": alert.dataset, "short": True}
            )

        if alert.dag_id:
            payload["attachments"][0]["fields"].append(
                {"title": "DAG", "value": alert.dag_id, "short": True}
            )

        # Send to Slack
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code != 200:
            raise Exception(f"Slack API error: {response.status_code} - {response.text}")

        logger.info(f"Sent alert to Slack: {alert.title}")

    def _send_to_email(self, alert: Alert) -> None:
        """Send alert via email (placeholder for SendGrid integration)."""
        # This would integrate with SendGrid or another email service
        logger.info(
            f"Email alert (not implemented): {alert.title}",
            extra={"alert": alert.to_dict()},
        )

        # TODO: Implement SendGrid integration
        # Example structure:
        # import sendgrid
        # from sendgrid.helpers.mail import Mail
        #
        # sg = sendgrid.SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
        # message = Mail(
        #     from_email='alerts@bostonpulse.com',
        #     to_emails=recipients,
        #     subject=alert.title,
        #     html_content=self._format_email_body(alert)
        # )
        # response = sg.send(message)

    def _send_to_pagerduty(self, alert: Alert) -> None:
        """Send alert to PagerDuty (placeholder)."""
        logger.info(
            f"PagerDuty alert (not implemented): {alert.title}",
            extra={"alert": alert.to_dict()},
        )

        # TODO: Implement PagerDuty integration
        # Example structure:
        # import pypd
        # pypd.api_key = os.getenv('PAGERDUTY_API_KEY')
        # pypd.EventV2.create(data={
        #     'routing_key': os.getenv('PAGERDUTY_ROUTING_KEY'),
        #     'event_action': 'trigger',
        #     'payload': {
        #         'summary': alert.title,
        #         'severity': alert.severity.value,
        #         'source': 'boston-pulse-pipeline',#     }
        # })


# =============================================================================
# Convenience Functions
# =============================================================================


def send_alert(
    title: str,
    message: str,
    severity: Literal["info", "warning", "critical"] = "info",
    dataset: str | None = None,
    config: Settings | None = None,
    **kwargs,
) -> bool:
    """
    Convenience function to send an alert.

    Args:
        title: Alert title
        message: Alert message
        severity: Severity level
        dataset: Associated dataset
        config: Configuration object
        kwargs: any additional arguments
    Returns:
        True if sent successfully
    """
    manager = AlertManager(config)
    return manager.send_alert(title, message, severity, dataset)
