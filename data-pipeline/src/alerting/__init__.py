"""
Boston Pulse - Alerting System

Alert management and notification system:
- Multi-channel alerting (Slack, Email, PagerDuty)
- Severity-based routing
- Rate limiting and deduplication

Components:
    - AlertManager: Main alert orchestration
    - SlackNotifier: Slack webhook integration
    - EmailNotifier: SendGrid/SMTP email integration
    - PagerDutyNotifier: PagerDuty integration for critical alerts
"""

from src.alerting.alert_manager import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    send_alert,
)

__version__ = "0.1.0"

__all__ = [
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertChannel",
    "send_alert",
]
