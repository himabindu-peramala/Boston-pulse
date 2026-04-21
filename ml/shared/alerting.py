"""
Boston Pulse ML - Alerting.

Slack alerts for ML pipeline events.
Mirrors the data-pipeline alerting pattern — uses same webhook.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_webhook_url() -> str | None:
    """Get Slack webhook URL from environment."""
    return os.getenv("SLACK_WEBHOOK_URL")


def _send_slack_message(message: dict[str, Any]) -> bool:
    """
    Send a message to Slack.

    Args:
        message: Slack message payload

    Returns:
        True if sent successfully
    """
    import requests

    webhook = _get_webhook_url()
    if not webhook:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return False

    try:
        response = requests.post(webhook, json=message, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Slack alert failed: {e}")
        return False


def alert_training_start(
    dataset: str,
    execution_date: str,
    dag_id: str,
) -> None:
    """Send alert when training starts."""
    message = {
        "text": f"🚀 *{dag_id}* started for `{execution_date}`",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"ML Training Started — {dataset}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {dataset}"},
                    {"type": "mrkdwn", "text": f"*Date:* {execution_date}"},
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_training_complete(
    dataset: str,
    execution_date: str,
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    bias_result: dict[str, Any],
    score_result: dict[str, Any],
    publish_result: dict[str, Any],
    dag_id: str,
) -> None:
    """Send alert when training completes successfully."""
    rmse_val = val_result.get("rmse_val", "N/A")
    if isinstance(rmse_val, float):
        rmse_val = f"{rmse_val:.4f}"

    bias_passed = bias_result.get("passed", False)
    bias_status = "PASSED ✅" if bias_passed else "FAILED ❌"

    h3_cells = score_result.get("h3_cells", 0)
    rows_upserted = publish_result.get("rows_upserted", 0)

    message = {
        "text": f"✅ *{dag_id}* complete for `{execution_date}`",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Navigate ML Training — {execution_date}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*RMSE (holdout):* {rmse_val}"},
                    {"type": "mrkdwn", "text": f"*Bias gate:* {bias_status}"},
                    {"type": "mrkdwn", "text": f"*Cells scored:* {h3_cells:,}"},
                    {"type": "mrkdwn", "text": f"*Firestore rows:* {rows_upserted:,}"},
                ],
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"DAG: `{dag_id}` | Dataset: `{dataset}`"},
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_gate_failure(
    dataset: str,
    execution_date: str,
    gate_name: str,
    error_message: str,
    dag_id: str,
) -> None:
    """Send alert when a gate fails."""
    message = {
        "text": f"❌ *{dag_id}* GATE FAILED: {gate_name}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🚨 ML Gate Failed — {gate_name}"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Dataset:* {dataset}\n*Date:* {execution_date}\n\n*Error:*\n```{error_message[:500]}```",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⚠️ Model NOT pushed to registry. Production model unchanged.",
                    },
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_model_pushed(
    dataset: str,
    execution_date: str,
    version: str,
    model_uri: str,
    rmse_val: float,
    dag_id: str,
) -> None:
    """Send alert when model is pushed to registry."""
    message = {
        "text": f"📦 Model pushed to registry: {version}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Model Published — {dataset}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Version:* {version}"},
                    {"type": "mrkdwn", "text": f"*RMSE:* {rmse_val:.4f}"},
                    {"type": "mrkdwn", "text": f"*URI:* `{model_uri}`"},
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_scores_published(
    dataset: str,
    execution_date: str,
    rows_upserted: int,
    h3_cells: int,
    duration_seconds: float,
    dag_id: str,
) -> None:
    """Send alert when scores are published to Firestore."""
    message = {
        "text": f"📊 Scores published to Firestore: {rows_upserted:,} rows",
        "blocks": [
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Rows upserted:* {rows_upserted:,}"},
                    {"type": "mrkdwn", "text": f"*H3 cells:* {h3_cells:,}"},
                    {"type": "mrkdwn", "text": f"*Duration:* {duration_seconds:.1f}s"},
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_model_promoted(
    dataset: str,
    execution_date: str,
    version: str,
    comparison: dict[str, Any],
    dag_id: str,
) -> None:
    """Send alert when model is promoted to production (Gate 3 passed)."""
    delta_pct = comparison.get("delta_pct")
    # Handle potential mock objects in tests
    try:
        delta_str = f"{float(delta_pct):+.2f}%" if delta_pct is not None else "N/A"
    except (TypeError, ValueError):
        delta_str = "N/A"

    prod_rmse = comparison.get("production_rmse")
    try:
        prod_rmse_str = f"{float(prod_rmse):.4f}" if prod_rmse is not None else "N/A"
    except (TypeError, ValueError):
        prod_rmse_str = "N/A"

    candidate_rmse = comparison.get("candidate_rmse", 0)
    try:
        candidate_rmse = float(candidate_rmse)
    except (TypeError, ValueError):
        candidate_rmse = 0.0

    message = {
        "text": f"🚀 Model {version} promoted to production",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Model Promoted — {dataset}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Version:* {version}"},
                    {"type": "mrkdwn", "text": f"*Candidate RMSE:* {candidate_rmse:.4f}"},
                    {"type": "mrkdwn", "text": f"*Production RMSE:* {prod_rmse_str}"},
                    {"type": "mrkdwn", "text": f"*Delta:* {delta_str}"},
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"✅ Gate 3 passed: {comparison.get('reason', '')}",
                    },
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_promotion_skipped(
    dataset: str,
    execution_date: str,
    version: str,
    comparison: dict[str, Any],
    dag_id: str,
) -> None:
    """Send alert when model promotion is skipped (Gate 3 failed)."""
    delta_pct = comparison.get("delta_pct")
    # Handle potential mock objects in tests
    try:
        delta_str = f"{float(delta_pct):+.2f}%" if delta_pct is not None else "N/A"
    except (TypeError, ValueError):
        delta_str = "N/A"

    prod_rmse = comparison.get("production_rmse")
    try:
        prod_rmse_str = f"{float(prod_rmse):.4f}" if prod_rmse is not None else "N/A"
    except (TypeError, ValueError):
        prod_rmse_str = "N/A"

    prod_version = comparison.get("production_version", "unknown")
    candidate_rmse = comparison.get("candidate_rmse", 0)
    try:
        candidate_rmse = float(candidate_rmse)
    except (TypeError, ValueError):
        candidate_rmse = 0.0

    message = {
        "text": f"⏸️ Model {version} NOT promoted — Gate 3 failed",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Promotion Skipped — {dataset}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Candidate Version:* {version}"},
                    {"type": "mrkdwn", "text": f"*Candidate RMSE:* {candidate_rmse:.4f}"},
                    {"type": "mrkdwn", "text": f"*Production RMSE:* {prod_rmse_str}"},
                    {"type": "mrkdwn", "text": f"*Delta:* {delta_str}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reason:* {comparison.get('reason', 'unknown')}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"⚠️ Candidate stays in staging. "
                            f"Production model `{prod_version}` unchanged. "
                            f"Firestore scores NOT updated."
                        ),
                    },
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_drift_report(summary: dict[str, Any]) -> None:
    """Send Slack alert with daily drift monitoring results."""
    model = summary.get("model", "unknown")
    execution_date = summary.get("execution_date", "unknown")
    n_drifted = summary.get("n_features_drifted", 0)
    n_total = summary.get("n_features_total", 0)
    drift_share = summary.get("drift_share", 0)
    dataset_drift = summary.get("dataset_drift_detected", False)
    html_uri = summary.get("html_gcs_uri", "")

    # Determine severity
    if dataset_drift or drift_share > 0.3:
        emoji = "🚨"
        severity = "HIGH"
        _color = "#dc3545"
    elif drift_share > 0.1:
        emoji = "⚠️"
        severity = "MEDIUM"
        _color = "#ffc107"
    else:
        emoji = "✅"
        severity = "LOW"
        _color = "#28a745"

    # Find top drifted features
    per_feature = summary.get("per_feature", {})
    drifted_features = [
        (k, v.get("drift_score", 0))
        for k, v in per_feature.items()
        if v.get("drift_detected", False)
    ]
    drifted_features.sort(key=lambda x: x[1], reverse=True)
    top_drifted = drifted_features[:5]

    top_drifted_text = (
        "\n".join([f"  • `{f}`: {s:.3f}" for f, s in top_drifted]) if top_drifted else "  None"
    )

    message = {
        "text": f"{emoji} Drift Report: {model} — {severity}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Daily Drift Report — {model}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Date:* {execution_date}"},
                    {"type": "mrkdwn", "text": f"*Severity:* {severity}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Features Drifted:* {n_drifted}/{n_total}",
                    },
                    {"type": "mrkdwn", "text": f"*Drift Share:* {drift_share:.1%}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Dataset Drift Detected:* {'Yes 🚨' if dataset_drift else 'No'}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Top Drifted Features:*\n{top_drifted_text}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reference Rows:* {summary.get('reference_rows', 'N/A'):,}\n"
                    f"*Current Rows:* {summary.get('current_rows', 'N/A'):,}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📊 <{html_uri}|View Full Evidently Report>",
                    },
                ],
            },
        ],
    }
    _send_slack_message(message)


def alert_retrain_triggered(
    model: str,
    reasons: list[str],
    summary: dict[str, Any],
) -> None:
    """Send Slack alert when drift monitoring triggers an automatic retrain."""
    drift_share = summary.get("drift_share", 0)

    # Format reasons as bullet points
    reasons_text = "\n".join(f"• {r}" for r in reasons) if reasons else "• Unknown"

    # Get top drifted features
    per_feature = summary.get("per_feature", {})
    drifted_features = [
        (k, v.get("drift_score", 0))
        for k, v in per_feature.items()
        if v.get("drift_detected", False)
    ]
    drifted_features.sort(key=lambda x: x[1], reverse=True)
    top_features = drifted_features[:3]
    top_features_text = (
        ", ".join([f"`{f}` ({s:.2f})" for f, s in top_features]) if top_features else "N/A"
    )

    message = {
        "text": f"🔄 Automatic retrain triggered: {model}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🔄 Automatic Retrain Triggered",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Model:* {model}"},
                    {"type": "mrkdwn", "text": f"*Date:* {summary.get('execution_date', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Drift Share:* {drift_share:.1%}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Features Drifted:* {summary.get('n_features_drifted', 0)}/{summary.get('n_features_total', 0)}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Trigger Reasons:*\n{reasons_text}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Top Drifted Features:* {top_features_text}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⚙️ GitHub Actions workflow `ml.yml` has been dispatched",
                    },
                ],
            },
        ],
    }
    _send_slack_message(message)
