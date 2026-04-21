"""
Boston Pulse Backend — Cloud Monitoring Instrumentation.

Emits custom metrics to Google Cloud Monitoring:
  - request_latency_ms: Request latency in milliseconds
  - request_count: Request count by endpoint and status
  - score_served: Risk score values served
  - score_not_found: Count of missing H3 scores

Metrics are prefixed with: custom.googleapis.com/bostonpulse/backend/
"""
from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "bostonpulse")

# Lazy-loaded monitoring client
_monitoring_client = None


def _get_monitoring_client():
    """Lazy-load the Cloud Monitoring client."""
    global _monitoring_client
    if _monitoring_client is None:
        try:
            from google.cloud import monitoring_v3

            _monitoring_client = monitoring_v3.MetricServiceClient()
            logger.info("Cloud Monitoring client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Cloud Monitoring client: {e}")
            _monitoring_client = "disabled"
    return _monitoring_client if _monitoring_client != "disabled" else None


def emit_metric(
    metric_name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> bool:
    """
    Emit a custom metric to Cloud Monitoring.

    Args:
        metric_name: Name of the metric (without prefix)
        value: Metric value (float)
        labels: Optional labels for the metric

    Returns:
        True if metric was emitted successfully, False otherwise
    """
    client = _get_monitoring_client()
    if client is None:
        return False

    labels = labels or {}

    try:
        from google.cloud import monitoring_v3

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/bostonpulse/backend/{metric_name}"

        for k, v in labels.items():
            series.metric.labels[k] = str(v)

        series.resource.type = "global"

        point = monitoring_v3.Point()
        point.value.double_value = float(value)
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]

        client.create_time_series(
            name=f"projects/{PROJECT_ID}",
            time_series=[series],
        )
        return True

    except Exception as e:
        # Never crash the request on metric failure
        logger.debug(f"Failed to emit metric {metric_name}: {e}")
        return False


def emit_request_metrics(
    endpoint: str,
    status_code: int,
    latency_ms: float,
    method: str = "GET",
) -> None:
    """Emit standard request metrics."""
    labels = {
        "endpoint": endpoint,
        "status": str(status_code),
        "method": method,
    }
    emit_metric("request_latency_ms", latency_ms, labels)
    emit_metric("request_count", 1, labels)

    # Track error rate separately
    if status_code >= 400:
        emit_metric("error_count", 1, labels)


def emit_score_metrics(
    risk_score: float,
    risk_tier: str,
    h3_prefix: str,
) -> None:
    """Emit metrics when serving a risk score."""
    emit_metric(
        "score_served",
        risk_score,
        {"risk_tier": risk_tier, "h3_prefix": h3_prefix},
    )


def emit_score_not_found(h3_prefix: str) -> None:
    """Emit metric when H3 score is not found."""
    emit_metric("score_not_found", 1, {"h3_prefix": h3_prefix})


# FastAPI middleware
async def fastapi_monitoring_middleware(request: Any, call_next: Callable) -> Any:
    """FastAPI middleware for request monitoring."""
    start = time.time()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        latency_ms = (time.time() - start) * 1000
        emit_request_metrics(
            endpoint=request.url.path,
            status_code=status,
            latency_ms=latency_ms,
            method=request.method,
        )


# Flask decorator for monitoring
def flask_monitor(f: Callable) -> Callable:
    """Decorator for monitoring Flask endpoints."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from flask import request

        start = time.time()
        status = 500

        try:
            response = f(*args, **kwargs)
            # Handle tuple responses (response, status_code)
            if isinstance(response, tuple):
                status = response[1] if len(response) > 1 else 200
            else:
                status = 200
            return response
        except Exception:
            status = 500
            raise
        finally:
            latency_ms = (time.time() - start) * 1000
            emit_request_metrics(
                endpoint=request.path,
                status_code=status,
                latency_ms=latency_ms,
                method=request.method,
            )

    return wrapper


class FlaskMonitoringMiddleware:
    """WSGI middleware for Flask request monitoring."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def __call__(self, environ: dict, start_response: Callable) -> Any:
        from flask import request

        start = time.time()
        status_code = 500

        def custom_start_response(status: str, headers: list, exc_info: Any = None) -> Any:
            nonlocal status_code
            status_code = int(status.split()[0])
            return start_response(status, headers, exc_info)

        try:
            return self.app(environ, custom_start_response)
        finally:
            latency_ms = (time.time() - start) * 1000
            path = environ.get("PATH_INFO", "/")
            method = environ.get("REQUEST_METHOD", "GET")
            emit_request_metrics(
                endpoint=path,
                status_code=status_code,
                latency_ms=latency_ms,
                method=method,
            )
