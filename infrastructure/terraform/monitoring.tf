# Boston Pulse - Cloud Monitoring Configuration
#
# Defines:
#   1. Monitoring dashboard for ML pipeline and backend metrics
#   2. Alert policies for latency and drift thresholds
#   3. Notification channels for Slack alerts
#
# Variables, providers, and the terraform block live in versions.tf,
# variables.tf, and providers.tf.

# ============================================================================
# Monitoring Dashboard
# ============================================================================

resource "google_monitoring_dashboard" "boston_pulse_ml" {
  project = var.project_id
  dashboard_json = jsonencode({
    displayName = "Boston Pulse ML & Backend"
    gridLayout = {
      columns = 2
      widgets = [
        # Row 1: API Metrics
        # NOTE: backend custom metrics are emitted as GAUGE/DOUBLE (see
        # backend/app/core/monitoring.py). ALIGN_PERCENTILE_* needs DISTRIBUTION
        # and ALIGN_RATE needs CUMULATIVE, so we use ALIGN_MEAN/ALIGN_MAX/ALIGN_SUM.
        {
          title = "API Request Latency (mean / peak)"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                legendTemplate = "mean"
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner   = "ALIGN_MAX"
                      crossSeriesReducer = "REDUCE_MAX"
                    }
                  }
                }
                legendTemplate = "peak"
              }
            ]
            yAxis = {
              label = "Latency (ms)"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "API Requests by Status (per minute)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_count\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_SUM"
                    groupByFields      = ["metric.labels.status"]
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              legendTemplate = "Status $${metric.labels.status}"
            }]
            yAxis = {
              label = "Requests/min"
              scale = "LINEAR"
            }
          }
        },

        # Row 2: Error Metrics
        {
          title = "API Errors (per minute)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/error_count\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_SUM"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              legendTemplate = "Errors"
            }]
            yAxis = {
              label = "Errors/min"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Score Not Found (per minute)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/score_not_found\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_SUM"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            }]
            yAxis = {
              label = "404s/min"
              scale = "LINEAR"
            }
          }
        },

        # Row 3: ML Drift Metrics
        {
          title = "Feature Drift Share (daily)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/ml/drift_share\""
                  aggregation = {
                    alignmentPeriod  = "86400s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
            }]
            yAxis = {
              label = "Drift Share"
              scale = "LINEAR"
            }
            thresholds = [{
              value = 0.3
              label = "High Drift Threshold"
            }]
          }
        },
        {
          title = "Dataset Drift Detected"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/ml/dataset_drift_detected\""
                  aggregation = {
                    alignmentPeriod  = "86400s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
            }]
            yAxis = {
              label = "Drift (1=Yes, 0=No)"
              scale = "LINEAR"
            }
          }
        },

        # Row 4: Per-Feature Drift
        {
          title = "Per-Feature Drift Score (Top Features)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/ml/feature_drift_score\""
                  aggregation = {
                    alignmentPeriod    = "86400s"
                    perSeriesAligner   = "ALIGN_MEAN"
                    groupByFields      = ["metric.labels.feature"]
                    crossSeriesReducer = "REDUCE_NONE"
                  }
                }
              }
              legendTemplate = "$${metric.labels.feature}"
            }]
            yAxis = {
              label = "Drift Score"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Features Drifted Count"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/ml/n_features_drifted\""
                  aggregation = {
                    alignmentPeriod  = "86400s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
            }]
            yAxis = {
              label = "Count"
              scale = "LINEAR"
            }
          }
        },

        # Row 5: Score Distribution
        {
          title = "Risk Scores Served by Tier"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/score_served\""
                  aggregation = {
                    alignmentPeriod    = "300s"
                    perSeriesAligner   = "ALIGN_COUNT"
                    groupByFields      = ["metric.labels.risk_tier"]
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              legendTemplate = "$${metric.labels.risk_tier}"
            }]
            yAxis = {
              label = "Scores Served"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Average Risk Score Served"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/score_served\""
                  aggregation = {
                    alignmentPeriod    = "300s"
                    perSeriesAligner   = "ALIGN_MEAN"
                    crossSeriesReducer = "REDUCE_MEAN"
                  }
                }
              }
            }]
            yAxis = {
              label = "Risk Score"
              scale = "LINEAR"
            }
          }
        }
      ]
    }
  })
}

# ============================================================================
# Custom Metric Descriptors
# ============================================================================
# Alert policies cannot reference a custom metric that has never been seen
# by Cloud Monitoring. In a fresh project no code has emitted yet, so we must
# declare the metric descriptors up-front. All metrics are emitted as
# DOUBLE/GAUGE (see backend/app/core/monitoring.py and
# data-pipeline/dags/monitoring/ml_drift_monitoring_dag.py).

resource "google_monitoring_metric_descriptor" "request_latency_ms" {
  project      = var.project_id
  type         = "custom.googleapis.com/bostonpulse/backend/request_latency_ms"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "ms"
  display_name = "Backend request latency (ms)"
  description  = "Per-request latency for the Flask backend."

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_metric_descriptor" "error_count" {
  project      = var.project_id
  type         = "custom.googleapis.com/bostonpulse/backend/error_count"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "Backend error count"
  description  = "Count of non-2xx responses from the backend."

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_metric_descriptor" "drift_share" {
  project      = var.project_id
  type         = "custom.googleapis.com/bostonpulse/ml/drift_share"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "ML feature drift share"
  description  = "Fraction of features flagged as drifted by Evidently."

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_metric_descriptor" "dataset_drift_detected" {
  project      = var.project_id
  type         = "custom.googleapis.com/bostonpulse/ml/dataset_drift_detected"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "Dataset drift detected (0/1)"
  description  = "1 if Evidently reports dataset-level drift, 0 otherwise."

  depends_on = [google_project_service.apis]
}

# ============================================================================
# Alert Policies
# ============================================================================

# Alert: High API Latency (peak > 500ms for 5 minutes)
# Note: backend emits request_latency_ms as GAUGE/DOUBLE (see
# backend/app/core/monitoring.py). Cloud Monitoring rejects
# ALIGN_PERCENTILE_* on GAUGE/DOUBLE because percentiles require DISTRIBUTION
# value type, so we alert on per-window peak latency instead of p95.
resource "google_monitoring_alert_policy" "high_latency" {
  project      = var.project_id
  display_name = "Boston Pulse: API peak latency > 500ms"
  combiner     = "OR"

  conditions {
    display_name = "peak latency threshold"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 500
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MAX"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s" # 7 days
  }

  documentation {
    content   = "API peak latency exceeded 500ms for 5 minutes. Check Cloud Run logs and Firestore performance."
    mime_type = "text/markdown"
  }

  depends_on = [google_monitoring_metric_descriptor.request_latency_ms]
}

# Alert: High Error Rate (> 5 errors per minute sustained 5 minutes)
# Note: error_count is emitted as GAUGE/DOUBLE, one point per error event.
# ALIGN_RATE requires CUMULATIVE kind, so we sum points over a 60s window
# and alert when sustained above the threshold.
resource "google_monitoring_alert_policy" "high_error_rate" {
  project      = var.project_id
  display_name = "Boston Pulse: API errors > 5/min"
  combiner     = "OR"

  conditions {
    display_name = "error count threshold"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/backend/error_count\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = "Backend error count exceeded 5 per minute for 5 minutes. Check application logs for errors."
    mime_type = "text/markdown"
  }

  depends_on = [google_monitoring_metric_descriptor.error_count]
}

# Alert: High Drift Detected
resource "google_monitoring_alert_policy" "high_drift" {
  project      = var.project_id
  display_name = "Boston Pulse: High feature drift detected"
  combiner     = "OR"

  conditions {
    display_name = "drift share threshold"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/ml/drift_share\" AND resource.type=\"global\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.3
      aggregations {
        alignment_period   = "86400s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = <<-EOT
      Feature drift share exceeded 30%. This indicates significant distribution shift between training and serving data.

      Actions:
      1. Review the Evidently drift report at gs://${local.ml_artifacts_bucket_name}/monitoring/drift_reports/crime_navigate/latest/report.html
      2. Consider retraining the model with fresh data
      3. Investigate root cause (data pipeline issues, seasonal patterns, etc.)
    EOT
    mime_type = "text/markdown"
  }

  depends_on = [google_monitoring_metric_descriptor.drift_share]
}

# Alert: Dataset Drift Detected
resource "google_monitoring_alert_policy" "dataset_drift" {
  project      = var.project_id
  display_name = "Boston Pulse: Dataset drift detected"
  combiner     = "OR"

  conditions {
    display_name = "dataset drift flag"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/ml/dataset_drift_detected\" AND resource.type=\"global\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.5
      aggregations {
        alignment_period   = "86400s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = "Evidently detected significant dataset-level drift. Model retraining may be required."
    mime_type = "text/markdown"
  }

  depends_on = [google_monitoring_metric_descriptor.dataset_drift_detected]
}

# ============================================================================
# Notification Channels
# ============================================================================

resource "google_monitoring_notification_channel" "slack" {
  count        = var.slack_auth_token != "" ? 1 : 0
  project      = var.project_id
  display_name = "Boston Pulse ML Alerts (Slack)"
  type         = "slack"

  labels = {
    channel_name = var.slack_channel
  }

  sensitive_labels {
    auth_token = var.slack_auth_token
  }
}

# ============================================================================
# Outputs
# ============================================================================

output "dashboard_url" {
  description = "URL to the Cloud Monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.boston_pulse_ml.id}?project=${var.project_id}"
}

output "alert_policies" {
  description = "Created alert policies"
  value = [
    google_monitoring_alert_policy.high_latency.display_name,
    google_monitoring_alert_policy.high_error_rate.display_name,
    google_monitoring_alert_policy.high_drift.display_name,
    google_monitoring_alert_policy.dataset_drift.display_name,
  ]
}
