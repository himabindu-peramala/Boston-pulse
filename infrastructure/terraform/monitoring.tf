# Boston Pulse - Cloud Monitoring Configuration
#
# This Terraform file defines:
#   1. Monitoring dashboard for ML pipeline and backend metrics
#   2. Alert policies for latency and drift thresholds
#   3. Notification channels for Slack alerts
#
# Usage:
#   terraform init
#   terraform plan -var="slack_auth_token=xoxb-..."
#   terraform apply -var="slack_auth_token=xoxb-..."

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "bostonpulse"
}

variable "slack_auth_token" {
  description = "Slack Bot OAuth Token for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_channel" {
  description = "Slack channel for ML alerts"
  type        = string
  default     = "#ml-alerts"
}

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
        {
          title = "API Request Latency (p50/p95)"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_50"
                    }
                  }
                }
                legendTemplate = "p50"
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_95"
                    }
                  }
                }
                legendTemplate = "p95"
              }
            ]
            yAxis = {
              label = "Latency (ms)"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "API Request Rate by Status"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_count\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_RATE"
                    groupByFields      = ["metric.labels.status"]
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              legendTemplate = "Status $${metric.labels.status}"
            }]
            yAxis = {
              label = "Requests/sec"
              scale = "LINEAR"
            }
          }
        },

        # Row 2: Error Metrics
        {
          title = "API Error Rate (non-2xx)"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/error_count\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              legendTemplate = "Errors"
            }]
            yAxis = {
              label = "Errors/sec"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Score Not Found Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/bostonpulse/backend/score_not_found\""
                  aggregation = {
                    alignmentPeriod    = "60s"
                    perSeriesAligner   = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            }]
            yAxis = {
              label = "404s/sec"
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
# Alert Policies
# ============================================================================

# Alert: High API Latency (p95 > 500ms for 5 minutes)
resource "google_monitoring_alert_policy" "high_latency" {
  project      = var.project_id
  display_name = "Boston Pulse: API p95 latency > 500ms"
  combiner     = "OR"

  conditions {
    display_name = "p95 latency threshold"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/backend/request_latency_ms\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 500
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_95"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s" # 7 days
  }

  documentation {
    content   = "API p95 latency exceeded 500ms for 5 minutes. Check Cloud Run logs and Firestore performance."
    mime_type = "text/markdown"
  }
}

# Alert: High Error Rate (> 5% errors for 5 minutes)
resource "google_monitoring_alert_policy" "high_error_rate" {
  project      = var.project_id
  display_name = "Boston Pulse: API error rate > 5%"
  combiner     = "OR"

  conditions {
    display_name = "error rate threshold"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/bostonpulse/backend/error_count\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = var.slack_auth_token != "" ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = "API error rate exceeded 5% for 5 minutes. Check application logs for errors."
    mime_type = "text/markdown"
  }
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
      1. Review the Evidently drift report at gs://boston-pulse-mlflow-artifacts/monitoring/drift_reports/crime_navigate/latest/report.html
      2. Consider retraining the model with fresh data
      3. Investigate root cause (data pipeline issues, seasonal patterns, etc.)
    EOT
    mime_type = "text/markdown"
  }
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
