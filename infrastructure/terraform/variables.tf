variable "project_id" {
  description = "GCP project ID where all resources will be created."
  type        = string
}

variable "region" {
  description = "Primary GCP region for Cloud Run, Artifact Registry, and GCS."
  type        = string
  default     = "us-east1"
}

variable "firestore_location" {
  description = "Firestore multi-region location (nam5 = US, eur3 = EU)."
  type        = string
  default     = "nam5"
}

variable "service_account_name" {
  description = "Account ID (prefix of the email) for the ML service account."
  type        = string
  default     = "boston-pulse-data-pipeline"
}

variable "github_repository" {
  description = "GitHub repository in owner/repo format for Workload Identity Federation."
  type        = string
  default     = ""
}

variable "training_job_name" {
  description = "Cloud Run Job name used for ML training."
  type        = string
  default     = "ml-training-job"
}

# -------- GCS bucket names --------
# GCS bucket names are globally unique. We default to "${project_id}-<suffix>"
# so fresh projects never collide with an existing team's buckets. Override
# these only if you are importing/adopting pre-existing buckets.

variable "data_bucket_name" {
  description = "GCS bucket for data-pipeline artifacts. Empty = derive as '<project_id>-data-pipeline'."
  type        = string
  default     = ""
}

variable "ml_artifacts_bucket_name" {
  description = "GCS bucket for ML artifacts (MLflow, model registry, baselines). Empty = derive as '<project_id>-mlflow-artifacts'."
  type        = string
  default     = ""
}

# -------- Monitoring --------

variable "slack_auth_token" {
  description = "Slack Bot OAuth token. Leave empty to skip creating Slack notification channel + alert policies."
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_channel" {
  description = "Slack channel for alerts."
  type        = string
  default     = "#ml-alerts"
}

locals {
  enable_slack_alerts = length(var.slack_auth_token) > 0

  data_bucket_name         = length(var.data_bucket_name) > 0 ? var.data_bucket_name : "${var.project_id}-data-pipeline"
  ml_artifacts_bucket_name = length(var.ml_artifacts_bucket_name) > 0 ? var.ml_artifacts_bucket_name : "${var.project_id}-mlflow-artifacts"
}
