# Boston Pulse — core infrastructure.
#
# Provisions:
#   - GCP API enablement
#   - Artifact Registry repos (Docker format for both images + model artifacts)
#   - GCS buckets (data-pipeline, ml-artifacts)
#   - Firestore (Native mode)
#   - ML service account + IAM bindings
#   - GitHub Workload Identity Federation for CI/CD auth
#
# The Cloud Run training job is created/updated by CI/CD after it pushes a
# real image to Artifact Registry — Terraform never references placeholders.

# ============================================================================
# APIs
# ============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# ============================================================================
# Artifact Registry
# ============================================================================

resource "google_artifact_registry_repository" "ml_images" {
  project       = var.project_id
  location      = var.region
  repository_id = "ml-images"
  format        = "DOCKER"
  description   = "ML training container images"

  depends_on = [google_project_service.apis]
}

# Docker format — ml/shared/artifact_registry.py writes OCI image manifests here.
resource "google_artifact_registry_repository" "ml_models" {
  project       = var.project_id
  location      = var.region
  repository_id = "ml-models"
  format        = "DOCKER"
  description   = "ML model artifacts (OCI format, written by shared.artifact_registry)"

  depends_on = [google_project_service.apis]
}

# ============================================================================
# GCS buckets
# ============================================================================

resource "google_storage_bucket" "data_pipeline" {
  project                     = var.project_id
  name                        = local.data_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "ml_artifacts" {
  project                     = var.project_id
  name                        = local.ml_artifacts_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

# ============================================================================
# Firestore (Native mode)
# ============================================================================

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.firestore_location
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.apis]

  # GCP does not let you delete the default database.
  lifecycle {
    prevent_destroy = false
  }
}

# ============================================================================
# Service account for Cloud Run Job + backend service
# ============================================================================

resource "google_service_account" "ml_runner" {
  project      = var.project_id
  account_id   = var.service_account_name
  display_name = "Boston Pulse ML / Backend"
  description  = "Used by Cloud Run jobs, the backend service, and Airflow to access GCS/Firestore/AR."

  depends_on = [google_project_service.apis]
}

locals {
  ml_runner_roles = toset([
    "roles/artifactregistry.writer",
    "roles/artifactregistry.reader",
    "roles/storage.objectAdmin",
    "roles/datastore.user",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
    "roles/run.invoker",
  ])
}

resource "google_project_iam_member" "ml_runner_bindings" {
  for_each = local.ml_runner_roles

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.ml_runner.email}"
}

# ============================================================================
# Workload Identity Federation for GitHub Actions
# ============================================================================

data "google_project" "current" {
  project_id = var.project_id
}

resource "google_iam_workload_identity_pool" "github" {
  count = var.github_repository != "" ? 1 : 0

  project                   = var.project_id
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions Pool"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count = var.github_repository != "" ? 1 : 0

  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Actions Provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repository}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "wif_binding" {
  count = var.github_repository != "" ? 1 : 0

  service_account_id = google_service_account.ml_runner.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github[0].workload_identity_pool_id}/attribute.repository/${var.github_repository}"
}

# ============================================================================
# Cloud Run Job (training only)
# ============================================================================

# The Cloud Run training job itself is created by the CI/CD pipeline
# (.github/workflows/ml.yml) after the real ml-training image is built and
# pushed to Artifact Registry. Terraform intentionally does not own the job
# so every artifact it manages is real (no placeholder images).
