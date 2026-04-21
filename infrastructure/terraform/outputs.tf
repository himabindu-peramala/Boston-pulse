output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}

output "service_account_email" {
  description = "Service account used by Cloud Run Job + backend."
  value       = google_service_account.ml_runner.email
}

output "ml_images_repo" {
  description = "Docker repo for training images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_images.repository_id}"
}

output "ml_models_repo" {
  description = "Docker repo for model artifacts (OCI)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_models.repository_id}"
}

output "training_image" {
  description = "Fully qualified :latest tag for the training image."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_images.repository_id}/ml-training:latest"
}

output "data_bucket" {
  value = google_storage_bucket.data_pipeline.name
}

output "ml_artifacts_bucket" {
  value = google_storage_bucket.ml_artifacts.name
}

output "cloud_run_training_job" {
  description = "Expected Cloud Run Job name (created by CI/CD after image push)."
  value       = var.training_job_name
}

output "wif_provider" {
  description = "WIF provider resource path for GitHub Actions auth."
  value       = var.github_repository != "" ? "projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github[0].workload_identity_pool_id}/providers/${google_iam_workload_identity_pool_provider.github[0].workload_identity_pool_provider_id}" : ""
}

output "wif_service_account" {
  description = "Service account for GitHub Actions WIF auth."
  value       = google_service_account.ml_runner.email
}
