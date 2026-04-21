# Boston Pulse — Airflow VM (optional).
#
# Creates a single GCE VM that clones the public repo, builds the prod
# Airflow image, and runs the data-pipeline Airflow stack. Intended for
# fresh-project demos so the data-pipeline bucket gets populated without
# any manual SSH work. Controlled by var.enable_airflow_vm.

resource "random_password" "airflow_admin" {
  count = var.enable_airflow_vm ? 1 : 0

  length  = 20
  special = false
}

resource "google_compute_address" "airflow" {
  count = var.enable_airflow_vm ? 1 : 0

  name    = "airflow-vm-ip"
  project = var.project_id
  region  = var.region

  depends_on = [google_project_service.apis]
}

resource "google_compute_firewall" "airflow_ui" {
  count = var.enable_airflow_vm ? 1 : 0

  name    = "allow-airflow-ui"
  project = var.project_id
  network = "default"

  source_ranges = var.airflow_allowed_cidrs
  target_tags   = ["airflow-vm"]

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  depends_on = [google_project_service.apis]
}

resource "google_compute_instance" "airflow" {
  count = var.enable_airflow_vm ? 1 : 0

  name         = var.vm_name
  project      = var.project_id
  zone         = var.vm_zone
  machine_type = var.vm_machine_type
  tags         = ["airflow-vm"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = var.vm_disk_gb
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = google_compute_address.airflow[0].address
    }
  }

  service_account {
    email  = google_service_account.ml_runner.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    GCP_PROJECT_ID         = var.project_id
    GCP_REGION             = var.region
    GCS_BUCKET             = google_storage_bucket.data_pipeline.name
    ARTIFACT_BUCKET        = google_storage_bucket.ml_artifacts.name
    AIRFLOW_ADMIN_PASSWORD = random_password.airflow_admin[0].result
    GITHUB_CLONE_URL       = var.github_clone_url
    GITHUB_BRANCH          = var.github_branch
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  allow_stopping_for_update = true

  depends_on = [
    google_project_service.apis,
    google_service_account.ml_runner,
    google_project_iam_member.ml_runner_bindings,
    google_storage_bucket.data_pipeline,
    google_storage_bucket.ml_artifacts,
  ]
}
