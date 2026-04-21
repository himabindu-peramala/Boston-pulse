# Boston Pulse Infrastructure (Terraform)

> **The easiest way to apply this module is `make demo-up` from the repo
> root** — it loads config from `.env.demo`, creates the GCP project if
> needed, links billing, and runs `terraform apply` with the right variables.
> See the [root README](../../README.md#quickstart-one-command-demo-on-a-fresh-gcp-project)
> for the full flow. Use the raw `terraform` commands below only if you need
> more control.

Provisions the infrastructure for Boston Pulse in one GCP project:

| Resource | File |
|---|---|
| GCP APIs (12) | `main.tf` |
| Artifact Registry (`ml-images`, `ml-models`, both Docker) | `main.tf` |
| GCS buckets (`boston-pulse-data-pipeline`, `boston-pulse-mlflow-artifacts`) | `main.tf` |
| Firestore `(default)` (Native mode) | `main.tf` |
| Service account `boston-pulse-data-pipeline@...` + 8 IAM roles | `main.tf` |
| Workload Identity Federation (GitHub pool + provider + SA binding) | `main.tf` |
| Airflow GCE VM + static IP + firewall (`0.0.0.0/0:8080` by default) | `vm.tf` |
| Cloud Monitoring dashboard (9 widgets) | `monitoring.tf` |
| 4 alert policies + optional Slack notification channel | `monitoring.tf` |

The Cloud Run training job is created by GitHub Actions (`.github/workflows/ml.yml`) after the real `ml-training` image is pushed to Artifact Registry — Terraform never references placeholder images.

The Airflow VM runs `infrastructure/terraform/startup.sh` on first boot, which installs Docker, clones the public repo, generates `/opt/airflow/.env` from VM metadata, builds the prod Airflow image, and starts the data-pipeline stack. Disable with `-var="enable_airflow_vm=false"`.

---

## Quickstart

```bash
cd infrastructure/terraform

terraform init

# Preview
terraform plan -var="project_id=$GCP_PROJECT_ID"

# Apply — creates ~25 resources in ~2 min
terraform apply -auto-approve -var="project_id=$GCP_PROJECT_ID"

# Optionally build/push training image
cd ../..
gcloud builds submit . \
  --config=ml/cloudbuild.yaml \
  --substitutions="_IMAGE=$(cd infrastructure/terraform && terraform output -raw training_image)" \
  --project="$GCP_PROJECT_ID"
```

## With Slack alerts

```bash
terraform apply -auto-approve \
  -var="project_id=$GCP_PROJECT_ID" \
  -var="slack_auth_token=xoxb-..."
```

Without the token, the dashboard and alert policies still get created; only the Slack notification channel is skipped.

## Outputs

After `terraform apply`:

```bash
terraform output service_account_email   # used by Cloud Run + backend
terraform output training_image          # Docker image URI for Cloud Run Job
terraform output cloud_run_training_job  # Cloud Run Job name
terraform output wif_provider            # GitHub Actions WIF provider path
terraform output wif_service_account     # GitHub Actions SA
terraform output data_bucket             # GCS bucket for pipeline data
terraform output ml_artifacts_bucket     # GCS bucket for ML artifacts
terraform output dashboard_url           # direct link to monitoring dashboard

terraform output airflow_url                    # http://<ip>:8080
terraform output airflow_admin_user             # admin
terraform output -raw airflow_admin_password    # generated password (sensitive)
```

## Teardown

```bash
terraform destroy -var="project_id=$GCP_PROJECT_ID"
```

**Note:** Firestore `(default)` cannot be deleted by GCP policy — `terraform destroy` will remove it from state but the DB remains in the project. To fully clean, delete the GCP project or drop Firestore collections manually.

## Variables

| Variable | Default | Description |
|---|---|---|
| `project_id` | *required* | GCP project to provision into |
| `region` | `us-east1` | Region for Cloud Run, AR, GCS |
| `firestore_location` | `nam5` | Firestore multi-region (nam5 = US) |
| `service_account_name` | `boston-pulse-data-pipeline` | Account ID for the ML SA |
| `github_repository` | `""` | `owner/repo` for WIF setup (set for CI/CD auth) |
| `training_job_name` | `ml-training-job` | Cloud Run training job name |
| `enable_airflow_vm` | `true` | Create the GCE VM running the Airflow data-pipeline |
| `vm_name` | `airflow-vm` | Instance name |
| `vm_zone` | `us-east1-b` | VM zone |
| `vm_machine_type` | `e2-standard-4` | Machine type (≥ 4 vCPU recommended for the image build) |
| `vm_disk_gb` | `50` | Boot disk size in GB |
| `github_clone_url` | Boston-pulse HTTPS URL | Public repo the VM clones on boot |
| `github_branch` | `main` | Branch the VM checks out |
| `airflow_allowed_cidrs` | `["0.0.0.0/0"]` | Source CIDRs allowed to hit port 8080 |
| `slack_auth_token` | `""` | Slack Bot OAuth token (optional) |
| `slack_channel` | `#ml-alerts` | Slack channel for alerts |

## State

Local state (`terraform.tfstate`) is used for simplicity. For team work, add a GCS backend:

```hcl
# versions.tf
terraform {
  backend "gcs" {
    bucket = "boston-pulse-tf-state"
    prefix = "infra"
  }
}
```
