# Boston Pulse Infrastructure (Terraform)

Provisions the infrastructure for Boston Pulse in one GCP project:

| Resource | File |
|---|---|
| GCP APIs (12) | `main.tf` |
| Artifact Registry (`ml-images`, `ml-models`, both Docker) | `main.tf` |
| GCS buckets (`boston-pulse-data-pipeline`, `boston-pulse-mlflow-artifacts`) | `main.tf` |
| Firestore `(default)` (Native mode) | `main.tf` |
| Service account `boston-pulse-data-pipeline@...` + 8 IAM roles | `main.tf` |
| Workload Identity Federation (GitHub pool + provider + SA binding) | `main.tf` |
| Cloud Run Job `ml-training-job` (training only) | `main.tf` |
| Cloud Monitoring dashboard (9 widgets) | `monitoring.tf` |
| 4 alert policies + optional Slack notification channel | `monitoring.tf` |

The image build/push still happens outside Terraform (Cloud Build or GitHub Actions), but Terraform now creates the training job and WIF wiring.

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
