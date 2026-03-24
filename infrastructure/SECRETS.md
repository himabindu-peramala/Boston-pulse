# Boston Pulse - Secrets & Environment Variables

This document defines all secrets and environment variables used across the Boston Pulse ML pipeline.

## GitHub Actions Secrets

Configure these in your repository settings under **Settings > Secrets and variables > Actions**.

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `WIF_PROVIDER` | Workload Identity Federation provider path | `projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `WIF_SERVICE_ACCOUNT` | Service account email for WIF | `boston-pulse-data-pipeline@bostonpulse.iam.gserviceaccount.com` |
| `GCP_PROJECT_ID` | GCP project ID | `bostonpulse` |
| `GCS_BUCKET` | Primary GCS bucket for data pipeline | `boston-pulse-data-pipeline` |
| `AIRFLOW_URL` | Airflow webserver URL (with port) | `http://YOUR_VM_IP:8080` |
| `AIRFLOW_USERNAME` | Airflow admin username | `airflow` |
| `AIRFLOW_PASSWORD` | Airflow admin password | (your password) |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook for alerts | `https://hooks.slack.com/services/XXX/YYY/ZZZ` |

## Environment Variables by Context

### Airflow Docker Containers (VM)

Set in `docker/docker-compose.prod.yml` or `.env` file:

| Variable | Description | Used By |
|----------|-------------|---------|
| `GCS_BUCKET` | Data pipeline bucket | DAGs, feature loaders |
| `GCP_PROJECT_ID` | GCP project | All GCP clients |
| `GOOGLE_CLOUD_PROJECT` | Alias for GCP project | Some GCP libraries |
| `MLFLOW_TRACKING_URI` | MLflow backend URI | Training DAG |
| `SLACK_WEBHOOK_URL` | Slack alerts | Alerting module |
| `FERNET_KEY` | Airflow encryption key | Airflow core |
| `WEBSERVER_SECRET_KEY` | Airflow webserver secret | Airflow webserver |
| `AIRFLOW_ADMIN_PASSWORD` | Initial admin password | Airflow init |

### ML Training Container

Passed via DAG's DockerOperator environment:

| Variable | Description |
|----------|-------------|
| `GCS_BUCKET` | Data pipeline bucket |
| `GCP_PROJECT_ID` | GCP project |
| `GOOGLE_CLOUD_PROJECT` | GCP project (alias) |
| `MLFLOW_TRACKING_URI` | MLflow backend |
| `GIT_SHA` | Git commit SHA (from CI) |
| `ML_IMAGE` | Training image URI (from CI) |
| `SLACK_WEBHOOK_URL` | Slack alerts |

### GitHub Actions Workflow

Available as `${{ secrets.NAME }}` or environment variables:

| Variable | Source | Description |
|----------|--------|-------------|
| `GITHUB_SHA` | Built-in | Current commit SHA |
| `GITHUB_REF_NAME` | Built-in | Branch name |
| `GITHUB_RUN_ID` | Built-in | Workflow run ID |

## Naming Conventions

### Bucket Names

| Purpose | Variable | Bucket Name |
|---------|----------|-------------|
| Data pipeline artifacts | `GCS_BUCKET` | `boston-pulse-data-pipeline` |
| MLflow artifacts | `artifact_bucket` (config) | `boston-pulse-mlflow-artifacts` |
| Deployment manifests | N/A | `boston-pulse-deployments` |

### Artifact Registry

| Repository | Format | Purpose |
|------------|--------|---------|
| `ml-models` | Generic | Model artifacts (.lgb, metadata, SHAP) |
| `ml-images` | Docker | Training container images |

## How to Obtain Values

### WIF_PROVIDER

Run after creating the identity pool:

```bash
PROJECT_NUMBER=$(gcloud projects describe bostonpulse --format='value(projectNumber)')
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
```

### WIF_SERVICE_ACCOUNT

This is your existing service account:

```bash
echo "boston-pulse-data-pipeline@bostonpulse.iam.gserviceaccount.com"
```

### FERNET_KEY

Generate a new Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### WEBSERVER_SECRET_KEY

Generate a random secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Local Development (.secrets file for `act`)

For local testing with `act`, create a `.secrets` file:

```
WIF_PROVIDER=projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider
WIF_SERVICE_ACCOUNT=boston-pulse-data-pipeline@bostonpulse.iam.gserviceaccount.com
GCP_PROJECT_ID=bostonpulse
GCS_BUCKET=boston-pulse-data-pipeline
AIRFLOW_URL=http://YOUR_VM_IP:8080
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=your_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

Run with: `act -s .secrets`

## Security Notes

1. **Never commit secrets** to version control
2. **Rotate credentials** periodically
3. **Use WIF** instead of service account keys when possible
4. **Limit IAM permissions** to minimum required roles
5. **Audit access** via Cloud Audit Logs
