#!/usr/bin/env bash
# Boston Pulse — Airflow VM startup script (runs once on first boot).
#
# This script:
#   1. Installs Docker + compose plugin + git + openssl
#   2. Clones the public Boston Pulse repo to /opt/boston-pulse
#   3. Syncs DAGs into /opt/airflow/dags
#   4. Generates /opt/airflow/.env with project-specific values
#   5. Builds the prod Airflow image and starts the compose stack
#
# All VM-specific values (project, bucket, admin password, clone URL) come
# from instance metadata set by Terraform. Fernet + webserver secrets are
# generated locally because they never need to leave the VM.

set -euo pipefail
exec > /var/log/boston-pulse-startup.log 2>&1

echo "=== Boston Pulse Airflow VM startup: $(date -u) ==="

META_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
meta() { curl -fsSL -H "Metadata-Flavor: Google" "$META_URL/$1"; }

GCP_PROJECT_ID=$(meta GCP_PROJECT_ID)
GCP_REGION=$(meta GCP_REGION)
GCS_BUCKET=$(meta GCS_BUCKET)
ARTIFACT_BUCKET=$(meta ARTIFACT_BUCKET)
AIRFLOW_ADMIN_PASSWORD=$(meta AIRFLOW_ADMIN_PASSWORD)
GITHUB_CLONE_URL=$(meta GITHUB_CLONE_URL)
GITHUB_BRANCH=$(meta GITHUB_BRANCH || echo main)

echo "Project: $GCP_PROJECT_ID  Bucket: $GCS_BUCKET  Region: $GCP_REGION"

# ─── 1. System packages ───────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg git rsync openssl jq

# Docker Engine + compose plugin (official repo)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker

# ─── 2. Clone the repo ────────────────────────────────────────────────────────
mkdir -p /opt
if [[ ! -d /opt/boston-pulse/.git ]]; then
  git clone --depth=1 --branch "$GITHUB_BRANCH" "$GITHUB_CLONE_URL" /opt/boston-pulse
else
  git -C /opt/boston-pulse fetch --depth=1 origin "$GITHUB_BRANCH"
  git -C /opt/boston-pulse reset --hard "origin/$GITHUB_BRANCH"
fi

# ─── 3. Layout expected by docker-compose.prod.yml ────────────────────────────
mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/mlflow
rsync -av --delete --exclude=__pycache__ --exclude='*.pyc' \
  /opt/boston-pulse/data-pipeline/dags/ /opt/airflow/dags/
if [[ -d /opt/boston-pulse/ml/dags ]]; then
  rsync -av --exclude=__pycache__ --exclude='*.pyc' \
    /opt/boston-pulse/ml/dags/ /opt/airflow/dags/
fi
chown -R 50000:0 /opt/airflow

# ─── 4. Generate /opt/airflow/.env ───────────────────────────────────────────
FERNET_KEY=$(openssl rand -base64 32)
WEBSERVER_SECRET_KEY=$(openssl rand -hex 32)

cat > /opt/airflow/.env <<EOF
BP_ENVIRONMENT=prod
GCP_PROJECT_ID=${GCP_PROJECT_ID}
GCP_REGION=${GCP_REGION}
GCP_BUCKET_NAME=${GCS_BUCKET}
GCS_BUCKET=${GCS_BUCKET}
GOOGLE_CLOUD_PROJECT=${GCP_PROJECT_ID}
ARTIFACT_BUCKET=${ARTIFACT_BUCKET}
MLFLOW_TRACKING_URI=sqlite:////opt/airflow/mlflow/mlflow.db
FERNET_KEY=${FERNET_KEY}
WEBSERVER_SECRET_KEY=${WEBSERVER_SECRET_KEY}
AIRFLOW_ADMIN_PASSWORD=${AIRFLOW_ADMIN_PASSWORD}
SLACK_WEBHOOK_URL=
# Override hardcoded bucket names in data-pipeline/configs/environments/prod.yaml.
# The pydantic-settings loader uses env_prefix=BP_ and nested delimiter __,
# so BP_STORAGE__BUCKETS__MAIN maps to settings.storage.buckets.main.
BP_STORAGE__BUCKETS__MAIN=${GCS_BUCKET}
BP_STORAGE__BUCKETS__DVC=${GCS_BUCKET}
BP_STORAGE__BUCKETS__TEMP=${GCS_BUCKET}
EOF
chmod 600 /opt/airflow/.env

# ─── 5. Build and start Airflow ──────────────────────────────────────────────
cd /opt/boston-pulse

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet || true

echo "Building prod Airflow image (may take a few minutes)..."
docker compose -f docker/docker-compose.prod.yml --env-file /opt/airflow/.env build

echo "Running airflow-init..."
docker compose -f docker/docker-compose.prod.yml --env-file /opt/airflow/.env up airflow-init

echo "Starting webserver + scheduler..."
docker compose -f docker/docker-compose.prod.yml --env-file /opt/airflow/.env up -d --remove-orphans

echo "=== Startup complete: $(date -u) ==="
