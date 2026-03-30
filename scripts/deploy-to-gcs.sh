#!/usr/bin/env bash
# scripts/deploy-to-gcs.sh
#
# Upload DAGs and ML code to GCS deployment bucket.
# Called by CI after tests pass.
#
# Usage:
#   ./scripts/deploy-to-gcs.sh
#
# Environment (required):
#   GCS_DEPLOY_BUCKET   Deployment bucket
#   GITHUB_SHA          Git commit SHA
#   GITHUB_RUN_ID       Workflow run ID
#
# Environment (optional):
#   DEPLOY_PREFIX       GCS prefix (default: airflow)

set -euo pipefail

# Required environment
: "${GCS_DEPLOY_BUCKET:?GCS_DEPLOY_BUCKET is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"

# Optional with defaults
DEPLOY_PREFIX="${DEPLOY_PREFIX:-airflow}"

# Paths
GCS_BASE="gs://${GCS_DEPLOY_BUCKET}/${DEPLOY_PREFIX}"
MANIFEST_PATH="${GCS_BASE}/manifest.json"
DAGS_PATH="${GCS_BASE}/dags/"
ML_PATH="${GCS_BASE}/ml/"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Find repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

log "Deploying to GCS"
log "  Bucket: $GCS_DEPLOY_BUCKET"
log "  Prefix: $DEPLOY_PREFIX"
log "  SHA: $GITHUB_SHA"

# Upload DAGs from both data-pipeline and ml
log "Uploading DAGs..."

# Data pipeline DAGs
if [[ -d "data-pipeline/dags" ]]; then
    gsutil -m rsync -r -d "data-pipeline/dags/" "${DAGS_PATH}data-pipeline/" || {
        log "WARNING: data-pipeline DAGs sync failed"
    }
fi

# Upload ML source code (for training container and shared modules)
log "Uploading ML code..."
if [[ -d "ml" ]]; then
    # Exclude tests, cache, and build artifacts
    gsutil -m rsync -r -x '.*__pycache__.*|.*\.pyc|.*\.egg-info.*|tests/.*|\.pytest_cache.*' \
        "ml/" "${ML_PATH}" || {
        log "WARNING: ML code sync failed"
    }
fi

# Create and upload manifest
log "Creating deployment manifest..."
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

MANIFEST=$(cat <<EOF
{
  "git_sha": "${GITHUB_SHA}",
  "timestamp": "${TIMESTAMP}",
  "workflow_run_id": "${GITHUB_RUN_ID}",
  "deployed_by": "github-actions",
  "components": {
    "dags": "${DAGS_PATH}",
    "ml": "${ML_PATH}"
  }
}
EOF
)

echo "$MANIFEST" | gsutil cp - "$MANIFEST_PATH"

log "Deployment complete"
log "  Manifest: $MANIFEST_PATH"
log "  DAGs: $DAGS_PATH"
log "  ML: $ML_PATH"

# Output for GitHub Actions
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "manifest_path=${MANIFEST_PATH}" >> "$GITHUB_OUTPUT"
    echo "dags_path=${DAGS_PATH}" >> "$GITHUB_OUTPUT"
    echo "ml_path=${ML_PATH}" >> "$GITHUB_OUTPUT"
fi
