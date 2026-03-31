#!/usr/bin/env bash
# infrastructure/gcp-setup.sh
#
# Run this script to create/verify all required GCP resources.
#
# Usage:
#   ./infrastructure/gcp-setup.sh [--dry-run]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Sufficient IAM permissions (Owner or Editor + specific roles)
#
# Resources created:
#   - GCS bucket: boston-pulse-deployments (for GCS-based deploy)
#   - Artifact Registry: ml-models (generic format for model artifacts)
#   - Artifact Registry: ml-images (Docker format for training images)
#   - Service account bindings for Workload Identity Federation
#   - Required API enablements

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-bostonpulse}"
REGION="${GCP_REGION:-us-east1}"

# Bucket names
DEPLOY_BUCKET="boston-pulse-deployments"
DATA_BUCKET="boston-pulse-data-pipeline"
MLFLOW_BUCKET="boston-pulse-mlflow-artifacts"

# Artifact Registry repositories
AR_MODELS_REPO="ml-models"
AR_IMAGES_REPO="ml-images"

# Service accounts
DATA_PIPELINE_SA="boston-pulse-data-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"

# GitHub Workload Identity Federation
WIF_POOL="github-pool"
WIF_PROVIDER="github-provider"
GITHUB_REPO="${GITHUB_REPOSITORY:-}"  # Set via env or leave empty to skip WIF

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No changes will be made ==="
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    if $DRY_RUN; then
        echo "[DRY RUN] Would execute: $*"
    else
        "$@"
    fi
}

# =============================================================================
# API Enablement
# =============================================================================
log "Enabling required APIs..."

APIS=(
    "artifactregistry.googleapis.com"
    "storage.googleapis.com"
    "firestore.googleapis.com"
    "iam.googleapis.com"
    "iamcredentials.googleapis.com"
    "cloudresourcemanager.googleapis.com"
)

for api in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        log "  ✓ $api already enabled"
    else
        log "  Enabling $api..."
        run_cmd gcloud services enable "$api" --project="$PROJECT_ID"
    fi
done

# =============================================================================
# GCS Buckets
# =============================================================================
log "Creating/verifying GCS buckets..."

create_bucket_if_not_exists() {
    local bucket_name="$1"
    local location="${2:-$REGION}"
    
    if gsutil ls -b "gs://${bucket_name}" &>/dev/null; then
        log "  ✓ gs://${bucket_name} already exists"
    else
        log "  Creating gs://${bucket_name}..."
        run_cmd gsutil mb -p "$PROJECT_ID" -l "$location" -b on "gs://${bucket_name}"
    fi
}

create_bucket_if_not_exists "$DEPLOY_BUCKET"
create_bucket_if_not_exists "$DATA_BUCKET"
create_bucket_if_not_exists "$MLFLOW_BUCKET"

# =============================================================================
# Artifact Registry Repositories
# =============================================================================
log "Creating/verifying Artifact Registry repositories..."

create_ar_repo_if_not_exists() {
    local repo_name="$1"
    local format="$2"
    local description="$3"
    
    if gcloud artifacts repositories describe "$repo_name" \
        --location="$REGION" \
        --project="$PROJECT_ID" &>/dev/null; then
        log "  ✓ AR repo $repo_name already exists"
    else
        log "  Creating AR repo $repo_name ($format format)..."
        run_cmd gcloud artifacts repositories create "$repo_name" \
            --repository-format="$format" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --description="$description"
    fi
}

# Generic format for model artifacts (LightGBM .lgb files, metadata, SHAP plots)
create_ar_repo_if_not_exists "$AR_MODELS_REPO" "generic" "ML model artifacts with stage-based promotion"

# Docker format for training container images
create_ar_repo_if_not_exists "$AR_IMAGES_REPO" "docker" "ML training Docker images"

# =============================================================================
# IAM Bindings for Service Account
# =============================================================================
log "Configuring IAM bindings for service account..."

grant_role_if_not_exists() {
    local sa="$1"
    local role="$2"
    local resource="${3:-}"
    
    if [[ -n "$resource" ]]; then
        # Resource-level binding (e.g., bucket)
        log "  Granting $role on $resource to $sa..."
        run_cmd gsutil iam ch "serviceAccount:${sa}:${role}" "$resource" 2>/dev/null || true
    else
        # Project-level binding
        local existing
        existing=$(gcloud projects get-iam-policy "$PROJECT_ID" \
            --flatten="bindings[].members" \
            --filter="bindings.role:$role AND bindings.members:serviceAccount:$sa" \
            --format="value(bindings.members)" 2>/dev/null || true)
        
        if [[ -n "$existing" ]]; then
            log "  ✓ $sa already has $role"
        else
            log "  Granting $role to $sa..."
            run_cmd gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$sa" \
                --role="$role" \
                --quiet
        fi
    fi
}

# Artifact Registry roles
grant_role_if_not_exists "$DATA_PIPELINE_SA" "roles/artifactregistry.writer"
grant_role_if_not_exists "$DATA_PIPELINE_SA" "roles/artifactregistry.reader"

# Storage roles (for GCS buckets)
grant_role_if_not_exists "$DATA_PIPELINE_SA" "roles/storage.objectAdmin"

# Firestore roles
grant_role_if_not_exists "$DATA_PIPELINE_SA" "roles/datastore.user"

# =============================================================================
# Workload Identity Federation (for GitHub Actions)
# =============================================================================
if [[ -n "$GITHUB_REPO" ]]; then
    log "Configuring Workload Identity Federation for GitHub Actions..."
    
    # Create identity pool if not exists
    if gcloud iam workload-identity-pools describe "$WIF_POOL" \
        --location="global" \
        --project="$PROJECT_ID" &>/dev/null; then
        log "  ✓ WIF pool $WIF_POOL already exists"
    else
        log "  Creating WIF pool $WIF_POOL..."
        run_cmd gcloud iam workload-identity-pools create "$WIF_POOL" \
            --location="global" \
            --project="$PROJECT_ID" \
            --display-name="GitHub Actions Pool"
    fi
    
    # Create OIDC provider if not exists
    if gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER" \
        --workload-identity-pool="$WIF_POOL" \
        --location="global" \
        --project="$PROJECT_ID" &>/dev/null; then
        log "  ✓ WIF provider $WIF_PROVIDER already exists"
    else
        log "  Creating WIF provider $WIF_PROVIDER..."
        run_cmd gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
            --workload-identity-pool="$WIF_POOL" \
            --location="global" \
            --project="$PROJECT_ID" \
            --issuer-uri="https://token.actions.githubusercontent.com" \
            --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
            --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
    fi
    
    # Bind service account to WIF
    WIF_MEMBER="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPO}"
    
    log "  Binding service account to WIF..."
    run_cmd gcloud iam service-accounts add-iam-policy-binding "$DATA_PIPELINE_SA" \
        --project="$PROJECT_ID" \
        --role="roles/iam.workloadIdentityUser" \
        --member="$WIF_MEMBER" \
        --quiet 2>/dev/null || log "  (binding may already exist)"
else
    log "Skipping WIF setup (GITHUB_REPOSITORY not set)"
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "=== Infrastructure Setup Complete ==="
log ""
log "GCS Buckets:"
log "  - gs://${DEPLOY_BUCKET} (deployment artifacts)"
log "  - gs://${DATA_BUCKET} (data pipeline)"
log "  - gs://${MLFLOW_BUCKET} (MLflow artifacts)"
log ""
log "Artifact Registry:"
log "  - ${REGION}-generic.pkg.dev/${PROJECT_ID}/${AR_MODELS_REPO} (model artifacts)"
log "  - ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_IMAGES_REPO} (training images)"
log ""
log "Service Account: ${DATA_PIPELINE_SA}"
log ""

if [[ -n "$GITHUB_REPO" ]]; then
    PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
    log "GitHub Actions Secrets (add to repository settings):"
    log "  WIF_PROVIDER: projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"
    log "  WIF_SERVICE_ACCOUNT: ${DATA_PIPELINE_SA}"
    log "  GCP_PROJECT_ID: ${PROJECT_ID}"
    log "  GCS_BUCKET: ${DATA_BUCKET}"
fi
