#!/usr/bin/env bash
# scripts/deploy_app.sh
#
# Deploys the training-job application layer on top of infrastructure provisioned by
# `infrastructure/terraform` (which you must have applied first).
#
# Steps:
#   1. Read outputs from Terraform (project, region, SA, image URI)
#   2. Build the ML training image via Cloud Build and push to AR
#   3. Create/update the Cloud Run Job that runs training
#   4. Print training job summary
#
# Usage:
#   ./scripts/deploy_app.sh                # build image + update training job
#   ./scripts/deploy_app.sh --skip-image   # re-use existing image
#   ./scripts/deploy_app.sh --dry-run      # preview, no changes

set -euo pipefail

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
SKIP_IMAGE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --skip-image) SKIP_IMAGE=true ;;
        --dry-run)    DRY_RUN=true ;;
        -h|--help)    sed -n '2,18p' "$0" | sed 's/^# //; s/^#//'; exit 0 ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TF_DIR="$REPO_ROOT/infrastructure/terraform"
cd "$REPO_ROOT"

BOLD=$(tput bold 2>/dev/null || true)
GREEN=$(tput setaf 2 2>/dev/null || true)
YELLOW=$(tput setaf 3 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)

step() { echo; echo "${BOLD}━━━ $* ━━━${RESET}"; }
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }
fail() { echo "  ${RED}✗${RESET} $*" >&2; exit 1; }
run()  { if $DRY_RUN; then echo "  [DRY RUN] $*"; else "$@"; fi; }

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
step "Preflight"

command -v gcloud >/dev/null    || fail "gcloud CLI not found"
command -v terraform >/dev/null || fail "terraform CLI not found (brew install terraform)"
[[ -d "$TF_DIR" ]]              || fail "Terraform directory not found at $TF_DIR"
[[ -f "$TF_DIR/terraform.tfstate" ]] \
    || fail "No Terraform state. Run 'terraform apply' in $TF_DIR first."

PROJECT_ID=$(terraform -chdir="$TF_DIR" output -raw project_id 2>/dev/null) \
    || fail "Could not read project_id from Terraform outputs"
REGION=$(terraform -chdir="$TF_DIR" output -raw region)
SA_EMAIL=$(terraform -chdir="$TF_DIR" output -raw service_account_email)
TRAINING_IMAGE=$(terraform -chdir="$TF_DIR" output -raw training_image)
DATA_BUCKET=$(terraform -chdir="$TF_DIR" output -raw data_bucket)
ARTIFACT_BUCKET=$(terraform -chdir="$TF_DIR" output -raw ml_artifacts_bucket)

ok "Project:         $PROJECT_ID"
ok "Region:          $REGION"
ok "Service acct:    $SA_EMAIL"
ok "Training image:  $TRAINING_IMAGE"
ok "Data bucket:     $DATA_BUCKET"
ok "Artifact bucket: $ARTIFACT_BUCKET"
$DRY_RUN && warn "DRY RUN — no changes will be applied"

# -----------------------------------------------------------------------------
# Step 1: Build + push ML training image
# -----------------------------------------------------------------------------
step "1/3  Build ML training image"

if $SKIP_IMAGE; then
    warn "Skipping image build (--skip-image)"
else
    run gcloud builds submit . \
        --config=ml/cloudbuild.yaml \
        --substitutions="_IMAGE=${TRAINING_IMAGE}" \
        --project="$PROJECT_ID"
    ok "Pushed $TRAINING_IMAGE"
fi

# -----------------------------------------------------------------------------
# Step 2: Cloud Run Job (training)
# -----------------------------------------------------------------------------
step "2/3  Cloud Run Job (training)"

JOB_NAME="ml-training-job"
# Env vars threaded to the container:
#   GCS_BUCKET       — data-pipeline bucket (read by ml/shared/config_loader)
#   ARTIFACT_BUCKET  — model/MLflow bucket (read by ml/shared/registry, baseline_snapshotter, vertex_runner)
JOB_ENV="GCP_PROJECT_ID=$PROJECT_ID,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$DATA_BUCKET,ARTIFACT_BUCKET=$ARTIFACT_BUCKET"
JOB_ARGS=(
    "--image=$TRAINING_IMAGE"
    "--region=$REGION"
    "--project=$PROJECT_ID"
    "--service-account=$SA_EMAIL"
    "--task-timeout=3600s"
    "--max-retries=1"
    "--memory=4Gi"
    "--cpu=2"
    "--set-env-vars=$JOB_ENV"
)

if gcloud run jobs describe "$JOB_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    run gcloud run jobs update "$JOB_NAME" "${JOB_ARGS[@]}" --quiet
    ok "Updated Cloud Run Job: $JOB_NAME"
else
    run gcloud run jobs create "$JOB_NAME" "${JOB_ARGS[@]}" --quiet
    ok "Created Cloud Run Job: $JOB_NAME"
fi

# # -----------------------------------------------------------------------------
# # Step 3: Backend service (Cloud Run)
# # -----------------------------------------------------------------------------
# step "3/3  Backend service (Cloud Run)"

# BACKEND_SERVICE="boston-pulse-backend"

# BACKEND_ENV="GCP_PROJECT_ID=$PROJECT_ID,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$DATA_BUCKET,ARTIFACT_BUCKET=$ARTIFACT_BUCKET"

# run gcloud run deploy "$BACKEND_SERVICE" \
#     --source backend/ \
#     --region "$REGION" \
#     --project "$PROJECT_ID" \
#     --service-account "$SA_EMAIL" \
#     --allow-unauthenticated \
#     --memory 1Gi \
#     --cpu 1 \
#     --timeout 60 \
#     --set-env-vars "$BACKEND_ENV" \
#     --quiet

# ok "Deployed $BACKEND_SERVICE"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
step "Done"

if ! $DRY_RUN; then
    ok "Training Job:   $JOB_NAME"
    ok "Dashboard:      https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"

    echo
    echo "Next:"
    echo "  • Run training: gcloud run jobs execute $JOB_NAME --region=$REGION --wait"
fi
