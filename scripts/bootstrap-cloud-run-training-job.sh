#!/usr/bin/env bash
# One-time bootstrap: create the Cloud Run Job used by .github/workflows/ml.yml
#
# Prerequisites: gcloud authenticated; APIs enabled (run.googleapis.com, artifactregistry.googleapis.com)
#
# Usage:
#   export GCP_PROJECT_ID=bostonpulse
#   export GCP_REGION=us-east1
#   ./scripts/bootstrap-cloud-run-training-job.sh
#
# CI updates --image and full --args (including --execution-date) before each execute.

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-bostonpulse}"
REGION="${GCP_REGION:-us-east1}"
JOB_NAME="${CLOUD_RUN_TRAINING_JOB:-ml-training-job}"
IMAGE="${TRAINING_IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/ml-training:latest}"
GCS_BUCKET="${GCS_BUCKET:-boston-pulse-data-pipeline}"
# SA used at runtime for GCS / Firestore / AR (must match your data access needs)
TRAINING_SA="${TRAINING_JOB_SA:-boston-pulse-data-pipeline@${PROJECT_ID}.iam.gserviceaccount.com}"

echo "Ensuring Cloud Run Job: $JOB_NAME in $PROJECT_ID ($REGION)"
echo "  Image: $IMAGE"

if gcloud run jobs describe "${JOB_NAME}" --project="${PROJECT_ID}" --region="${REGION}" &>/dev/null; then
  echo "Job exists; updating template..."
  gcloud run jobs update "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --service-account="${TRAINING_SA}" \
    --command=python \
    --args=-m,models.crime_navigate.cli,train,--execution-date,1970-01-01,--stage,staging,--output-json,/tmp/results.json \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GCS_BUCKET=${GCS_BUCKET}"
else
  echo "Creating job..."
  gcloud run jobs create "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --service-account="${TRAINING_SA}" \
    --tasks=1 \
    --max-retries=1 \
    --task-timeout=3600 \
    --memory=4Gi \
    --cpu=2 \
    --command=python \
    --args=-m,models.crime_navigate.cli,train,--execution-date,1970-01-01,--stage,staging,--output-json,/tmp/results.json \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GCS_BUCKET=${GCS_BUCKET}"
fi

echo "Done. CI runs: gcloud run jobs update ... then gcloud run jobs execute ... --wait"
