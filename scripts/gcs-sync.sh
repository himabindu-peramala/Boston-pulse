#!/usr/bin/env bash
# scripts/gcs-sync.sh
#
# GCS-based deployment sync for Boston Pulse Airflow.
# Polls GCS for new deployment manifests and syncs DAGs/source to local.
#
# This runs as a systemd service on the Airflow VM, replacing git-based sync.
#
# Usage:
#   ./scripts/gcs-sync.sh [--once]
#
# Options:
#   --once    Run once and exit (for testing)
#
# Environment:
#   GCS_DEPLOY_BUCKET   Deployment bucket (default: boston-pulse-deployments)
#   DEPLOY_PREFIX       GCS prefix for manifests (default: airflow)
#   LOCAL_DAGS_DIR      Local DAGs directory (default: /opt/airflow/dags)
#   LOCAL_ML_DIR        Local ML directory (default: /opt/airflow/ml)
#   POLL_INTERVAL       Seconds between polls (default: 60)
#   STATE_FILE          File to track last deployed SHA (default: /opt/airflow/.last_deployed_sha)

set -euo pipefail

# Configuration with defaults
GCS_DEPLOY_BUCKET="${GCS_DEPLOY_BUCKET:-boston-pulse-deployments}"
DEPLOY_PREFIX="${DEPLOY_PREFIX:-airflow}"
LOCAL_DAGS_DIR="${LOCAL_DAGS_DIR:-/opt/airflow/dags}"
LOCAL_ML_DIR="${LOCAL_ML_DIR:-/opt/airflow/ml}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
STATE_FILE="${STATE_FILE:-/opt/airflow/.last_deployed_sha}"

MANIFEST_PATH="gs://${GCS_DEPLOY_BUCKET}/${DEPLOY_PREFIX}/manifest.json"
DAGS_PATH="gs://${GCS_DEPLOY_BUCKET}/${DEPLOY_PREFIX}/dags/"
ML_PATH="gs://${GCS_DEPLOY_BUCKET}/${DEPLOY_PREFIX}/ml/"

RUN_ONCE=false
if [[ "${1:-}" == "--once" ]]; then
    RUN_ONCE=true
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_current_sha() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo ""
    fi
}

set_current_sha() {
    local sha="$1"
    echo "$sha" > "$STATE_FILE"
    log "Updated state file: $sha"
}

fetch_manifest() {
    local tmp_manifest
    tmp_manifest=$(mktemp)
    
    if gsutil -q cp "$MANIFEST_PATH" "$tmp_manifest" 2>/dev/null; then
        cat "$tmp_manifest"
        rm -f "$tmp_manifest"
        return 0
    else
        rm -f "$tmp_manifest"
        return 1
    fi
}

sync_deployment() {
    local manifest="$1"
    local sha
    sha=$(echo "$manifest" | jq -r '.git_sha // empty')
    
    if [[ -z "$sha" ]]; then
        log "ERROR: Manifest missing git_sha"
        return 1
    fi
    
    local current_sha
    current_sha=$(get_current_sha)
    
    if [[ "$sha" == "$current_sha" ]]; then
        log "Already at SHA $sha, skipping sync"
        return 0
    fi
    
    log "New deployment detected: $current_sha -> $sha"
    
    # Sync DAGs
    log "Syncing DAGs from $DAGS_PATH..."
    mkdir -p "$LOCAL_DAGS_DIR"
    gsutil -m rsync -r -d "$DAGS_PATH" "$LOCAL_DAGS_DIR/" || {
        log "ERROR: DAG sync failed"
        return 1
    }
    
    # Sync ML code (if present)
    if gsutil -q ls "$ML_PATH" &>/dev/null; then
        log "Syncing ML code from $ML_PATH..."
        mkdir -p "$LOCAL_ML_DIR"
        gsutil -m rsync -r -d "$ML_PATH" "$LOCAL_ML_DIR/" || {
            log "ERROR: ML sync failed"
            return 1
        }
    fi
    
    # Update state
    set_current_sha "$sha"
    
    # Log deployment info
    local timestamp
    timestamp=$(echo "$manifest" | jq -r '.timestamp // "unknown"')
    local workflow_run
    workflow_run=$(echo "$manifest" | jq -r '.workflow_run_id // "unknown"')
    
    log "Deployment complete:"
    log "  SHA: $sha"
    log "  Timestamp: $timestamp"
    log "  Workflow: $workflow_run"
    
    return 0
}

main_loop() {
    log "Starting GCS sync daemon"
    log "  Bucket: $GCS_DEPLOY_BUCKET"
    log "  Prefix: $DEPLOY_PREFIX"
    log "  DAGs dir: $LOCAL_DAGS_DIR"
    log "  ML dir: $LOCAL_ML_DIR"
    log "  Poll interval: ${POLL_INTERVAL}s"
    
    while true; do
        local manifest
        if manifest=$(fetch_manifest); then
            sync_deployment "$manifest" || log "Sync failed, will retry"
        else
            log "No manifest found at $MANIFEST_PATH"
        fi
        
        if $RUN_ONCE; then
            log "Single run complete, exiting"
            break
        fi
        
        sleep "$POLL_INTERVAL"
    done
}

# Ensure required tools are available
for cmd in gsutil jq; do
    if ! command -v "$cmd" &>/dev/null; then
        log "ERROR: Required command '$cmd' not found"
        exit 1
    fi
done

main_loop
