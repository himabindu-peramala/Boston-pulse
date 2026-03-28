"""
Boston Pulse ML - Vertex AI Training Runner.

Submits training jobs to Vertex AI CustomJob.
This is an optional backend for the training pipeline.

Usage:
    from shared.vertex_runner import submit_vertex_job
    result = submit_vertex_job(
        execution_date="2026-03-23",
        ml_image="us-east1-docker.pkg.dev/bostonpulse/ml-images/ml-training:abc123",
        cfg=config,
    )

The Vertex job runs the same CLI as the Docker-on-VM path:
    python -m models.crime_navigate.cli train --execution-date ... --output-json ...

Results are written to GCS and returned via XCom.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def submit_vertex_job(
    execution_date: str,
    ml_image: str,
    cfg: dict[str, Any],
    git_sha: str = "unknown",
    stage: str = "staging",
    wait_for_completion: bool = True,
    poll_interval: int = 60,
) -> dict[str, Any]:
    """
    Submit a training job to Vertex AI.

    Args:
        execution_date: Date string (YYYY-MM-DD)
        ml_image: Full Docker image URI
        cfg: Training configuration
        git_sha: Git commit SHA
        stage: Initial model stage
        wait_for_completion: Whether to wait for job to complete
        poll_interval: Seconds between status polls

    Returns:
        Dict with job results or status
    """
    try:
        from google.cloud import aiplatform
    except ImportError:
        raise ImportError(
            "google-cloud-aiplatform not installed. "
            "Install with: pip install google-cloud-aiplatform"
        ) from None

    vertex_cfg = cfg.get("training_backend", {}).get("vertex", {})
    project = cfg.get("registry", {}).get("project", "bostonpulse")
    region = vertex_cfg.get("region", "us-east1")
    staging_bucket = vertex_cfg.get("staging_bucket", "boston-pulse-mlflow-artifacts")

    aiplatform.init(
        project=project,
        location=region,
        staging_bucket=f"gs://{staging_bucket}",
    )

    results_gcs_path = f"gs://{staging_bucket}/vertex_results/{execution_date}/results.json"

    command = [
        "python",
        "-m",
        "models.crime_navigate.cli",
        "train",
        "--execution-date",
        execution_date,
        "--stage",
        stage,
        "--output-json",
        "/tmp/results.json",
    ]

    env_vars = [
        {
            "name": "GCS_BUCKET",
            "value": cfg.get("data", {}).get("bucket", "boston-pulse-data-pipeline"),
        },
        {"name": "GCP_PROJECT_ID", "value": project},
        {"name": "GOOGLE_CLOUD_PROJECT", "value": project},
        {"name": "GIT_SHA", "value": git_sha},
        {"name": "ML_IMAGE", "value": ml_image},
        {"name": "RESULTS_GCS_PATH", "value": results_gcs_path},
    ]

    machine_type = vertex_cfg.get("machine_type", "n1-standard-4")
    accelerator_type = vertex_cfg.get("accelerator_type")
    accelerator_count = vertex_cfg.get("accelerator_count", 0)
    boot_disk_size_gb = vertex_cfg.get("boot_disk_size_gb", 100)
    timeout_seconds = vertex_cfg.get("timeout_seconds", 14400)
    service_account = vertex_cfg.get("service_account")

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": ml_image,
                "command": command[:2],
                "args": command[2:],
                "env": env_vars,
            },
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": boot_disk_size_gb,
            },
        }
    ]

    if accelerator_type and accelerator_count > 0:
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = accelerator_count

    job_name = f"crime-navigate-train-{execution_date.replace('-', '')}"

    logger.info(f"Submitting Vertex AI job: {job_name}")
    logger.info(f"  Image: {ml_image}")
    logger.info(f"  Machine: {machine_type}")
    logger.info(f"  Region: {region}")

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        labels={
            "execution_date": execution_date.replace("-", "_"),
            "git_sha": git_sha[:8] if len(git_sha) > 8 else git_sha,
            "pipeline": "crime_navigate_train",
        },
    )

    if service_account:
        job.run(
            service_account=service_account,
            timeout=timeout_seconds,
            sync=wait_for_completion,
        )
    else:
        job.run(
            timeout=timeout_seconds,
            sync=wait_for_completion,
        )

    if wait_for_completion:
        return _fetch_results(results_gcs_path, job)
    else:
        return {
            "status": "submitted",
            "job_name": job.display_name,
            "job_resource_name": job.resource_name,
            "results_gcs_path": results_gcs_path,
        }


def _fetch_results(results_gcs_path: str, job: Any) -> dict[str, Any]:
    """Fetch results from GCS after job completion."""
    from google.cloud import storage

    logger.info(f"Fetching results from {results_gcs_path}")

    client = storage.Client()
    bucket_name = results_gcs_path.split("/")[2]
    blob_path = "/".join(results_gcs_path.split("/")[3:])

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    max_retries = 5
    for _i in range(max_retries):
        if blob.exists():
            content = blob.download_as_text()
            results = json.loads(content)
            results["vertex_job_name"] = job.display_name
            results["vertex_job_resource"] = job.resource_name
            return results
        time.sleep(10)

    logger.warning(f"Results not found at {results_gcs_path}")
    return {
        "status": "completed_no_results",
        "vertex_job_name": job.display_name,
        "vertex_job_resource": job.resource_name,
        "results_gcs_path": results_gcs_path,
    }


def get_backend(cfg: dict[str, Any]) -> str:
    """Get the configured training backend."""
    return cfg.get("training_backend", {}).get("backend", "docker_on_vm")


def is_vertex_enabled(cfg: dict[str, Any]) -> bool:
    """Check if Vertex AI backend is enabled."""
    return get_backend(cfg) == "vertex"
