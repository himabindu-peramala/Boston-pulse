"""
Boston Pulse ML - Crime Navigate Training DAG (Docker-based).

This DAG runs the ML training pipeline inside a versioned Docker container.
The container image is built in CI and tagged with the git SHA, ensuring
the exact code version that passed tests is used for training.

Trigger: GitHub Actions workflow on push to main
  - CI builds docker/ml-training.Dockerfile
  - Pushes to us-east1-docker.pkg.dev/bostonpulse/ml-images/ml-training:<sha>
  - Triggers this DAG with conf: {"ml_image": "<full-image-uri>", "git_sha": "<sha>"}
The DockerOperator pulls the specified image and runs the training CLI.
This guarantees reproducibility: the training code matches the tested commit.

Schedule: None (triggered by CI only)
  - Weekly schedule is handled by GitHub Actions cron
  - Manual runs via workflow_dispatch also trigger this DAG
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "crime_navigate_train"
DATASET = "crime_navigate"

# Default image for manual DAG runs without CI trigger
DEFAULT_ML_IMAGE = "us-east1-docker.pkg.dev/bostonpulse/ml-images/ml-training:latest"

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


def get_ml_image(**context: Any) -> str:
    """
    Get the ML training image from DAG run conf or use default.

    When triggered by CI, conf contains:
      {"ml_image": "us-east1-docker.pkg.dev/bostonpulse/ml-images/ml-training:<sha>"}

    For manual runs or schedule without CI, falls back to :latest tag.
    """
    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        image = dag_run.conf.get("ml_image")
        if image:
            return image

    return DEFAULT_ML_IMAGE


def run_training_container(**context: Any) -> dict[str, Any]:
    """
    Run the ML training pipeline inside a Docker container.

    Uses the Docker SDK to pull and run the versioned ML image.
    The container has access to GCP credentials via the metadata server
    (same as the Airflow containers on the VM).
    """
    import json

    import docker

    execution_date = context["ds"]
    dag_run = context.get("dag_run")

    # Get image from conf or default
    ml_image = get_ml_image(**context)
    git_sha = dag_run.conf.get("git_sha", "unknown") if dag_run and dag_run.conf else "unknown"

    print(f"Running training with image: {ml_image}")
    print(f"Git SHA: {git_sha}")
    print(f"Execution date: {execution_date}")

    # Environment variables for the training container
    env_vars = {
        "GCS_BUCKET": os.getenv("GCS_BUCKET", "boston-pulse-data-pipeline"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID", "bostonpulse"),
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", "bostonpulse"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlflow.db"),
        "GIT_SHA": git_sha,
        "ML_IMAGE": ml_image,
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL", ""),
    }

    # Command to run inside the container
    command = [
        "python",
        "-m",
        "models.crime_navigate.cli",
        "train",
        "--execution-date",
        execution_date,
        "--stage",
        "staging",
        "--output-json",
        "/tmp/results.json",
    ]

    client = docker.from_env()

    # Pull the image first
    print(f"Pulling image: {ml_image}")
    try:
        client.images.pull(ml_image)
    except docker.errors.APIError as e:
        print(f"Warning: Could not pull image (may already exist locally): {e}")

    # Run the container
    print(f"Starting container with command: {' '.join(command)}")

    try:
        container = client.containers.run(
            image=ml_image,
            command=command,
            environment=env_vars,
            # Network mode host allows access to GCP metadata server at 169.254.169.254
            network_mode="host",
            # Remove container after completion
            remove=True,
            # Stream logs
            detach=False,
            stdout=True,
            stderr=True,
        )

        # container is bytes when detach=False
        output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
        print("Container output:")
        print(output)

        # Try to parse the JSON results from the output
        # The CLI prints JSON at the end
        try:
            # Find the last JSON object in the output
            lines = output.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    results = json.loads(line)
                    return results
        except (json.JSONDecodeError, IndexError):
            pass

        return {"status": "success", "output": output[:1000]}

    except docker.errors.ContainerError as e:
        print(f"Container failed with exit code {e.exit_status}")
        print(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Training container failed: {e.stderr}") from e
    except docker.errors.ImageNotFound:
        raise RuntimeError(f"ML training image not found: {ml_image}") from None
    except docker.errors.APIError as e:
        raise RuntimeError(f"Docker API error: {e}") from e


def on_task_failure(context: Any) -> None:
    """Callback for task failures."""
    from shared.alerting import alert_gate_failure

    task_id = context["task_instance"].task_id
    execution_date = context["ds"]
    exception = context.get("exception", "Unknown error")

    alert_gate_failure(DATASET, execution_date, f"Task: {task_id}", str(exception), DAG_ID)


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Navigate crime risk scoring — containerized training",
    # No schedule - triggered by CI workflow
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["navigate", "ml", "crime", "training", "docker"],
    # Accept conf parameters from CI trigger
    params={
        "ml_image": DEFAULT_ML_IMAGE,
        "git_sha": "manual",
    },
) as dag:

    # Single task that runs the entire pipeline in a container
    # This replaces the chain of PythonOperator tasks
    t_train = PythonOperator(
        task_id="run_training",
        python_callable=run_training_container,
        on_failure_callback=on_task_failure,
        # Increase timeout for full pipeline
        execution_timeout=timedelta(hours=4),
    )
