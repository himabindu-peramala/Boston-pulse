"""
Boston Pulse ML - Crime Navigate Training DAG (vNext).

Staging DAG for testing new training backends (Vertex AI).
Runs in parallel with the production DAG until validated.

This DAG supports two backends via config:
  - docker_on_vm: Current production path (Docker container on Airflow VM)
  - vertex: Vertex AI CustomJob (scalable, managed)

The backend is selected via training_backend.backend in config.

Trigger: Manual or scheduled (separate from production DAG)
Schedule: None (triggered manually for testing)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

DAG_ID = "crime_navigate_train_vnext"
DATASET = "crime_navigate"

DEFAULT_ML_IMAGE = "us-east1-docker.pkg.dev/bostonpulse/ml-images/ml-training:latest"

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


def load_config() -> dict[str, Any]:
    """Load training configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "crime_navigate_train.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_ml_image(**context: Any) -> str:
    """Get ML training image from DAG conf or default."""
    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        image = dag_run.conf.get("ml_image")
        if image:
            return image
    return DEFAULT_ML_IMAGE


def choose_backend(**context: Any) -> str:
    """Branch to appropriate backend based on config."""
    cfg = load_config()
    backend = cfg.get("training_backend", {}).get("backend", "docker_on_vm")

    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        backend_override = dag_run.conf.get("backend")
        if backend_override:
            backend = backend_override

    print(f"Selected backend: {backend}")

    if backend == "vertex":
        return "run_vertex_training"
    else:
        return "run_docker_training"


def run_docker_training(**context: Any) -> dict[str, Any]:
    """Run training in Docker container on VM (current production path)."""
    import json

    import docker

    execution_date = context["ds"]
    dag_run = context.get("dag_run")

    ml_image = get_ml_image(**context)
    git_sha = dag_run.conf.get("git_sha", "unknown") if dag_run and dag_run.conf else "unknown"

    print(f"Running Docker training: {ml_image}")
    print(f"Git SHA: {git_sha}")

    env_vars = {
        "GCS_BUCKET": os.getenv("GCS_BUCKET", "boston-pulse-data-pipeline"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID", "bostonpulse"),
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", "bostonpulse"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlflow.db"),
        "GIT_SHA": git_sha,
        "ML_IMAGE": ml_image,
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL", ""),
    }

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

    print(f"Pulling image: {ml_image}")
    try:
        client.images.pull(ml_image)
    except docker.errors.APIError as e:
        print(f"Warning: Could not pull image: {e}")

    print(f"Starting container: {' '.join(command)}")

    try:
        container = client.containers.run(
            image=ml_image,
            command=command,
            environment=env_vars,
            network_mode="host",
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
        )

        output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
        print("Container output:")
        print(output)

        try:
            lines = output.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
        except (json.JSONDecodeError, IndexError):
            pass

        return {"status": "success", "backend": "docker_on_vm", "output": output[:1000]}

    except docker.errors.ContainerError as e:
        raise RuntimeError(f"Docker training failed: {e.stderr}") from e
    except docker.errors.ImageNotFound:
        raise RuntimeError(f"Image not found: {ml_image}") from None


def run_vertex_training(**context: Any) -> dict[str, Any]:
    """Run training via Vertex AI CustomJob."""
    from shared.vertex_runner import submit_vertex_job

    execution_date = context["ds"]
    dag_run = context.get("dag_run")

    ml_image = get_ml_image(**context)
    git_sha = dag_run.conf.get("git_sha", "unknown") if dag_run and dag_run.conf else "unknown"

    cfg = load_config()

    print("Submitting Vertex AI job")
    print(f"Image: {ml_image}")
    print(f"Git SHA: {git_sha}")

    result = submit_vertex_job(
        execution_date=execution_date,
        ml_image=ml_image,
        cfg=cfg,
        git_sha=git_sha,
        stage="staging",
        wait_for_completion=True,
    )

    result["backend"] = "vertex"
    return result


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
    description="Navigate crime risk scoring — vNext (Vertex/Docker backend selection)",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["navigate", "ml", "crime", "training", "vnext", "staging"],
    params={
        "ml_image": DEFAULT_ML_IMAGE,
        "git_sha": "manual",
        "backend": "",
    },
) as dag:

    t_choose_backend = BranchPythonOperator(
        task_id="choose_backend",
        python_callable=choose_backend,
    )

    t_docker = PythonOperator(
        task_id="run_docker_training",
        python_callable=run_docker_training,
        on_failure_callback=on_task_failure,
        execution_timeout=timedelta(hours=4),
    )

    t_vertex = PythonOperator(
        task_id="run_vertex_training",
        python_callable=run_vertex_training,
        on_failure_callback=on_task_failure,
        execution_timeout=timedelta(hours=4),
    )

    t_choose_backend >> [t_docker, t_vertex]
