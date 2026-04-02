"""Unit tests for shared/vertex_runner.py."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared import vertex_runner


def test_get_backend_default(sample_cfg: dict[str, Any]) -> None:
    assert vertex_runner.get_backend(sample_cfg) == "docker_on_vm"


def test_get_backend_vertex(sample_cfg: dict[str, Any]) -> None:
    cfg = {**sample_cfg, "training_backend": {"backend": "vertex"}}
    assert vertex_runner.get_backend(cfg) == "vertex"


def test_is_vertex_enabled(sample_cfg: dict[str, Any]) -> None:
    assert vertex_runner.is_vertex_enabled(sample_cfg) is False
    cfg = {**sample_cfg, "training_backend": {"backend": "vertex"}}
    assert vertex_runner.is_vertex_enabled(cfg) is True


def test_fetch_results_downloads_json(mocker: Any) -> None:
    blob = MagicMock()
    blob.exists.return_value = True
    blob.download_as_text.return_value = json.dumps({"status": "success", "rmse": 0.1})
    bucket = MagicMock()
    bucket.blob.return_value = blob
    client = MagicMock()
    client.bucket.return_value = bucket
    mocker.patch("google.cloud.storage.Client", return_value=client)

    job = MagicMock()
    job.display_name = "job-1"
    job.resource_name = "projects/p/locations/l/jobs/1"

    out = vertex_runner._fetch_results("gs://my-bucket/path/to/results.json", job)
    assert out["status"] == "success"
    assert out["vertex_job_name"] == "job-1"


def test_fetch_results_missing_after_retries(mocker: Any) -> None:
    blob = MagicMock()
    blob.exists.return_value = False
    bucket = MagicMock()
    bucket.blob.return_value = blob
    mocker.patch(
        "google.cloud.storage.Client",
        return_value=MagicMock(bucket=MagicMock(return_value=bucket)),
    )
    mocker.patch("shared.vertex_runner.time.sleep")
    job = MagicMock(display_name="j", resource_name="r")
    out = vertex_runner._fetch_results("gs://b/prefix/r.json", job)
    assert out["status"] == "completed_no_results"


@pytest.fixture
def fake_aiplatform() -> MagicMock:
    """Inject a fake google.cloud.aiplatform for submit_vertex_job."""
    mod = MagicMock()
    job = MagicMock()
    job.display_name = "crime-navigate-train-20240115"
    job.resource_name = "projects/x/locations/y/customJobs/1"
    mod.CustomJob.return_value = job
    old = sys.modules.get("google.cloud.aiplatform")
    sys.modules["google.cloud.aiplatform"] = mod
    yield mod
    if old is not None:
        sys.modules["google.cloud.aiplatform"] = old
    else:
        del sys.modules["google.cloud.aiplatform"]


def test_submit_vertex_job_submitted_async(
    mocker: Any,
    sample_cfg: dict[str, Any],
    fake_aiplatform: MagicMock,
) -> None:
    out = vertex_runner.submit_vertex_job(
        execution_date="2024-01-15",
        ml_image="us-east1-docker.pkg.dev/p/img:tag",
        cfg=sample_cfg,
        wait_for_completion=False,
    )

    assert out["status"] == "submitted"
    assert "vertex_results/2024-01-15/results.json" in out["results_gcs_path"]
    job = fake_aiplatform.CustomJob.return_value
    job.run.assert_called_once()
    assert job.run.call_args[1]["sync"] is False


def test_submit_vertex_job_waits_and_fetches(
    mocker: Any,
    sample_cfg: dict[str, Any],
    fake_aiplatform: MagicMock,
) -> None:
    mocker.patch.object(
        vertex_runner,
        "_fetch_results",
        return_value={"status": "success"},
    )

    out = vertex_runner.submit_vertex_job(
        execution_date="2024-01-15",
        ml_image="us-east1-docker.pkg.dev/p/img:tag",
        cfg=sample_cfg,
        wait_for_completion=True,
    )

    assert out["status"] == "success"
    job = fake_aiplatform.CustomJob.return_value
    job.run.assert_called_once()


def test_submit_vertex_job_with_service_account(
    mocker: Any,
    sample_cfg: dict[str, Any],
    fake_aiplatform: MagicMock,
) -> None:
    cfg = {
        **sample_cfg,
        "training_backend": {
            "vertex": {
                "service_account": "svc@project.iam.gserviceaccount.com",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
        },
    }
    mocker.patch.object(vertex_runner, "_fetch_results", return_value={})

    vertex_runner.submit_vertex_job(
        execution_date="2024-01-15",
        ml_image="us-east1-docker.pkg.dev/p/img:tag",
        cfg=cfg,
        wait_for_completion=True,
    )

    job = fake_aiplatform.CustomJob.return_value
    call_kw = job.run.call_args[1]
    assert call_kw["service_account"] == "svc@project.iam.gserviceaccount.com"
    cj = fake_aiplatform.CustomJob.call_args[1]
    wps = cj["worker_pool_specs"][0]["machine_spec"]
    assert wps["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert wps["accelerator_count"] == 1


def test_submit_vertex_import_error() -> None:
    real_import = __import__

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google.cloud.aiplatform":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="google-cloud-aiplatform"):
        # Patch only for the dynamic import inside submit_vertex_job
        import builtins

        old = builtins.__import__
        builtins.__import__ = guard  # type: ignore[assignment]
        try:
            vertex_runner.submit_vertex_job(
                execution_date="2024-01-15",
                ml_image="x",
                cfg={"data": {}, "registry": {}},
            )
        finally:
            builtins.__import__ = old


def test_submit_vertex_job_git_sha_label_short(
    mocker: Any,
    sample_cfg: dict[str, Any],
    fake_aiplatform: MagicMock,
) -> None:
    mocker.patch.object(vertex_runner, "_fetch_results", return_value={})

    vertex_runner.submit_vertex_job(
        execution_date="2024-01-15",
        ml_image="img",
        cfg=sample_cfg,
        git_sha="abcd1234",
        wait_for_completion=True,
    )

    labels = fake_aiplatform.CustomJob.call_args[1]["labels"]
    assert labels["git_sha"] == "abcd1234"
