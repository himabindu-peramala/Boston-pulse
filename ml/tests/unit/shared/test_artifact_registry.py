"""Tests for shared/artifact_registry.py ArtifactRegistryClient."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.artifact_registry import (
    ArtifactRegistryClient,
    ensure_repository_exists,
)


@pytest.fixture
def ar_cfg() -> dict[str, Any]:
    return {
        "registry": {
            "project": "test-project",
            "location": "us-east1",
            "repository": "test-models",
            "package": "navigate/test-model",
        },
        "model": {"name": "test-model"},
    }


@pytest.fixture
def ar_client(ar_cfg: dict[str, Any]) -> ArtifactRegistryClient:
    return ArtifactRegistryClient(ar_cfg)


class TestArtifactRegistryClient:
    """Tests for ArtifactRegistryClient."""

    def test_init_sets_correct_paths(self, ar_client: ArtifactRegistryClient) -> None:
        """Client initializes with correct AR paths including namespaced package."""
        assert ar_client.project == "test-project"
        assert ar_client.location == "us-east1"
        assert ar_client.repository == "test-models"
        assert ar_client.package == "navigate/test-model"
        assert ar_client.model_name == "test-model"
        assert ar_client.ar_host == "us-east1-generic.pkg.dev"
        assert ar_client.ar_path == "test-project/test-models/navigate/test-model"

    def test_create_artifact_package_creates_tarball(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """Package creation produces valid tarball with model and metadata."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model-content")

        metadata = {"version": "20260322", "rmse": 0.5}

        package_path = ar_client._create_artifact_package(str(model_file), metadata, shap_path=None)

        assert Path(package_path).exists()
        assert package_path.endswith(".tar.gz")

        with tarfile.open(package_path, "r:gz") as tar:
            names = tar.getnames()
            assert "model.lgb" in names
            assert "metadata.json" in names

            meta_member = tar.getmember("metadata.json")
            meta_file = tar.extractfile(meta_member)
            meta_content = json.loads(meta_file.read().decode())
            assert meta_content["version"] == "20260322"
            assert meta_content["rmse"] == 0.5

    def test_create_artifact_package_includes_shap(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """Package includes SHAP plot when provided."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")
        shap_file = tmp_path / "shap.png"
        shap_file.write_bytes(b"png-content")

        package_path = ar_client._create_artifact_package(
            str(model_file), {}, shap_path=str(shap_file)
        )

        with tarfile.open(package_path, "r:gz") as tar:
            assert "shap_summary.png" in tar.getnames()

    def test_compute_sha256_returns_hex_digest(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """SHA256 computation returns valid hex string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        sha = ar_client._compute_sha256(str(test_file))

        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    @patch.object(ArtifactRegistryClient, "_run_gcloud")
    @patch.object(ArtifactRegistryClient, "_set_tag")
    def test_push_calls_gcloud_upload(
        self,
        mock_set_tag: MagicMock,
        mock_run_gcloud: MagicMock,
        ar_client: ArtifactRegistryClient,
        tmp_path: Path,
    ) -> None:
        """Push uploads package via gcloud artifacts generic upload."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")

        mock_run_gcloud.return_value = MagicMock(returncode=0)

        result = ar_client.push(
            model_path=str(model_file),
            version="20260322",
            metadata={"rmse": 0.5},
            stage="staging",
        )

        assert "ar_uri" in result
        assert result["version"] == "20260322"
        assert result["stage"] == "staging"
        assert "sha256" in result

        upload_call = mock_run_gcloud.call_args_list[0]
        assert "artifacts" in upload_call[0][0]
        assert "generic" in upload_call[0][0]
        assert "upload" in upload_call[0][0]

    @patch.object(ArtifactRegistryClient, "_set_tag")
    def test_promote_to_production_sets_tag(
        self, mock_set_tag: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Promotion sets the production tag."""
        result = ar_client.promote_to_production("20260322")

        assert result["version"] == "20260322"
        assert result["stage"] == "production"
        mock_set_tag.assert_called_with("20260322", "production")

    @patch.object(ArtifactRegistryClient, "_run_gcloud")
    def test_pull_downloads_and_extracts(
        self, mock_run_gcloud: MagicMock, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """Pull downloads package and extracts model + metadata."""
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        model_file = package_dir / "model.lgb"
        model_file.write_bytes(b"model-content")
        meta_file = package_dir / "metadata.json"
        meta_file.write_text(json.dumps({"version": "20260322"}))

        tar_path = tmp_path / "test-model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.lgb")
            tar.add(meta_file, arcname="metadata.json")

        def mock_download(args: list[str], check: bool = True):
            dest = None
            for _i, arg in enumerate(args):
                if arg.startswith("--destination="):
                    dest = arg.split("=")[1]
            if dest:
                import shutil

                shutil.copy(tar_path, Path(dest) / "test-model.tar.gz")
            return MagicMock(returncode=0)

        mock_run_gcloud.side_effect = mock_download

        local_dir = tmp_path / "output"
        local_dir.mkdir()
        model_path, metadata = ar_client.pull("production", str(local_dir))

        assert Path(model_path).exists()
        assert metadata["version"] == "20260322"

    @patch.object(ArtifactRegistryClient, "_run_gcloud")
    def test_list_versions_parses_json_output(
        self, mock_run_gcloud: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """List versions parses gcloud JSON output."""
        mock_run_gcloud.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                [
                    {"name": "projects/p/locations/l/repositories/r/packages/m/versions/20260322"},
                    {"name": "projects/p/locations/l/repositories/r/packages/m/versions/20260315"},
                ]
            ),
        )

        versions = ar_client.list_versions()

        assert len(versions) == 2
        assert versions[0]["version"] == "20260322"
        assert versions[1]["version"] == "20260315"

    @patch.object(ArtifactRegistryClient, "_run_gcloud")
    def test_get_version_by_tag_returns_version(
        self, mock_run_gcloud: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag extracts version from tag info."""
        mock_run_gcloud.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "name": "projects/p/locations/l/repositories/r/packages/m/tags/production",
                        "version": "projects/p/locations/l/repositories/r/packages/m/versions/20260322",
                    }
                ]
            ),
        )

        version = ar_client.get_version_by_tag("production")

        assert version == "20260322"

    @patch.object(ArtifactRegistryClient, "_run_gcloud")
    def test_get_version_by_tag_returns_none_if_not_found(
        self, mock_run_gcloud: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag returns None if tag doesn't exist."""
        mock_run_gcloud.return_value = MagicMock(returncode=0, stdout="[]")

        version = ar_client.get_version_by_tag("nonexistent")

        assert version is None

    @patch.object(ArtifactRegistryClient, "promote_to_production")
    def test_rollback_calls_promote(
        self, mock_promote: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Rollback delegates to promote_to_production."""
        mock_promote.return_value = {"version": "20260315", "stage": "production"}

        result = ar_client.rollback("20260315")

        mock_promote.assert_called_once_with("20260315")
        assert result["version"] == "20260315"


class TestEnsureRepositoryExists:
    """Tests for ensure_repository_exists helper."""

    @patch("subprocess.run")
    def test_creates_repo_if_not_exists(self, mock_run: MagicMock) -> None:
        """Creates repository if describe fails."""
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0),
        ]

        ensure_repository_exists("proj", "us-east1", "ml-models")

        assert mock_run.call_count == 2
        create_call = mock_run.call_args_list[1]
        assert "create" in create_call[0][0]

    @patch("subprocess.run")
    def test_skips_create_if_exists(self, mock_run: MagicMock) -> None:
        """Skips creation if repository already exists."""
        mock_run.return_value = MagicMock(returncode=0)

        ensure_repository_exists("proj", "us-east1", "ml-models")

        assert mock_run.call_count == 1
