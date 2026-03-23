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


@pytest.fixture
def mock_credentials():
    """Mock ADC credentials."""
    with patch("shared.artifact_registry.google.auth.default") as mock_auth:
        mock_creds = MagicMock()
        mock_creds.token = "fake-token-12345"
        mock_creds.valid = True
        mock_auth.return_value = (mock_creds, "test-project")
        yield mock_creds


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

    def test_init_sets_sdk_resource_names(self, ar_client: ArtifactRegistryClient) -> None:
        """Client initializes with correct SDK resource names."""
        assert ar_client._parent == (
            "projects/test-project/locations/us-east1/repositories/test-models"
        )
        assert ar_client._package_parent == (
            "projects/test-project/locations/us-east1/repositories/test-models"
            "/packages/navigate/test-model"
        )

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

    @patch.object(ArtifactRegistryClient, "_upload_generic_artifact")
    @patch.object(ArtifactRegistryClient, "_set_tag")
    def test_push_uploads_via_sdk(
        self,
        mock_set_tag: MagicMock,
        mock_upload: MagicMock,
        ar_client: ArtifactRegistryClient,
        tmp_path: Path,
    ) -> None:
        """Push uploads package via Python SDK."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")

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

        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert call_args[0][1] == "20260322"

    @patch.object(ArtifactRegistryClient, "_set_tag")
    def test_promote_to_production_sets_tag(
        self, mock_set_tag: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Promotion sets the production tag."""
        result = ar_client.promote_to_production("20260322")

        assert result["version"] == "20260322"
        assert result["stage"] == "production"
        mock_set_tag.assert_called_with("20260322", "production")

    @patch.object(ArtifactRegistryClient, "_download_generic_artifact")
    @patch.object(ArtifactRegistryClient, "get_version_by_tag")
    def test_pull_downloads_and_extracts(
        self,
        mock_get_tag: MagicMock,
        mock_download: MagicMock,
        ar_client: ArtifactRegistryClient,
        tmp_path: Path,
    ) -> None:
        """Pull downloads package and extracts model + metadata."""
        mock_get_tag.return_value = "20260322"

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

        def mock_download_impl(version: str, local_dir: str):
            import shutil

            shutil.copy(tar_path, Path(local_dir) / "test-model.tar.gz")

        mock_download.side_effect = mock_download_impl

        local_dir = tmp_path / "output"
        local_dir.mkdir()
        model_path, metadata = ar_client.pull("production", str(local_dir))

        assert Path(model_path).exists()
        assert metadata["version"] == "20260322"
        mock_get_tag.assert_called_once_with("production")

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_list_versions_uses_sdk(
        self, mock_ar_class: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """List versions uses Python SDK."""
        mock_sdk_client = MagicMock()
        mock_ar_class.return_value = mock_sdk_client

        mock_version1 = MagicMock()
        mock_version1.name = "projects/p/locations/l/repositories/r/packages/m/versions/20260322"
        mock_version1.create_time = None
        mock_version1.update_time = None

        mock_version2 = MagicMock()
        mock_version2.name = "projects/p/locations/l/repositories/r/packages/m/versions/20260315"
        mock_version2.create_time = None
        mock_version2.update_time = None

        mock_sdk_client.list_versions.return_value = [mock_version1, mock_version2]

        ar_client._ar_client = None
        versions = ar_client.list_versions()

        assert len(versions) == 2
        assert versions[0]["version"] == "20260322"
        assert versions[1]["version"] == "20260315"

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_get_version_by_tag_returns_version(
        self, mock_ar_class: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag extracts version from tag info using SDK."""
        mock_sdk_client = MagicMock()
        mock_ar_class.return_value = mock_sdk_client

        mock_tag = MagicMock()
        mock_tag.version = "projects/p/locations/l/repositories/r/packages/m/versions/20260322"
        mock_sdk_client.get_tag.return_value = mock_tag

        ar_client._ar_client = None
        version = ar_client.get_version_by_tag("production")

        assert version == "20260322"
        mock_sdk_client.get_tag.assert_called_once()

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_get_version_by_tag_returns_none_if_not_found(
        self, mock_ar_class: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag returns None if tag doesn't exist."""
        mock_sdk_client = MagicMock()
        mock_ar_class.return_value = mock_sdk_client
        mock_sdk_client.get_tag.side_effect = Exception("Tag not found")

        ar_client._ar_client = None
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

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_list_tags_returns_dict(
        self, mock_ar_class: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """List tags returns dict of tag_name → version_id."""
        mock_sdk_client = MagicMock()
        mock_ar_class.return_value = mock_sdk_client

        mock_tag1 = MagicMock()
        mock_tag1.name = "projects/p/locations/l/repositories/r/packages/m/tags/production"
        mock_tag1.version = "projects/p/locations/l/repositories/r/packages/m/versions/20260322"

        mock_tag2 = MagicMock()
        mock_tag2.name = "projects/p/locations/l/repositories/r/packages/m/tags/staging"
        mock_tag2.version = "projects/p/locations/l/repositories/r/packages/m/versions/20260323"

        mock_sdk_client.list_tags.return_value = [mock_tag1, mock_tag2]

        ar_client._ar_client = None
        tags = ar_client.list_tags()

        assert tags == {"production": "20260322", "staging": "20260323"}


class TestEnsureRepositoryExists:
    """Tests for ensure_repository_exists helper."""

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_creates_repo_if_not_exists(self, mock_ar_class: MagicMock) -> None:
        """Creates repository if get_repository fails."""
        mock_client = MagicMock()
        mock_ar_class.return_value = mock_client
        mock_client.get_repository.side_effect = Exception("Not found")

        mock_operation = MagicMock()
        mock_client.create_repository.return_value = mock_operation

        ensure_repository_exists("proj", "us-east1", "ml-models")

        mock_client.get_repository.assert_called_once()
        mock_client.create_repository.assert_called_once()
        mock_operation.result.assert_called_once()

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_skips_create_if_exists(self, mock_ar_class: MagicMock) -> None:
        """Skips creation if repository already exists."""
        mock_client = MagicMock()
        mock_ar_class.return_value = mock_client
        mock_client.get_repository.return_value = MagicMock()

        ensure_repository_exists("proj", "us-east1", "ml-models")

        mock_client.get_repository.assert_called_once()
        mock_client.create_repository.assert_not_called()
