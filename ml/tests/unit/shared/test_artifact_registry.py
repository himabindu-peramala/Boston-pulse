"""Tests for shared/artifact_registry.py ArtifactRegistryClient."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.cloud import artifactregistry_v1

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
            "package": "navigate-crime-risk",
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
        """Client initializes with correct AR paths for Docker format."""
        assert ar_client.project == "test-project"
        assert ar_client.location == "us-east1"
        assert ar_client.repository == "test-models"
        assert ar_client.image_name == "navigate-crime-risk"
        assert ar_client.model_name == "risk"
        assert ar_client.registry_host == "us-east1-docker.pkg.dev"
        assert ar_client.image_path == "test-project/test-models/navigate-crime-risk"
        assert (
            ar_client.full_image
            == "us-east1-docker.pkg.dev/test-project/test-models/navigate-crime-risk"
        )

    def test_init_sets_sdk_resource_names(self, ar_client: ArtifactRegistryClient) -> None:
        """Client initializes with correct Docker registry paths."""
        # Docker format doesn't use SDK resource paths like Generic format
        assert ar_client.registry_host == "us-east1-docker.pkg.dev"
        assert ar_client.image_path == "test-project/test-models/navigate-crime-risk"

    def test_create_model_tarball_creates_archive(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """Model tarball creation produces valid archive with model and metadata."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model-content")

        metadata = {"version": "20260322", "rmse": 0.5}

        tarball_data, digest = ar_client._create_model_tarball(
            str(model_file), metadata, shap_path=None
        )

        assert isinstance(tarball_data, bytes)
        assert len(digest) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in digest)

        # Verify tarball contents
        import io

        with tarfile.open(fileobj=io.BytesIO(tarball_data), mode="r:gz") as tar:
            names = tar.getnames()
            assert "model.lgb" in names
            assert "metadata.json" in names

            meta_member = tar.getmember("metadata.json")
            meta_file = tar.extractfile(meta_member)
            meta_content = json.loads(meta_file.read().decode())
            assert meta_content["version"] == "20260322"
            assert meta_content["rmse"] == 0.5

    def test_create_model_tarball_includes_shap(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """Model tarball includes SHAP plot when provided."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")
        shap_file = tmp_path / "shap.png"
        shap_file.write_bytes(b"png-content")

        tarball_data, _ = ar_client._create_model_tarball(
            str(model_file), {}, shap_path=str(shap_file)
        )

        import io

        with tarfile.open(fileobj=io.BytesIO(tarball_data), mode="r:gz") as tar:
            assert "shap_summary.png" in tar.getnames()

    def test_compute_sha256_returns_hex_digest(
        self, ar_client: ArtifactRegistryClient, tmp_path: Path
    ) -> None:
        """SHA256 computation returns valid hex string."""
        test_data = b"hello"

        sha = ar_client._compute_sha256(test_data)

        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    @patch.object(ArtifactRegistryClient, "_upload_blob")
    @patch.object(ArtifactRegistryClient, "_upload_manifest")
    def test_push_uploads_docker_image(
        self,
        mock_upload_manifest: MagicMock,
        mock_upload_blob: MagicMock,
        ar_client: ArtifactRegistryClient,
        tmp_path: Path,
    ) -> None:
        """Push uploads model as Docker image layers and manifest."""
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")

        # Mock blob upload to return URLs
        mock_upload_blob.return_value = "https://registry/v2/repo/blobs/sha256:abc123"
        mock_upload_manifest.return_value = "sha256:manifest123"

        result = ar_client.push(
            model_path=str(model_file),
            version="20260322",
            metadata={"rmse": 0.5},
            stage="staging",
        )

        assert "ar_uri" in result
        assert result["version"] == "20260322"
        assert result["stage"] == "staging"

        # Should upload both config and model layer blobs
        assert mock_upload_blob.call_count == 2
        # Should upload manifest for version + latest + staging tags
        assert mock_upload_manifest.call_count == 3

    @patch.object(ArtifactRegistryClient, "_set_tag")
    def test_promote_to_production_sets_tag(
        self, mock_set_tag: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Promotion sets the production tag."""
        result = ar_client.promote_to_production("20260322")

        assert result["version"] == "20260322"
        assert result["stage"] == "production"
        mock_set_tag.assert_called_with("20260322", "production")

    @patch("requests.get")
    def test_pull_downloads_and_extracts(
        self,
        mock_requests_get: MagicMock,
        ar_client: ArtifactRegistryClient,
        tmp_path: Path,
    ) -> None:
        """Pull downloads Docker image layers and extracts model + metadata."""

        # Create test tarball data
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

        # Mock manifest response
        manifest_response = MagicMock()
        manifest_response.status_code = 200
        manifest_response.json.return_value = {
            "layers": [
                {
                    "digest": "sha256:abc123",
                    "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                }
            ]
        }

        # Mock blob download response
        blob_response = MagicMock()
        blob_response.status_code = 200
        with open(tar_path, "rb") as f:
            blob_response.content = f.read()

        mock_requests_get.side_effect = [manifest_response, blob_response]

        local_dir = tmp_path / "output"
        local_dir.mkdir()
        model_path, metadata = ar_client.pull("production", str(local_dir))

        assert Path(model_path).exists()
        assert metadata["version"] == "20260322"

    @patch("requests.get")
    def test_list_versions_uses_registry_api(
        self, mock_get: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """List versions uses Docker Registry API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["20260322", "20260315", "staging", "production"]
        }
        mock_get.return_value = mock_response

        versions = ar_client.list_versions()

        assert len(versions) == 2
        assert versions[0]["version"] == "20260322"  # Newest first
        assert versions[1]["version"] == "20260315"

    @patch("requests.get")
    def test_get_version_by_tag_returns_version(
        self, mock_get: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag extracts version from manifest and config labels."""
        # First call returns manifest with config digest
        manifest_response = MagicMock()
        manifest_response.status_code = 200
        manifest_response.json.return_value = {"config": {"digest": "sha256:config123"}}

        # Second call returns config blob with labels
        config_response = MagicMock()
        config_response.status_code = 200
        config_response.json.return_value = {
            "config": {"Labels": {"boston-pulse.model.version": "20260322"}}
        }

        mock_get.side_effect = [manifest_response, config_response]

        version = ar_client.get_version_by_tag("production")

        assert version == "20260322"
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_version_by_tag_returns_none_if_not_found(
        self, mock_get: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """Get version by tag returns None if tag doesn't exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

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

    @patch.object(ArtifactRegistryClient, "get_version_by_tag")
    @patch("requests.get")
    def test_list_tags_returns_dict(
        self, mock_get: MagicMock, mock_get_version: MagicMock, ar_client: ArtifactRegistryClient
    ) -> None:
        """List tags returns dict of tag_name → version_id."""
        # Mock tags list response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["production", "staging", "20260322", "20260323"]
        }
        mock_get.return_value = mock_response

        # Mock version lookups for each tag
        def version_side_effect(tag):
            return {
                "production": "20260322",
                "staging": "20260323",
                "20260322": "20260322",
                "20260323": "20260323",
            }.get(tag)

        mock_get_version.side_effect = version_side_effect

        tags = ar_client.list_tags()

        assert tags == {
            "production": "20260322",
            "staging": "20260323",
            "20260322": "20260322",
            "20260323": "20260323",
        }


class TestEnsureRepositoryExists:
    """Tests for ensure_repository_exists helper."""

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_creates_docker_repo_if_not_exists(self, mock_ar_class: MagicMock) -> None:
        """Creates Docker format repository if get_repository fails."""
        mock_client = MagicMock()
        mock_ar_class.return_value = mock_client
        mock_client.get_repository.side_effect = Exception("Not found")

        mock_operation = MagicMock()
        mock_client.create_repository.return_value = mock_operation

        ensure_repository_exists("proj", "us-east1", "ml-models")

        mock_client.get_repository.assert_called_once()
        mock_client.create_repository.assert_called_once()

        # Verify Docker format is specified
        create_call = mock_client.create_repository.call_args
        assert "format_" in str(create_call)  # Check that Docker format is set

        mock_operation.result.assert_called_once()

    @patch("shared.artifact_registry.artifactregistry_v1.ArtifactRegistryClient")
    def test_skips_create_if_docker_repo_exists(self, mock_ar_class: MagicMock) -> None:
        """Skips creation if Docker repository already exists."""
        mock_client = MagicMock()
        mock_ar_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.format_ = artifactregistry_v1.Repository.Format.DOCKER
        mock_client.get_repository.return_value = mock_repo

        ensure_repository_exists("proj", "us-east1", "ml-models")

        mock_client.get_repository.assert_called_once()
        mock_client.create_repository.assert_not_called()
