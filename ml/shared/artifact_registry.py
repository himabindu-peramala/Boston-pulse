"""
Boston Pulse ML - Artifact Registry Client.

Pushes model artifacts to GCP Artifact Registry using the Python SDK.
No gcloud CLI dependency — uses Application Default Credentials (ADC).

Works identically on:
  - GCE VM (service account via metadata server at 169.254.169.254)
  - Local Mac (gcloud auth application-default login)
  - GitHub Actions (Workload Identity Federation)
  - Any environment with ADC configured

This is for DEPLOYABLE artifacts only — model.lgb and metadata.json.
Run artifacts (features.parquet, scores.parquet, etc.) stay in GCS.

Repository structure with namespaced packages:
  ml-models/                              (repository)
    ├── navigate/crime-risk               (package - current model)
    │   ├── 20260316                      (version)
    │   ├── 20260322                      (version)
    │   └── production → 20260322         (tag)
    ├── navigate/transit-risk             (package - future)
    └── chatbot/intent-model              (package - future)

Package naming convention: {domain}/{model-purpose}
  - domain: product area (navigate, chatbot, etc.)
  - model-purpose: what the model predicts (crime-risk, transit-risk, etc.)

Versioning pattern:
  - Dated versions (e.g., "20260322") are immutable snapshots
  - Tags: "staging" → "production" promotion flow
  - "latest" tag always points to most recent successful training

Stage promotion flow:
  1. Train → push with "staging" tag + dated version
  2. Gates pass → promote to "production" tag
  3. Production service pulls "production" tag at startup
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import google.auth
import google.auth.transport.requests
import requests
from google.cloud import artifactregistry_v1

logger = logging.getLogger(__name__)


class ArtifactRegistryError(Exception):
    """Raised when AR operations fail."""

    pass


class ArtifactRegistryClient:
    """
    Client for GCP Artifact Registry Generic repositories.

    Uses Python SDK only — no gcloud CLI dependency.
    Authentication is handled by google-auth library via ADC.

    Provides versioned model storage with stage-based promotion:
    - staging: newly trained model, not yet validated for production
    - production: validated model, safe for serving
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize AR client.

        Args:
            cfg: Training config with registry section containing:
                - project: GCP project ID
                - location: AR location (e.g., us-east1)
                - repository: AR repository name
                - package: Namespaced package name (e.g., "navigate/crime-risk")
        """
        registry_cfg = cfg.get("registry", {})
        self.project = registry_cfg.get("project", "bostonpulse")
        self.location = registry_cfg.get("location", "us-east1")
        self.repository = registry_cfg.get("repository", "ml-models")

        # Package name uses namespace convention: {domain}/{model-purpose}
        self.package = registry_cfg.get(
            "package", cfg.get("model", {}).get("name", "navigate/crime-risk")
        )

        # For display and file naming, use the model-purpose part
        self.model_name = self.package.split("/")[-1] if "/" in self.package else self.package

        self.ar_host = f"{self.location}-generic.pkg.dev"
        self.ar_path = f"{self.project}/{self.repository}/{self.package}"

        # Fully qualified AR resource names for SDK calls
        self._parent = (
            f"projects/{self.project}/locations/{self.location}" f"/repositories/{self.repository}"
        )
        self._package_parent = f"{self._parent}/packages/{self.package}"

        # AR client for tag/version management — uses ADC automatically
        self._ar_client: artifactregistry_v1.ArtifactRegistryClient | None = None

    @property
    def ar_client(self) -> artifactregistry_v1.ArtifactRegistryClient:
        """Lazy-load AR client to avoid import overhead if not needed."""
        if self._ar_client is None:
            self._ar_client = artifactregistry_v1.ArtifactRegistryClient()
        return self._ar_client

    def _get_credentials(self) -> google.auth.credentials.Credentials:
        """Get ADC credentials with cloud-platform scope."""
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials

    def _create_artifact_package(
        self,
        model_path: str,
        metadata: dict[str, Any],
        shap_path: str | None = None,
    ) -> str:
        """
        Create a tarball package containing model + metadata.

        Returns path to the created tarball.
        """
        import tarfile

        tmpdir = tempfile.mkdtemp()
        package_path = Path(tmpdir) / f"{self.model_name}.tar.gz"

        with tarfile.open(package_path, "w:gz") as tar:
            tar.add(model_path, arcname="model.lgb")

            meta_path = Path(tmpdir) / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2, default=str))
            tar.add(meta_path, arcname="metadata.json")

            if shap_path and Path(shap_path).exists():
                tar.add(shap_path, arcname="shap_summary.png")

        return str(package_path)

    def _compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _upload_generic_artifact(self, package_path: str, version: str) -> None:
        """
        Upload a tarball to AR as a generic artifact using the REST API.

        The AR Python SDK doesn't have a direct upload method for generic artifacts,
        so we use the REST API with ADC credentials.
        """
        credentials = self._get_credentials()

        # AR generic upload endpoint (v1beta2 for generic artifacts)
        # URL-encode the package name (slashes become %2F)
        _encoded_package = self.package.replace("/", "%2F")
        upload_url = (
            f"https://{self.location}-artifactregistry.googleapis.com/v1beta2"
            f"/projects/{self.project}/locations/{self.location}"
            f"/repositories/{self.repository}/genericArtifacts:create"
        )

        with open(package_path, "rb") as f:
            file_content = f.read()

        # The filename in AR will be the tarball name
        filename = Path(package_path).name

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/octet-stream",
        }

        params = {
            "packageId": self.package,
            "versionId": version,
            "filename": filename,
        }

        response = requests.post(
            upload_url,
            headers=headers,
            params=params,
            data=file_content,
            timeout=300,
        )

        if response.status_code not in (200, 201):
            raise ArtifactRegistryError(f"AR upload failed: {response.status_code} {response.text}")

        logger.debug(f"AR upload response: {response.status_code}")

    def push(
        self,
        model_path: str,
        version: str,
        metadata: dict[str, Any],
        stage: str = "staging",
        shap_path: str | None = None,
    ) -> dict[str, str]:
        """
        Push model to Artifact Registry.

        Args:
            model_path: Local path to model.lgb file
            version: Version string (e.g., "20260322")
            metadata: Model metadata dict
            stage: Initial stage tag ("staging" or "production")
            shap_path: Optional path to SHAP summary plot

        Returns:
            Dict with artifact URIs and version info
        """
        full_metadata = {
            **metadata,
            "version": version,
            "package": self.package,
            "model_name": self.model_name,
            "stage": stage,
        }

        package_path = self._create_artifact_package(model_path, full_metadata, shap_path)
        sha256 = self._compute_sha256(package_path)

        ar_uri = f"https://{self.ar_host}/{self.ar_path}"

        self._upload_generic_artifact(package_path, version)

        logger.info(f"Pushed model to AR: {ar_uri}:{version}")

        self._set_tag(version, version)

        if stage == "staging":
            self._set_tag(version, "staging")
        elif stage == "production":
            self._set_tag(version, "staging")
            self._set_tag(version, "production")

        self._set_tag(version, "latest")

        Path(package_path).unlink(missing_ok=True)

        return {
            "ar_uri": f"{ar_uri}:{version}",
            "version": version,
            "stage": stage,
            "sha256": sha256,
            "ar_path": f"{self.ar_path}:{version}",
        }

    def _set_tag(self, version: str, tag: str) -> None:
        """Set a tag pointing to a specific version using the Python SDK."""
        tag_name = f"{self._package_parent}/tags/{tag}"
        version_name = f"{self._package_parent}/versions/{version}"

        try:
            existing_tag = self.ar_client.get_tag(name=tag_name)
            # Tag exists — update it
            existing_tag.version = version_name
            update_request = artifactregistry_v1.UpdateTagRequest(
                tag=existing_tag,
                update_mask={"paths": ["version"]},
            )
            self.ar_client.update_tag(request=update_request)
            logger.debug(f"Updated tag '{tag}' → {version}")

        except Exception:
            # Tag does not exist — create it
            new_tag = artifactregistry_v1.Tag(
                name=tag_name,
                version=version_name,
            )
            create_request = artifactregistry_v1.CreateTagRequest(
                parent=self._package_parent,
                tag_id=tag,
                tag=new_tag,
            )
            self.ar_client.create_tag(request=create_request)
            logger.debug(f"Created tag '{tag}' → {version}")

        logger.info(f"Set tag '{tag}' → version {version}")

    def promote_to_production(self, version: str) -> dict[str, str]:
        """
        Promote a staged model to production.

        Args:
            version: Version to promote

        Returns:
            Dict with promotion info
        """
        self._set_tag(version, "production")

        logger.info(f"Promoted version {version} to production")

        return {
            "version": version,
            "stage": "production",
            "ar_path": f"{self.ar_path}:production",
        }

    def pull(
        self,
        version_or_tag: str = "production",
        local_dir: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Pull model from Artifact Registry.

        Args:
            version_or_tag: Version string or tag ("staging", "production", "latest")
            local_dir: Directory to extract to (creates temp if not provided)

        Returns:
            (local model path, metadata dict)
        """
        import tarfile

        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        # Resolve tag to version if needed
        actual_version = version_or_tag
        if version_or_tag in ("staging", "production", "latest"):
            resolved = self.get_version_by_tag(version_or_tag)
            if resolved:
                actual_version = resolved
            else:
                raise ArtifactRegistryError(
                    f"Tag '{version_or_tag}' not found or does not point to a version"
                )

        # Download using REST API
        self._download_generic_artifact(actual_version, local_dir)

        tar_files = list(Path(local_dir).glob("*.tar.gz"))
        if not tar_files:
            raise ArtifactRegistryError(f"No package found after download for {version_or_tag}")

        with tarfile.open(tar_files[0], "r:gz") as tar:
            tar.extractall(local_dir)

        model_path = str(Path(local_dir) / "model.lgb")
        meta_path = Path(local_dir) / "metadata.json"

        if not Path(model_path).exists():
            raise ArtifactRegistryError("model.lgb not found in package")

        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        logger.info(f"Pulled model from AR: {version_or_tag}")
        return model_path, metadata

    def _download_generic_artifact(self, version: str, local_dir: str) -> None:
        """
        Download a generic artifact from AR using the REST API.

        Lists files in the version and downloads each one.
        """
        credentials = self._get_credentials()

        # List files in the version
        version_name = f"{self._package_parent}/versions/{version}"

        try:
            _version_obj = self.ar_client.get_version(name=version_name)
        except Exception as e:
            raise ArtifactRegistryError(f"Version '{version}' not found: {e}") from e

        # Get the list of files via REST API
        list_url = (
            f"https://{self.location}-artifactregistry.googleapis.com/v1beta2"
            f"/{version_name}/files"
        )

        headers = {"Authorization": f"Bearer {credentials.token}"}
        response = requests.get(list_url, headers=headers, timeout=60)

        if response.status_code != 200:
            # Fallback: try to download the package directly by name pattern
            download_url = (
                f"https://{self.location}-artifactregistry.googleapis.com/v1beta2"
                f"/{version_name}:download"
            )
            response = requests.get(download_url, headers=headers, stream=True, timeout=300)

            if response.status_code == 200:
                dest_path = Path(local_dir) / f"{self.model_name}.tar.gz"
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return

            raise ArtifactRegistryError(
                f"Failed to list/download files for version '{version}': "
                f"{response.status_code} {response.text}"
            )

        files_data = response.json()
        files = files_data.get("files", [])

        if not files:
            raise ArtifactRegistryError(f"No files found in version '{version}'")

        # Download each file
        for file_info in files:
            file_name = file_info.get("name", "").split("/")[-1]
            if not file_name:
                continue

            download_url = (
                f"https://{self.location}-artifactregistry.googleapis.com/v1beta2"
                f"/{file_info['name']}:download"
            )

            response = requests.get(download_url, headers=headers, stream=True, timeout=300)
            if response.status_code != 200:
                logger.warning(f"Failed to download {file_name}: {response.status_code}")
                continue

            dest_path = Path(local_dir) / file_name
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.debug(f"Downloaded {file_name} to {dest_path}")

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all available model versions.

        Returns:
            List of version info dicts
        """
        request = artifactregistry_v1.ListVersionsRequest(parent=self._package_parent)

        versions = []
        try:
            for version in self.ar_client.list_versions(request=request):
                version_id = version.name.split("/")[-1]
                # Skip tag-like entries
                if version_id not in ("staging", "production", "latest"):
                    versions.append(
                        {
                            "version": version_id,
                            "createTime": (
                                version.create_time.isoformat() if version.create_time else None
                            ),
                            "updateTime": (
                                version.update_time.isoformat() if version.update_time else None
                            ),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to list versions: {e}")
            return []

        return versions

    def get_version_by_tag(self, tag: str) -> str | None:
        """
        Get the version that a tag points to.

        Args:
            tag: Tag name ("staging", "production", "latest")

        Returns:
            Version string or None if tag doesn't exist
        """
        tag_name = f"{self._package_parent}/tags/{tag}"

        try:
            tag_obj = self.ar_client.get_tag(name=tag_name)
            # tag.version is the full resource name — extract the ID
            return tag_obj.version.split("/")[-1] if tag_obj.version else None
        except Exception as e:
            logger.debug(f"Tag '{tag}' not found: {e}")
            return None

    def list_tags(self) -> dict[str, str]:
        """Return dict of tag_name → version_id."""
        request = artifactregistry_v1.ListTagsRequest(parent=self._package_parent)

        tags = {}
        try:
            for tag in self.ar_client.list_tags(request=request):
                tag_id = tag.name.split("/")[-1]
                version_id = tag.version.split("/")[-1] if tag.version else None
                if version_id:
                    tags[tag_id] = version_id
        except Exception as e:
            logger.warning(f"Failed to list tags: {e}")

        return tags

    def rollback(self, version: str) -> dict[str, str]:
        """
        Rollback production to a specific version.

        Args:
            version: Version to rollback to

        Returns:
            Dict with rollback info
        """
        return self.promote_to_production(version)


def ensure_repository_exists(
    project: str,
    location: str,
    repository: str,
) -> None:
    """
    Ensure the AR repository exists, create if not.

    Uses Python SDK — no gcloud CLI needed.
    Call this during infrastructure setup, not during training.
    """
    client = artifactregistry_v1.ArtifactRegistryClient()
    repo_name = f"projects/{project}/locations/{location}/repositories/{repository}"

    try:
        client.get_repository(name=repo_name)
        logger.debug(f"AR repository exists: {repository}")
    except Exception:
        logger.info(f"Creating AR repository: {repository}")
        parent = f"projects/{project}/locations/{location}"
        repo = artifactregistry_v1.Repository(
            name=repo_name,
            format_=artifactregistry_v1.Repository.Format.GENERIC,
            description="Boston Pulse ML model artifacts",
        )
        request = artifactregistry_v1.CreateRepositoryRequest(
            parent=parent,
            repository_id=repository,
            repository=repo,
        )
        operation = client.create_repository(request=request)
        operation.result()  # Wait for completion
        logger.info(f"Created AR repository: {repository}")
