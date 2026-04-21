"""
Boston Pulse ML - Artifact Registry Client (Docker Format).

Pushes model artifacts to GCP Artifact Registry Docker repositories.
Uses Docker Registry HTTP API v2 directly — no Docker daemon required.

Works identically on:
  - GCE VM (service account via metadata server)
  - Local Mac (gcloud auth application-default login)
  - GitHub Actions (Workload Identity Federation)
  - Cloud Run (service account)

Repository structure:
  ml-models/                              (Docker repository)
    └── navigate-crime-risk               (image name)
        ├── 20260316                      (tag - dated version)
        ├── 20260322                      (tag - dated version)
        ├── staging                       (tag - points to latest staged)
        ├── production                    (tag - points to promoted version)
        └── latest                        (tag - most recent)

Stage promotion flow:
  1. Train → push with version tag + "staging" + "latest" tags
  2. Gates pass → add "production" tag to same digest
  3. Production service pulls ":production" tag at startup
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from datetime import UTC, datetime
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
    Client for GCP Artifact Registry Docker repositories.

    Uses Docker Registry HTTP API v2 directly — no Docker daemon needed.
    Authentication via Google ADC (Application Default Credentials).

    Provides versioned model storage with tag-based promotion:
    - staging: newly trained model, not yet validated
    - production: validated model, safe for serving
    - latest: most recent successful training
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize AR client.

        Args:
            cfg: Training config with registry section containing:
                - project: GCP project ID
                - location: AR location (e.g., us-east1)
                - repository: AR repository name
                - package: Image name (e.g., "navigate-crime-risk")
        """
        registry_cfg = cfg.get("registry", {})
        self.project = registry_cfg.get("project", "bostonpulse")
        self.location = registry_cfg.get("location", "us-east1")
        self.repository = registry_cfg.get("repository", "ml-models")

        # Image name — use dashes, not slashes (Docker naming convention)
        package = registry_cfg.get(
            "package", cfg.get("model", {}).get("name", "navigate-crime-risk")
        )
        # Convert any slashes to dashes for Docker compatibility
        self.image_name = package.replace("/", "-")
        self.model_name = (
            self.image_name.split("-")[-1] if "-" in self.image_name else self.image_name
        )

        # Docker registry host and path
        self.registry_host = f"{self.location}-docker.pkg.dev"
        self.image_path = f"{self.project}/{self.repository}/{self.image_name}"
        self.full_image = f"{self.registry_host}/{self.image_path}"

        # AR SDK client for listing (optional, used for version queries)
        self._ar_client: artifactregistry_v1.ArtifactRegistryClient | None = None

        # Cache credentials
        self._credentials = None
        self._token_expiry = None

    @property
    def ar_client(self) -> artifactregistry_v1.ArtifactRegistryClient:
        """Lazy-load AR SDK client."""
        if self._ar_client is None:
            self._ar_client = artifactregistry_v1.ArtifactRegistryClient()
        return self._ar_client

    def _get_auth_token(self) -> str:
        """Get OAuth2 token for Docker Registry authentication."""
        if self._credentials is None or self._is_token_expired():
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            self._credentials = credentials
            self._token_expiry = credentials.expiry

        return self._credentials.token

    def _is_token_expired(self) -> bool:
        """Check if cached token is expired."""
        if self._token_expiry is None:
            return True
        # Handle both timezone-aware and naive datetimes from google-auth
        now = datetime.now(UTC)
        expiry = self._token_expiry
        if expiry.tzinfo is None:
            # Assume UTC if naive
            expiry = expiry.replace(tzinfo=UTC)
        return now >= expiry

    def _get_auth_header(self) -> dict[str, str]:
        """Get Authorization header for Docker Registry API."""
        token = self._get_auth_token()
        return {"Authorization": f"Bearer {token}"}

    def _compute_sha256(self, data: bytes) -> str:
        """Compute SHA256 hash of bytes."""
        return hashlib.sha256(data).hexdigest()

    def _create_model_tarball(
        self,
        model_path: str,
        metadata: dict[str, Any],
        shap_path: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Create a tarball containing model + metadata.

        Returns:
            (tarball_bytes, sha256_digest)
        """
        import io
        import tarfile

        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            # Add model file
            tar.add(model_path, arcname="model.lgb")

            # Add metadata as JSON
            meta_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")
            meta_info = tarfile.TarInfo(name="metadata.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

            # Add SHAP plot if provided
            if shap_path and Path(shap_path).exists():
                tar.add(shap_path, arcname="shap_summary.png")

        tarball_bytes = buffer.getvalue()
        digest = self._compute_sha256(tarball_bytes)

        return tarball_bytes, digest

    def _upload_blob(self, data: bytes, digest: str) -> str:
        """
        Upload a blob to the Docker registry.

        Uses Docker Registry HTTP API v2 blob upload.

        Returns:
            The blob digest (sha256:...)
        """
        full_digest = f"sha256:{digest}"

        # Check if blob already exists
        check_url = f"https://{self.registry_host}/v2/{self.image_path}/blobs/{full_digest}"
        headers = self._get_auth_header()

        response = requests.head(check_url, headers=headers, timeout=30)
        if response.status_code == 200:
            logger.debug(f"Blob {digest[:12]} already exists")
            return full_digest

        # Start upload session
        upload_url = f"https://{self.registry_host}/v2/{self.image_path}/blobs/uploads/"
        response = requests.post(upload_url, headers=headers, timeout=30)

        if response.status_code not in (202,):
            raise ArtifactRegistryError(
                f"Failed to start blob upload: {response.status_code} {response.text}"
            )

        # Get the upload location from response header
        location = response.headers.get("Location")
        if not location:
            raise ArtifactRegistryError("No upload location returned")

        # Ensure location is absolute
        if not location.startswith("http"):
            location = f"https://{self.registry_host}{location}"

        # Upload the blob in one request (monolithic upload)
        separator = "&" if "?" in location else "?"
        put_url = f"{location}{separator}digest={full_digest}"

        headers = self._get_auth_header()
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(len(data))

        response = requests.put(put_url, headers=headers, data=data, timeout=300)

        if response.status_code not in (201,):
            raise ArtifactRegistryError(
                f"Failed to upload blob: {response.status_code} {response.text}"
            )

        logger.debug(f"Uploaded blob {digest[:12]} ({len(data)} bytes)")
        return full_digest

    def _create_config_blob(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        """
        Create OCI config blob for the image.

        This contains image metadata in OCI format.
        """
        config = {
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Labels": {
                    "org.opencontainers.image.title": self.model_name,
                    "org.opencontainers.image.version": metadata.get("version", "unknown"),
                    "boston-pulse.model.name": self.model_name,
                    "boston-pulse.model.version": metadata.get("version", "unknown"),
                    "boston-pulse.model.stage": metadata.get("stage", "staging"),
                    "boston-pulse.model.val_rmse": str(metadata.get("val_rmse", "")),
                }
            },
            "rootfs": {"type": "layers", "diff_ids": []},  # Will be populated after layer upload
            "history": [
                {
                    "created": datetime.now(UTC).isoformat(),
                    "comment": f"Boston Pulse ML model: {self.model_name}",
                }
            ],
        }

        config_bytes = json.dumps(config, separators=(",", ":")).encode("utf-8")
        digest = self._compute_sha256(config_bytes)

        return config_bytes, digest

    def _upload_manifest(
        self, config_digest: str, config_size: int, layer_digest: str, layer_size: int, tag: str
    ) -> str:
        """
        Upload OCI image manifest with a specific tag.

        Returns:
            The manifest digest
        """
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": config_size,
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": layer_digest,
                    "size": layer_size,
                }
            ],
        }

        manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        manifest_digest = f"sha256:{self._compute_sha256(manifest_bytes)}"

        # Push manifest with tag
        manifest_url = f"https://{self.registry_host}/v2/{self.image_path}/manifests/{tag}"

        headers = self._get_auth_header()
        headers["Content-Type"] = "application/vnd.oci.image.manifest.v1+json"

        response = requests.put(manifest_url, headers=headers, data=manifest_bytes, timeout=60)

        if response.status_code not in (201, 200):
            raise ArtifactRegistryError(
                f"Failed to upload manifest for tag '{tag}': {response.status_code} {response.text}"
            )

        logger.debug(f"Uploaded manifest with tag '{tag}'")
        return manifest_digest

    def push(
        self,
        model_path: str,
        version: str,
        metadata: dict[str, Any],
        stage: str = "staging",
        shap_path: str | None = None,
    ) -> dict[str, str]:
        """
        Push model to Artifact Registry as a Docker image.

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
            "image_name": self.image_name,
            "model_name": self.model_name,
            "stage": stage,
            "pushed_at": datetime.now(UTC).isoformat(),
        }

        # Create model layer (tarball)
        layer_bytes, layer_sha = self._create_model_tarball(model_path, full_metadata, shap_path)

        # Create config blob
        config_bytes, config_sha = self._create_config_blob(full_metadata)

        # Upload blobs
        layer_digest = self._upload_blob(layer_bytes, layer_sha)
        config_digest = self._upload_blob(config_bytes, config_sha)

        # Upload manifest with version tag
        manifest_digest = self._upload_manifest(
            config_digest, len(config_bytes), layer_digest, len(layer_bytes), version
        )

        logger.info(f"Pushed model to AR: {self.full_image}:{version}")

        # Add additional tags
        tags_to_add = ["latest"]
        if stage == "staging":
            tags_to_add.append("staging")
        elif stage == "production":
            tags_to_add.extend(["staging", "production"])

        for tag in tags_to_add:
            self._upload_manifest(
                config_digest, len(config_bytes), layer_digest, len(layer_bytes), tag
            )
            logger.info(f"Tagged {self.full_image}:{tag}")

        return {
            "ar_uri": f"{self.full_image}:{version}",
            "version": version,
            "stage": stage,
            "manifest_digest": manifest_digest,
            "ar_path": f"{self.image_path}:{version}",
        }

    def _set_tag(self, version: str, tag: str) -> None:
        """
        Set a tag pointing to a specific version.

        For Docker repos, this means copying the manifest to the new tag.
        """
        # Get the manifest for the source version
        source_url = f"https://{self.registry_host}/v2/{self.image_path}/manifests/{version}"

        headers = self._get_auth_header()
        headers["Accept"] = (
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
        )

        response = requests.get(source_url, headers=headers, timeout=30)

        if response.status_code != 200:
            raise ArtifactRegistryError(
                f"Failed to get manifest for version '{version}': {response.status_code}"
            )

        manifest_bytes = response.content
        content_type = response.headers.get(
            "Content-Type", "application/vnd.oci.image.manifest.v1+json"
        )

        # Push manifest with new tag
        target_url = f"https://{self.registry_host}/v2/{self.image_path}/manifests/{tag}"

        headers = self._get_auth_header()
        headers["Content-Type"] = content_type

        response = requests.put(target_url, headers=headers, data=manifest_bytes, timeout=60)

        if response.status_code not in (201, 200):
            raise ArtifactRegistryError(
                f"Failed to set tag '{tag}': {response.status_code} {response.text}"
            )

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
            "ar_path": f"{self.image_path}:production",
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
        import io
        import tarfile

        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        # Get manifest
        manifest_url = (
            f"https://{self.registry_host}/v2/{self.image_path}/manifests/{version_or_tag}"
        )

        headers = self._get_auth_header()
        headers["Accept"] = (
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
        )

        response = requests.get(manifest_url, headers=headers, timeout=30)

        if response.status_code != 200:
            raise ArtifactRegistryError(
                f"Failed to get manifest for '{version_or_tag}': {response.status_code}"
            )

        manifest = response.json()

        # Get layer digest (our model tarball)
        layers = manifest.get("layers", [])
        if not layers:
            raise ArtifactRegistryError(f"No layers in manifest for '{version_or_tag}'")

        layer_digest = layers[0]["digest"]

        # Download layer blob
        blob_url = f"https://{self.registry_host}/v2/{self.image_path}/blobs/{layer_digest}"
        response = requests.get(blob_url, headers=self._get_auth_header(), stream=True, timeout=300)

        if response.status_code != 200:
            raise ArtifactRegistryError(f"Failed to download layer: {response.status_code}")

        # Extract tarball
        tarball_bytes = response.content
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            tar.extractall(local_dir)

        model_path = str(Path(local_dir) / "model.lgb")
        meta_path = Path(local_dir) / "metadata.json"

        if not Path(model_path).exists():
            raise ArtifactRegistryError("model.lgb not found in image")

        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        logger.info(f"Pulled model from AR: {version_or_tag}")
        return model_path, metadata

    def get_version_by_tag(self, tag: str) -> str | None:
        """
        Get the version that a tag points to.

        For Docker repos, we extract version from image labels.

        Args:
            tag: Tag name ("staging", "production", "latest")

        Returns:
            Version string or None if tag doesn't exist
        """
        try:
            # Get manifest
            manifest_url = f"https://{self.registry_host}/v2/{self.image_path}/manifests/{tag}"

            headers = self._get_auth_header()
            headers["Accept"] = (
                "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
            )

            response = requests.get(manifest_url, headers=headers, timeout=30)

            if response.status_code != 200:
                return None

            manifest = response.json()
            config_digest = manifest.get("config", {}).get("digest")

            if not config_digest:
                return None

            # Get config blob to read labels
            config_url = f"https://{self.registry_host}/v2/{self.image_path}/blobs/{config_digest}"
            response = requests.get(config_url, headers=self._get_auth_header(), timeout=30)

            if response.status_code != 200:
                return None

            config = response.json()
            labels = config.get("config", {}).get("Labels", {})

            return labels.get("boston-pulse.model.version")

        except Exception as e:
            logger.debug(f"Failed to get version for tag '{tag}': {e}")
            return None

    def list_tags(self) -> dict[str, str]:
        """Return dict of tag_name → version (from labels)."""
        tags = {}

        try:
            # List tags via Docker Registry API
            tags_url = f"https://{self.registry_host}/v2/{self.image_path}/tags/list"
            response = requests.get(tags_url, headers=self._get_auth_header(), timeout=30)

            if response.status_code != 200:
                return tags

            tag_list = response.json().get("tags", [])

            for tag in tag_list:
                version = self.get_version_by_tag(tag)
                if version:
                    tags[tag] = version

        except Exception as e:
            logger.warning(f"Failed to list tags: {e}")

        return tags

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all available model versions (dated tags).

        Returns:
            List of version info dicts
        """
        versions = []

        try:
            tags_url = f"https://{self.registry_host}/v2/{self.image_path}/tags/list"
            response = requests.get(tags_url, headers=self._get_auth_header(), timeout=30)

            if response.status_code != 200:
                return versions

            tag_list = response.json().get("tags", [])

            # Filter to dated versions (YYYYMMDD format)
            for tag in tag_list:
                if tag.isdigit() and len(tag) == 8:
                    versions.append(
                        {
                            "version": tag,
                            "tag": tag,
                        }
                    )

        except Exception as e:
            logger.warning(f"Failed to list versions: {e}")

        return sorted(versions, key=lambda x: x["version"], reverse=True)

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
    Ensure the AR Docker repository exists, create if not.

    Uses Python SDK — no gcloud CLI needed.
    Call this during infrastructure setup, not during training.
    """
    client = artifactregistry_v1.ArtifactRegistryClient()
    repo_name = f"projects/{project}/locations/{location}/repositories/{repository}"

    try:
        repo = client.get_repository(name=repo_name)
        # Check if it's Docker format
        if repo.format_ != artifactregistry_v1.Repository.Format.DOCKER:
            logger.warning(
                f"Repository {repository} exists but is not Docker format. "
                f"Current format: {repo.format_}. Please recreate as Docker."
            )
        else:
            logger.debug(f"AR Docker repository exists: {repository}")
    except Exception:
        logger.info(f"Creating AR Docker repository: {repository}")
        parent = f"projects/{project}/locations/{location}"
        repo = artifactregistry_v1.Repository(
            name=repo_name,
            format_=artifactregistry_v1.Repository.Format.DOCKER,
            description="Boston Pulse ML model artifacts (Docker format)",
        )
        request = artifactregistry_v1.CreateRepositoryRequest(
            parent=parent,
            repository_id=repository,
            repository=repo,
        )
        operation = client.create_repository(request=request)
        operation.result()
        logger.info(f"Created AR Docker repository: {repository}")
