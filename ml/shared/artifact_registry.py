"""
Boston Pulse ML - Artifact Registry Client.

Pushes model artifacts to GCP Artifact Registry.
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
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactRegistryError(Exception):
    """Raised when AR operations fail."""

    pass


class ArtifactRegistryClient:
    """
    Client for GCP Artifact Registry Generic repositories.

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
        # e.g., "navigate/crime-risk", "navigate/transit-risk", "chatbot/intent-model"
        self.package = registry_cfg.get(
            "package", cfg.get("model", {}).get("name", "navigate/crime-risk")
        )

        # For display and file naming, use the model-purpose part
        self.model_name = self.package.split("/")[-1] if "/" in self.package else self.package

        self.ar_host = f"{self.location}-generic.pkg.dev"
        self.ar_path = f"{self.project}/{self.repository}/{self.package}"

    def _run_gcloud(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a gcloud command."""
        cmd = ["gcloud"] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise ArtifactRegistryError(f"gcloud command failed: {result.stderr or result.stdout}")
        return result

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

        self._run_gcloud(
            [
                "artifacts",
                "generic",
                "upload",
                f"--project={self.project}",
                f"--location={self.location}",
                f"--repository={self.repository}",
                f"--package={self.package}",
                f"--version={version}",
                f"--source={package_path}",
            ]
        )

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
        """Set a tag pointing to a specific version."""
        try:
            self._run_gcloud(
                [
                    "artifacts",
                    "tags",
                    "create",
                    tag,
                    f"--project={self.project}",
                    f"--location={self.location}",
                    f"--repository={self.repository}",
                    f"--package={self.package}",
                    f"--version={version}",
                ],
                check=False,
            )
        except ArtifactRegistryError:
            self._run_gcloud(
                [
                    "artifacts",
                    "tags",
                    "update",
                    tag,
                    f"--project={self.project}",
                    f"--location={self.location}",
                    f"--repository={self.repository}",
                    f"--package={self.package}",
                    f"--version={version}",
                ]
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

        _download_path = Path(local_dir) / "package.tar.gz"

        self._run_gcloud(
            [
                "artifacts",
                "generic",
                "download",
                f"--project={self.project}",
                f"--location={self.location}",
                f"--repository={self.repository}",
                f"--package={self.package}",
                f"--version={version_or_tag}",
                f"--destination={local_dir}",
            ]
        )

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

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all available model versions.

        Returns:
            List of version info dicts
        """
        result = self._run_gcloud(
            [
                "artifacts",
                "versions",
                "list",
                f"--project={self.project}",
                f"--location={self.location}",
                f"--repository={self.repository}",
                f"--package={self.package}",
                "--format=json",
            ]
        )

        if result.stdout:
            versions = json.loads(result.stdout)
            return [
                {
                    "version": v.get("name", "").split("/")[-1],
                    "createTime": v.get("createTime"),
                    "updateTime": v.get("updateTime"),
                }
                for v in versions
            ]
        return []

    def get_version_by_tag(self, tag: str) -> str | None:
        """
        Get the version that a tag points to.

        Args:
            tag: Tag name ("staging", "production", "latest")

        Returns:
            Version string or None if tag doesn't exist
        """
        result = self._run_gcloud(
            [
                "artifacts",
                "tags",
                "list",
                f"--project={self.project}",
                f"--location={self.location}",
                f"--repository={self.repository}",
                f"--package={self.package}",
                "--format=json",
            ],
            check=False,
        )

        if result.stdout:
            tags = json.loads(result.stdout)
            for t in tags:
                if t.get("name", "").endswith(f"/{tag}"):
                    version_path = t.get("version", "")
                    return version_path.split("/")[-1] if version_path else None
        return None

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

    Call this during infrastructure setup, not during training.
    """
    result = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            repository,
            f"--project={project}",
            f"--location={location}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.info(f"Creating AR repository: {repository}")
        subprocess.run(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                repository,
                f"--project={project}",
                f"--location={location}",
                "--repository-format=generic",
                "--description=Boston Pulse ML model artifacts",
            ],
            check=True,
        )
        logger.info(f"Created AR repository: {repository}")
    else:
        logger.debug(f"AR repository exists: {repository}")
