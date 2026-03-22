"""
Boston Pulse ML - Model Registry.

GCP Artifact Registry push and pull for LightGBM model files.

Versioning pattern:
  dated/:  always written — permanent record for forensics/rollback
  latest/: only updated after ALL gates pass

This is the two-layer rollback mechanism:
  Layer 1: latest/ never updated if a gate fails — production never sees bad model
  Layer 2: every dated version is permanently available for manual re-pointing
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from google.cloud import storage

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    GCS-backed model registry for LightGBM models.

    Provides versioned storage with a latest/ pointer that only updates
    after all validation gates pass.
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize registry.

        Args:
            cfg: Training config with registry section
        """
        self.project = cfg.get("registry", {}).get("project", "boston-pulse")
        self.location = cfg.get("registry", {}).get("location", "us-east1")
        self.repository = cfg.get("registry", {}).get("repository", "ml-models")
        self.model_name = cfg.get("model", {}).get("name", "crime-navigate-model")
        self.bucket_name = cfg.get("registry", {}).get(
            "artifact_bucket", "boston-pulse-mlflow-artifacts"
        )
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def _model_prefix(self, version: str) -> str:
        """Get GCS prefix for a model version."""
        return f"registry/{self.model_name}/{version}"

    def push(
        self,
        model_path: str,
        version: str,
        metadata: dict[str, Any],
        update_latest: bool = True,
        shap_path: str | None = None,
    ) -> str:
        """
        Push model to GCS-backed registry.

        Always writes to dated/{version}/.
        Only writes to latest/ if update_latest=True (both gates must pass first).

        Args:
            model_path: Local path to model.lgb file
            version: Version string (e.g. "20260316")
            metadata: Model metadata (rmse, bias_passed, git_sha, feature_list, etc.)
            update_latest: Whether to update the latest/ pointer
            shap_path: Optional path to SHAP summary plot

        Returns:
            GCS path of dated version
        """
        dated_prefix = self._model_prefix(version)

        # Upload model file
        model_blob = self.bucket.blob(f"{dated_prefix}/model.lgb")
        model_blob.upload_from_filename(model_path)
        logger.info(f"Uploaded model to gs://{self.bucket_name}/{dated_prefix}/model.lgb")

        # Upload metadata
        full_metadata = {
            **metadata,
            "version": version,
            "model_name": self.model_name,
        }
        meta_blob = self.bucket.blob(f"{dated_prefix}/metadata.json")
        meta_blob.upload_from_string(
            json.dumps(full_metadata, indent=2, default=str),
            content_type="application/json",
        )

        # Upload SHAP plot if provided
        if shap_path and Path(shap_path).exists():
            shap_blob = self.bucket.blob(f"{dated_prefix}/shap_summary.png")
            shap_blob.upload_from_filename(shap_path)
            logger.info("Uploaded SHAP plot to registry")

        dated_uri = f"gs://{self.bucket_name}/{dated_prefix}/model.lgb"
        logger.info(f"Pushed model to registry: {dated_uri}")

        if update_latest:
            self._update_latest(version, model_blob, meta_blob, shap_path)

        return dated_uri

    def _update_latest(
        self,
        version: str,
        model_blob: storage.Blob,
        meta_blob: storage.Blob,
        shap_path: str | None = None,
    ) -> None:
        """Update the latest/ pointer to a specific version."""
        latest_prefix = self._model_prefix("latest")

        # Copy model to latest/
        self.bucket.copy_blob(model_blob, self.bucket, f"{latest_prefix}/model.lgb")
        self.bucket.copy_blob(meta_blob, self.bucket, f"{latest_prefix}/metadata.json")

        # Copy SHAP if exists
        dated_prefix = self._model_prefix(version)
        shap_blob = self.bucket.blob(f"{dated_prefix}/shap_summary.png")
        if shap_blob.exists():
            self.bucket.copy_blob(shap_blob, self.bucket, f"{latest_prefix}/shap_summary.png")

        logger.info(f"Updated latest/ pointer to version {version}")

    def pull_latest(self, local_dir: str | None = None) -> tuple[str, dict[str, Any]]:
        """
        Pull latest model to a local directory.

        Args:
            local_dir: Directory to download to (creates temp if not provided)

        Returns:
            (local model path, metadata dict)
        """
        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        latest_prefix = self._model_prefix("latest")

        model_path = str(Path(local_dir) / "model.lgb")
        self.bucket.blob(f"{latest_prefix}/model.lgb").download_to_filename(model_path)

        meta_content = self.bucket.blob(f"{latest_prefix}/metadata.json").download_as_text()
        metadata = json.loads(meta_content)

        logger.info(f"Pulled latest model: version={metadata.get('version')}")
        return model_path, metadata

    def pull_version(
        self, version: str, local_dir: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Pull a specific model version.

        Args:
            version: Version string (e.g. "20260316")
            local_dir: Directory to download to

        Returns:
            (local model path, metadata dict)
        """
        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        prefix = self._model_prefix(version)

        model_path = str(Path(local_dir) / "model.lgb")
        self.bucket.blob(f"{prefix}/model.lgb").download_to_filename(model_path)

        meta_content = self.bucket.blob(f"{prefix}/metadata.json").download_as_text()
        metadata = json.loads(meta_content)

        logger.info(f"Pulled model version: {version}")
        return model_path, metadata

    def get_latest_metadata(self) -> dict[str, Any]:
        """Get metadata of the currently deployed model without downloading the model file."""
        latest_prefix = self._model_prefix("latest")
        content = self.bucket.blob(f"{latest_prefix}/metadata.json").download_as_text()
        return json.loads(content)

    def list_versions(self) -> list[str]:
        """List all available model versions."""
        prefix = f"registry/{self.model_name}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        versions = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 3:
                version = parts[2]
                if version != "latest":
                    versions.add(version)

        return sorted(versions)

    def rollback_to_version(self, version: str) -> dict[str, Any]:
        """
        Rollback latest/ pointer to a specific version.

        Args:
            version: Version to rollback to

        Returns:
            Metadata of the rolled-back version
        """
        prefix = self._model_prefix(version)

        # Verify version exists
        model_blob = self.bucket.blob(f"{prefix}/model.lgb")
        if not model_blob.exists():
            raise ValueError(f"Version {version} not found in registry")

        meta_blob = self.bucket.blob(f"{prefix}/metadata.json")

        # Update latest
        self._update_latest(version, model_blob, meta_blob)

        logger.info(f"Rolled back to version {version}")
        return self.get_latest_metadata()
