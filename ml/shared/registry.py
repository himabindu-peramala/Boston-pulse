"""
Boston Pulse ML - Model Registry.

Unified interface for model artifact storage with two backends:
  - Artifact Registry (AR): Primary storage for deployable model artifacts
  - GCS: Backup storage and SHAP plots (MLflow artifacts)

Repository structure with namespaced packages:
  ml-models/                              (repository)
    ├── navigate/crime-risk               (package - current model)
    │   ├── 20260316                      (version)
    │   ├── 20260322                      (version)
    │   └── production → 20260322         (tag)
    ├── navigate/transit-risk             (package - future)
    └── chatbot/intent-model              (package - future)

Versioning pattern:
  - Dated versions (e.g., "20260322") are immutable snapshots
  - Stage tags: "staging" → "production" promotion flow
  - "latest" always points to most recent successful training

Production deployment flow:
  1. Train model → push to AR with "staging" tag
  2. Gates pass → promote to "production" tag
  3. Production service pulls "production" tag at startup

The GCS backup ensures:
  - Fallback if AR is unavailable
  - SHAP plots accessible via MLflow UI
  - Compatibility with existing infrastructure
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from google.cloud import storage

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry with Artifact Registry primary and GCS backup.

    Provides versioned storage with stage-based promotion for production deployment.
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize registry.

        Args:
            cfg: Training config with registry section containing:
                - project: GCP project ID
                - location: AR location (e.g., us-east1)
                - repository: AR repository name
                - package: Namespaced package name (e.g., "navigate/crime-risk")
                - artifact_bucket: GCS bucket for backup
        """
        registry_cfg = cfg.get("registry", {})
        self.project = registry_cfg.get("project", "bostonpulse")
        self.location = registry_cfg.get("location", "us-east1")
        self.repository = registry_cfg.get("repository", "ml-models")

        # Package name uses namespace convention: {domain}/{model-purpose}
        self.package = registry_cfg.get(
            "package", cfg.get("model", {}).get("name", "navigate/crime-risk")
        )
        # For display and GCS paths, use the model-purpose part
        self.model_name = self.package.split("/")[-1] if "/" in self.package else self.package

        # ARTIFACT_BUCKET (set by Terraform-deployed Cloud Run job/service)
        # wins over the YAML config so a fresh project can point at its own
        # project-scoped bucket without editing configs.
        self.gcs_bucket_name = os.environ.get(
            "ARTIFACT_BUCKET",
            registry_cfg.get("artifact_bucket", "boston-pulse-mlflow-artifacts"),
        )
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)

        self.use_artifact_registry = registry_cfg.get("use_artifact_registry", True)

        self._ar_client = None
        self._cfg = cfg

    @property
    def ar_client(self):
        """Lazy-load AR client to avoid import errors if not needed."""
        if self._ar_client is None and self.use_artifact_registry:
            try:
                from shared.artifact_registry import ArtifactRegistryClient

                self._ar_client = ArtifactRegistryClient(
                    {
                        "registry": {
                            "project": self.project,
                            "location": self.location,
                            "repository": self.repository,
                            "package": self.package,
                        },
                        "model": {"name": self.model_name},
                    }
                )
            except Exception as e:
                logger.warning(f"AR client init failed, using GCS only: {e}")
                self.use_artifact_registry = False
        return self._ar_client

    def _gcs_prefix(self, version: str) -> str:
        """Get GCS prefix for a model version."""
        return f"registry/{self.model_name}/{version}"

    def push(
        self,
        model_path: str,
        version: str,
        metadata: dict[str, Any],
        update_latest: bool = True,
        shap_path: str | None = None,
        stage: str = "staging",
    ) -> str:
        """
        Push model to registry (AR primary, GCS backup).

        Args:
            model_path: Local path to model.lgb file
            version: Version string (e.g., "20260322")
            metadata: Model metadata dict
            update_latest: Whether to update the latest pointer
            shap_path: Optional path to SHAP summary plot
            stage: Initial stage ("staging" or "production")

        Returns:
            Primary artifact URI (AR if enabled, else GCS)
        """
        ar_uri = None
        gcs_uri = None

        if self.use_artifact_registry and self.ar_client:
            try:
                result = self.ar_client.push(
                    model_path=model_path,
                    version=version,
                    metadata=metadata,
                    stage=stage,
                    shap_path=shap_path,
                )
                ar_uri = result["ar_uri"]
                logger.info(f"Pushed to Artifact Registry: {ar_uri}")
            except Exception as e:
                logger.warning(f"AR push failed, falling back to GCS: {e}")

        gcs_uri = self._push_to_gcs(
            model_path=model_path,
            version=version,
            metadata=metadata,
            update_latest=update_latest,
            shap_path=shap_path,
        )

        return ar_uri if ar_uri else gcs_uri

    def _push_to_gcs(
        self,
        model_path: str,
        version: str,
        metadata: dict[str, Any],
        update_latest: bool = True,
        shap_path: str | None = None,
    ) -> str:
        """Push model to GCS (backup storage)."""
        prefix = self._gcs_prefix(version)

        model_blob = self.gcs_bucket.blob(f"{prefix}/model.lgb")
        model_blob.upload_from_filename(model_path)
        logger.info(f"Uploaded model to GCS: gs://{self.gcs_bucket_name}/{prefix}/model.lgb")

        full_metadata = {
            **metadata,
            "version": version,
            "model_name": self.model_name,
        }
        meta_blob = self.gcs_bucket.blob(f"{prefix}/metadata.json")
        meta_blob.upload_from_string(
            json.dumps(full_metadata, indent=2, default=str),
            content_type="application/json",
        )

        if shap_path and Path(shap_path).exists():
            shap_blob = self.gcs_bucket.blob(f"{prefix}/shap_summary.png")
            shap_blob.upload_from_filename(shap_path)
            logger.info("Uploaded SHAP plot to GCS")

        gcs_uri = f"gs://{self.gcs_bucket_name}/{prefix}/model.lgb"

        if update_latest:
            self._update_gcs_latest(version, model_blob, meta_blob, shap_path)

        return gcs_uri

    def _update_gcs_latest(
        self,
        version: str,
        model_blob: storage.Blob,
        meta_blob: storage.Blob,
        shap_path: str | None = None,
    ) -> None:
        """Update the GCS latest/ pointer."""
        latest_prefix = self._gcs_prefix("latest")

        self.gcs_bucket.copy_blob(model_blob, self.gcs_bucket, f"{latest_prefix}/model.lgb")
        self.gcs_bucket.copy_blob(meta_blob, self.gcs_bucket, f"{latest_prefix}/metadata.json")

        dated_prefix = self._gcs_prefix(version)
        shap_blob = self.gcs_bucket.blob(f"{dated_prefix}/shap_summary.png")
        if shap_blob.exists():
            self.gcs_bucket.copy_blob(
                shap_blob, self.gcs_bucket, f"{latest_prefix}/shap_summary.png"
            )

        logger.info(f"Updated GCS latest/ pointer to version {version}")

    def promote_to_production(self, version: str) -> dict[str, Any]:
        """
        Promote a staged model to production.

        Args:
            version: Version to promote

        Returns:
            Dict with promotion info
        """
        result = {"version": version, "stage": "production"}

        if self.use_artifact_registry and self.ar_client:
            try:
                ar_result = self.ar_client.promote_to_production(version)
                result["ar_path"] = ar_result.get("ar_path")
                logger.info(f"Promoted {version} to production in AR")
            except Exception as e:
                logger.warning(f"AR promotion failed: {e}")

        self._update_gcs_production(version)

        return result

    def _update_gcs_production(self, version: str) -> None:
        """Update GCS production/ pointer."""
        src_prefix = self._gcs_prefix(version)
        prod_prefix = self._gcs_prefix("production")

        for suffix in ["model.lgb", "metadata.json", "shap_summary.png"]:
            src_blob = self.gcs_bucket.blob(f"{src_prefix}/{suffix}")
            if src_blob.exists():
                self.gcs_bucket.copy_blob(src_blob, self.gcs_bucket, f"{prod_prefix}/{suffix}")

        logger.info(f"Updated GCS production/ pointer to version {version}")

    def pull_latest(self, local_dir: str | None = None) -> tuple[str, dict[str, Any]]:
        """
        Pull latest model.

        Tries AR first (production tag), falls back to GCS.

        Args:
            local_dir: Directory to download to

        Returns:
            (local model path, metadata dict)
        """
        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        if self.use_artifact_registry and self.ar_client:
            try:
                return self.ar_client.pull("production")
            except Exception as e:
                logger.warning(f"AR pull failed, using GCS: {e}")

        return self._pull_from_gcs("latest", local_dir)

    def pull_version(
        self, version: str, local_dir: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Pull a specific model version.

        Args:
            version: Version string
            local_dir: Directory to download to

        Returns:
            (local model path, metadata dict)
        """
        if local_dir is None:
            local_dir = tempfile.mkdtemp()

        if self.use_artifact_registry and self.ar_client:
            try:
                return self.ar_client.pull(version, local_dir)
            except Exception as e:
                logger.warning(f"AR pull failed for {version}, using GCS: {e}")

        return self._pull_from_gcs(version, local_dir)

    def _pull_from_gcs(self, version: str, local_dir: str) -> tuple[str, dict[str, Any]]:
        """Pull model from GCS."""
        prefix = self._gcs_prefix(version)

        model_path = str(Path(local_dir) / "model.lgb")
        self.gcs_bucket.blob(f"{prefix}/model.lgb").download_to_filename(model_path)

        meta_content = self.gcs_bucket.blob(f"{prefix}/metadata.json").download_as_text()
        metadata = json.loads(meta_content)

        logger.info(f"Pulled model from GCS: version={metadata.get('version')}")
        return model_path, metadata

    def get_latest_metadata(self) -> dict[str, Any]:
        """Get metadata of the currently deployed model."""
        if self.use_artifact_registry and self.ar_client:
            try:
                _, metadata = self.ar_client.pull("production")
                return metadata
            except Exception as e:
                logger.warning(f"AR metadata fetch failed: {e}")

        prefix = self._gcs_prefix("latest")
        content = self.gcs_bucket.blob(f"{prefix}/metadata.json").download_as_text()
        return json.loads(content)

    def list_versions(self) -> list[str]:
        """List all available model versions."""
        versions = set()

        if self.use_artifact_registry and self.ar_client:
            try:
                ar_versions = self.ar_client.list_versions()
                versions.update(v["version"] for v in ar_versions)
            except Exception as e:
                logger.warning(f"AR list failed: {e}")

        prefix = f"registry/{self.model_name}/"
        blobs = self.gcs_client.list_blobs(self.gcs_bucket_name, prefix=prefix)

        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 3:
                version = parts[2]
                if version not in ("latest", "production"):
                    versions.add(version)

        return sorted(versions)

    def rollback_to_version(self, version: str) -> dict[str, Any]:
        """
        Rollback to a specific version.

        Args:
            version: Version to rollback to

        Returns:
            Metadata of the rolled-back version
        """
        prefix = self._gcs_prefix(version)
        model_blob = self.gcs_bucket.blob(f"{prefix}/model.lgb")
        if not model_blob.exists():
            raise ValueError(f"Version {version} not found in registry")

        if self.use_artifact_registry and self.ar_client:
            try:
                self.ar_client.rollback(version)
            except Exception as e:
                logger.warning(f"AR rollback failed: {e}")

        meta_blob = self.gcs_bucket.blob(f"{prefix}/metadata.json")
        self._update_gcs_latest(version, model_blob, meta_blob)
        self._update_gcs_production(version)

        logger.info(f"Rolled back to version {version}")
        return self.get_latest_metadata()

    def get_production_version(self) -> str | None:
        """Get the version currently tagged as production."""
        if self.use_artifact_registry and self.ar_client:
            try:
                return self.ar_client.get_version_by_tag("production")
            except Exception:
                pass

        try:
            meta = self.get_latest_metadata()
            return meta.get("version")
        except Exception:
            return None

    def compare_to_production(
        self,
        candidate_val_rmse: float,
        tolerance: float = 0.02,
    ) -> dict[str, Any]:
        """
        Compare a candidate model's val_rmse against the current production model.

        This is Gate 3 — prevents promoting a model that is worse than production.
        Critical for daily training to avoid replacing a good model with a bad one.

        Args:
            candidate_val_rmse: val_rmse of the candidate model just trained
            tolerance: fractional tolerance — candidate can be up to this much worse
                       than production and still promote (prevents churn on noise).
                       0.02 means candidate up to 2% worse than prod is still accepted.

        Returns:
            {
                'should_promote': bool,
                'reason': str,
                'production_version': str | None,
                'production_rmse': float | None,
                'candidate_rmse': float,
                'delta_pct': float | None,      # positive = candidate worse
            }
        """
        result: dict[str, Any] = {
            "should_promote": False,
            "reason": "",
            "production_version": None,
            "production_rmse": None,
            "candidate_rmse": candidate_val_rmse,
            "delta_pct": None,
        }

        try:
            prod_meta = self.get_latest_metadata()
        except Exception as e:
            # Cold start — no production model exists
            result["should_promote"] = True
            result["reason"] = f"cold start: no production model found ({e})"
            return result

        prod_rmse = prod_meta.get("val_rmse")
        prod_version = prod_meta.get("version")
        result["production_version"] = prod_version
        result["production_rmse"] = prod_rmse

        if prod_rmse is None:
            result["should_promote"] = True
            result["reason"] = "production model has no val_rmse in metadata"
            return result

        delta_pct = ((candidate_val_rmse - prod_rmse) / prod_rmse) * 100
        result["delta_pct"] = delta_pct

        if candidate_val_rmse <= prod_rmse * (1 + tolerance):
            result["should_promote"] = True
            result["reason"] = (
                f"candidate rmse {candidate_val_rmse:.4f} within {tolerance:.0%} "
                f"of prod {prod_rmse:.4f} (delta={delta_pct:+.2f}%)"
            )
        else:
            result["should_promote"] = False
            result["reason"] = (
                f"candidate rmse {candidate_val_rmse:.4f} worse than "
                f"prod {prod_rmse:.4f} by {delta_pct:+.2f}% — keeping prod"
            )

        return result

    def get_ar_uri(self, version: str) -> str:
        """Get the Artifact Registry URI for a version."""
        return f"https://{self.location}-generic.pkg.dev/{self.project}/{self.repository}/{self.package}:{version}"

    def _ar_get_version_for_tag(self, tag: str) -> str | None:
        """Get the version a tag points to in AR."""
        if self.use_artifact_registry and self.ar_client:
            return self.ar_client.get_version_by_tag(tag)
        return None
