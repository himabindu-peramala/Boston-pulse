"""Tests for shared/registry.py ModelRegistry."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shared.registry import ModelRegistry


@pytest.fixture
def registry_cfg(sample_cfg: dict) -> dict:
    return sample_cfg


@pytest.fixture
def mock_bucket(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    bucket = MagicMock()
    client = MagicMock()
    client.bucket.return_value = bucket
    monkeypatch.setattr("shared.registry.storage.Client", lambda: client)
    return bucket


@pytest.fixture
def registry_no_ar(registry_cfg: dict, mock_bucket: MagicMock) -> ModelRegistry:
    """Registry with AR disabled (GCS only)."""
    registry_cfg["registry"]["use_artifact_registry"] = False
    return ModelRegistry(registry_cfg)


class TestModelRegistryGCSOnly:
    """Tests for ModelRegistry with GCS backend only."""

    def test_push_uploads_model_and_metadata(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"x")
        reg = ModelRegistry(registry_cfg)
        meta = {"rmse": 0.5}
        uri = reg.push(str(model_file), "20260101", meta, update_latest=False)
        assert "model.lgb" in uri
        mock_bucket.blob.assert_called()
        mock_bucket.blob.return_value.upload_from_filename.assert_called()

    def test_push_with_shap(self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"x")
        shap = tmp_path / "shap.png"
        shap.write_bytes(b"png")
        reg = ModelRegistry(registry_cfg)
        reg.push(str(model_file), "20260101", {}, update_latest=False, shap_path=str(shap))

    def test_push_updates_latest(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"x")
        blob = MagicMock()
        mock_bucket.blob.return_value = blob
        mock_bucket.copy_blob = MagicMock()
        reg = ModelRegistry(registry_cfg)
        reg.push(str(model_file), "20260101", {"k": 1}, update_latest=True)

    def test_pull_latest(self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        meta_blob = MagicMock()
        meta_blob.download_as_text.return_value = json.dumps({"version": "v1"})
        model_blob = MagicMock()

        def blob_side_effect(name: str) -> MagicMock:
            b = MagicMock()
            if name.endswith("metadata.json"):
                b.download_as_text = meta_blob.download_as_text
            else:
                b.download_to_filename = model_blob.download_to_filename
            return b

        mock_bucket.blob.side_effect = blob_side_effect
        reg = ModelRegistry(registry_cfg)
        path, meta = reg.pull_latest(str(tmp_path))
        assert meta.get("version") == "v1"
        assert path.endswith("model.lgb")

    def test_get_latest_metadata(self, mock_bucket: MagicMock, registry_cfg: dict) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        mb = MagicMock()
        mb.download_as_text.return_value = json.dumps({"version": "v9"})
        mock_bucket.blob.return_value = mb
        reg = ModelRegistry(registry_cfg)
        assert reg.get_latest_metadata()["version"] == "v9"

    def test_list_versions(self, registry_cfg: dict, mock_bucket: MagicMock) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        b1 = MagicMock()
        b1.name = "registry/crime-navigate-model/20240101/model.lgb"
        b2 = MagicMock()
        b2.name = "registry/crime-navigate-model/latest/model.lgb"
        reg = ModelRegistry(registry_cfg)
        reg.gcs_client.list_blobs = MagicMock(return_value=[b1, b2])
        versions = reg.list_versions()
        assert "20240101" in versions
        assert "latest" not in versions

    def test_rollback_to_version_calls_update_and_returns_metadata(
        self, mock_bucket: MagicMock, registry_cfg: dict
    ) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        reg = ModelRegistry(registry_cfg)
        model_blob = MagicMock()
        model_blob.exists.return_value = True
        meta_blob = MagicMock()
        mock_bucket.blob.side_effect = lambda p: (
            model_blob if p.endswith("model.lgb") else meta_blob
        )
        reg._update_gcs_latest = MagicMock()
        reg._update_gcs_production = MagicMock()
        reg.get_latest_metadata = MagicMock(return_value={"version": "20240101"})
        out = reg.rollback_to_version("20240101")
        assert out["version"] == "20240101"
        reg._update_gcs_latest.assert_called_once()

    def test_rollback_missing_version_raises(
        self, mock_bucket: MagicMock, registry_cfg: dict
    ) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        mb = MagicMock()
        mb.exists.return_value = False
        mock_bucket.blob.return_value = mb
        reg = ModelRegistry(registry_cfg)
        with pytest.raises(ValueError, match="not found"):
            reg.rollback_to_version("nope")

    def test_pull_version(self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
        registry_cfg["registry"]["use_artifact_registry"] = False
        meta = MagicMock()
        meta.download_as_text.return_value = json.dumps({"version": "x"})

        def blob(name: str) -> MagicMock:
            b = MagicMock()
            if name.endswith("metadata.json"):
                b.download_as_text = meta.download_as_text
            else:
                b.download_to_filename = MagicMock()
            return b

        mock_bucket.blob.side_effect = blob
        reg = ModelRegistry(registry_cfg)
        p, m = reg.pull_version("20240101", str(tmp_path))
        assert p.endswith("model.lgb")
        assert m["version"] == "x"


class TestModelRegistryWithAR:
    """Tests for ModelRegistry with AR enabled (dual backend)."""

    def test_push_calls_ar_and_gcs(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        """Push uploads to both AR and GCS when AR is enabled."""
        registry_cfg["registry"]["use_artifact_registry"] = True
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model-content")

        mock_ar_client = MagicMock()
        mock_ar_client.push.return_value = {
            "ar_uri": "https://us-east1-generic.pkg.dev/proj/repo/model:20260322",
            "version": "20260322",
            "stage": "staging",
            "sha256": "abc123",
        }

        reg = ModelRegistry(registry_cfg)
        reg._ar_client = mock_ar_client
        reg.use_artifact_registry = True

        uri = reg.push(
            str(model_file), "20260322", {"rmse": 0.5}, update_latest=True, stage="staging"
        )

        mock_ar_client.push.assert_called_once()
        assert "pkg.dev" in uri
        mock_bucket.blob.assert_called()

    def test_push_falls_back_to_gcs_on_ar_failure(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        """Push falls back to GCS if AR fails."""
        registry_cfg["registry"]["use_artifact_registry"] = True
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")

        mock_ar_client = MagicMock()
        mock_ar_client.push.side_effect = Exception("AR unavailable")

        reg = ModelRegistry(registry_cfg)
        reg._ar_client = mock_ar_client
        reg.use_artifact_registry = True

        uri = reg.push(str(model_file), "20260322", {}, update_latest=False)

        assert "gs://" in uri
        mock_bucket.blob.assert_called()

    def test_promote_to_production_updates_both(
        self, mock_bucket: MagicMock, registry_cfg: dict
    ) -> None:
        """Promotion updates both AR and GCS."""
        registry_cfg["registry"]["use_artifact_registry"] = True

        mock_ar_client = MagicMock()
        mock_ar_client.promote_to_production.return_value = {
            "version": "20260322",
            "stage": "production",
            "ar_path": "proj/repo/model:production",
        }

        src_blob = MagicMock()
        src_blob.exists.return_value = True
        mock_bucket.blob.return_value = src_blob
        mock_bucket.copy_blob = MagicMock()

        reg = ModelRegistry(registry_cfg)
        reg._ar_client = mock_ar_client
        reg.use_artifact_registry = True

        result = reg.promote_to_production("20260322")

        mock_ar_client.promote_to_production.assert_called_once_with("20260322")
        assert result["stage"] == "production"
        assert "ar_path" in result

    def test_pull_latest_tries_ar_first(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        """Pull tries AR first, falls back to GCS."""
        registry_cfg["registry"]["use_artifact_registry"] = True

        mock_ar_client = MagicMock()
        mock_ar_client.pull.return_value = (
            str(tmp_path / "model.lgb"),
            {"version": "20260322", "stage": "production"},
        )

        reg = ModelRegistry(registry_cfg)
        reg._ar_client = mock_ar_client
        reg.use_artifact_registry = True

        path, meta = reg.pull_latest()

        mock_ar_client.pull.assert_called_once_with("production")
        assert meta["version"] == "20260322"

    def test_get_production_version_from_ar(
        self, mock_bucket: MagicMock, registry_cfg: dict
    ) -> None:
        """Get production version queries AR tag."""
        registry_cfg["registry"]["use_artifact_registry"] = True

        mock_ar_client = MagicMock()
        mock_ar_client.get_version_by_tag.return_value = "20260322"

        reg = ModelRegistry(registry_cfg)
        reg._ar_client = mock_ar_client
        reg.use_artifact_registry = True

        version = reg.get_production_version()

        mock_ar_client.get_version_by_tag.assert_called_once_with("production")
        assert version == "20260322"


class TestModelRegistryStageFlow:
    """Tests for the staging → production promotion flow."""

    def test_full_stage_promotion_flow(
        self, mock_bucket: MagicMock, registry_cfg: dict, tmp_path
    ) -> None:
        """Test complete flow: push staging → promote to production."""
        registry_cfg["registry"]["use_artifact_registry"] = False
        model_file = tmp_path / "model.lgb"
        model_file.write_bytes(b"model")

        blob = MagicMock()
        blob.exists.return_value = True
        mock_bucket.blob.return_value = blob
        mock_bucket.copy_blob = MagicMock()

        reg = ModelRegistry(registry_cfg)

        uri = reg.push(
            str(model_file),
            "20260322",
            {"rmse": 0.5},
            update_latest=True,
            stage="staging",
        )
        assert "20260322" in uri

        result = reg.promote_to_production("20260322")
        assert result["stage"] == "production"
        assert result["version"] == "20260322"
