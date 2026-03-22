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


def test_push_uploads_model_and_metadata(
    mock_bucket: MagicMock, registry_cfg: dict, tmp_path
) -> None:
    model_file = tmp_path / "model.lgb"
    model_file.write_bytes(b"x")
    reg = ModelRegistry(registry_cfg)
    meta = {"rmse": 0.5}
    uri = reg.push(str(model_file), "20260101", meta, update_latest=False)
    assert "model.lgb" in uri
    mock_bucket.blob.assert_called()
    mock_bucket.blob.return_value.upload_from_filename.assert_called()


def test_push_with_shap(mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
    model_file = tmp_path / "model.lgb"
    model_file.write_bytes(b"x")
    shap = tmp_path / "shap.png"
    shap.write_bytes(b"png")
    reg = ModelRegistry(registry_cfg)
    reg.push(str(model_file), "20260101", {}, update_latest=False, shap_path=str(shap))


def test_push_updates_latest(mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
    model_file = tmp_path / "model.lgb"
    model_file.write_bytes(b"x")
    blob = MagicMock()
    mock_bucket.blob.return_value = blob
    mock_bucket.copy_blob = MagicMock()
    reg = ModelRegistry(registry_cfg)
    reg.push(str(model_file), "20260101", {"k": 1}, update_latest=True)


def test_pull_latest(mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
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


def test_get_latest_metadata(mock_bucket: MagicMock, registry_cfg: dict) -> None:
    mb = MagicMock()
    mb.download_as_text.return_value = json.dumps({"version": "v9"})
    mock_bucket.blob.return_value = mb
    reg = ModelRegistry(registry_cfg)
    assert reg.get_latest_metadata()["version"] == "v9"


def test_list_versions(registry_cfg: dict, mock_bucket: MagicMock) -> None:
    b1 = MagicMock()
    b1.name = "registry/crime-navigate-model/20240101/model.lgb"
    b2 = MagicMock()
    b2.name = "registry/crime-navigate-model/latest/model.lgb"
    reg = ModelRegistry(registry_cfg)
    reg.client.list_blobs = MagicMock(return_value=[b1, b2])
    versions = reg.list_versions()
    assert "20240101" in versions
    assert "latest" not in versions


def test_rollback_to_version_calls_update_and_returns_metadata(
    mock_bucket: MagicMock, registry_cfg: dict
) -> None:
    reg = ModelRegistry(registry_cfg)
    model_blob = MagicMock()
    model_blob.exists.return_value = True
    meta_blob = MagicMock()
    mock_bucket.blob.side_effect = lambda p: model_blob if p.endswith("model.lgb") else meta_blob
    reg._update_latest = MagicMock()  # type: ignore[method-assign]
    reg.get_latest_metadata = MagicMock(return_value={"version": "20240101"})  # type: ignore[method-assign]
    out = reg.rollback_to_version("20240101")
    assert out["version"] == "20240101"
    reg._update_latest.assert_called_once()


def test_rollback_missing_version_raises(mock_bucket: MagicMock, registry_cfg: dict) -> None:
    mb = MagicMock()
    mb.exists.return_value = False
    mock_bucket.blob.return_value = mb
    reg = ModelRegistry(registry_cfg)
    with pytest.raises(ValueError, match="not found"):
        reg.rollback_to_version("nope")


def test_pull_version(mock_bucket: MagicMock, registry_cfg: dict, tmp_path) -> None:
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
