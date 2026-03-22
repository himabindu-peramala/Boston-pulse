"""Tests for shared/gcs_loader.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from shared.gcs_loader import GCSLoader


@pytest.fixture
def mock_storage_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    bucket = MagicMock()
    client.bucket.return_value = bucket
    monkeypatch.setattr("shared.gcs_loader.storage.Client", lambda *a, **kw: client)
    return client


def test_init_uses_gcp_project(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-x")
    loader = GCSLoader("my-bucket")
    assert loader.bucket_name == "my-bucket"
    mock_storage_client.bucket.assert_called_with("my-bucket")


def test_init_emulator_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulator path passes api_endpoint in client_options (assert on Client class mock)."""
    client = MagicMock()
    client.bucket.return_value = MagicMock()
    mock_client_cls = MagicMock(return_value=client)
    monkeypatch.setattr("shared.gcs_loader.storage.Client", mock_client_cls)
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://localhost:4443")
    GCSLoader("b")
    mock_client_cls.assert_called_once()
    kwargs = mock_client_cls.call_args.kwargs
    assert kwargs["client_options"]["api_endpoint"] == "http://localhost:4443"


def test_list_partitions_filters(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    b1 = MagicMock()
    b1.name = "features/x/dt=2024-01-01/data.parquet"
    b2 = MagicMock()
    b2.name = "features/x/dt=2024-01-03/data.parquet"
    mock_storage_client.list_blobs.return_value = [b1, b2]
    loader = GCSLoader("buck")
    dates = loader.list_partitions("features/x", after="2024-01-02")
    assert dates == ["2024-01-03"]


def test_read_partition_calls_parquet(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch, mocker: pytest.Mock
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    df = pd.DataFrame({"a": [1]})
    mread = mocker.patch("shared.gcs_loader.pd.read_parquet", return_value=df)
    loader = GCSLoader("buck")
    out = loader.read_partition("pre", "2024-01-01")
    assert len(out) == 1
    mread.assert_called_once()


def test_read_all_partitions_no_partitions_raises(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    mock_storage_client.list_blobs.return_value = []
    loader = GCSLoader("buck")
    with pytest.raises(FileNotFoundError, match="No partitions found"):
        loader.read_all_partitions("pre", after="2099-01-01")


def test_read_all_partitions_concat(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch, mocker: pytest.Mock
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    blob = MagicMock()
    blob.name = "pre/dt=2024-01-01/data.parquet"
    mock_storage_client.list_blobs.return_value = [blob]
    mread = mocker.patch(
        "shared.gcs_loader.pd.read_parquet",
        return_value=pd.DataFrame({"x": [1]}),
    )
    loader = GCSLoader("buck")
    out = loader.read_all_partitions("pre")
    assert len(out) == 1
    assert "date" in out.columns
    mread.assert_called()


def test_read_all_partitions_skips_bad_partition(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch, mocker: pytest.Mock
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    b1 = MagicMock()
    b1.name = "pre/dt=2024-01-01/data.parquet"
    mock_storage_client.list_blobs.return_value = [b1]

    def _read(*_a: object, **_k: object) -> pd.DataFrame:
        raise OSError("bad")

    mocker.patch("shared.gcs_loader.pd.read_parquet", side_effect=_read)
    loader = GCSLoader("buck")
    with pytest.raises(RuntimeError, match="All partitions failed"):
        loader.read_all_partitions("pre")


def test_write_parquet(mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    bucket = mock_storage_client.bucket.return_value
    blob = MagicMock()
    bucket.blob.return_value = blob
    loader = GCSLoader("buck")
    path = loader.write_parquet(pd.DataFrame({"a": [1]}), "ml/out", "2024-01-01")
    assert path.startswith("gs://buck/")
    blob.upload_from_file.assert_called_once()


def test_write_json_read_json_file_exists(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    bucket = mock_storage_client.bucket.return_value
    blob = MagicMock()
    bucket.blob.return_value = blob
    blob.download_as_text.return_value = '{"k": 1}'
    blob.exists.return_value = True
    loader = GCSLoader("buck")
    w = loader.write_json({"a": 1}, "p", "2024-01-01", "f.json")
    assert w.startswith("gs://")
    assert loader.read_json("p", "2024-01-01", "f.json") == {"k": 1}
    assert loader.file_exists("p", "2024-01-01", "f.json") is True


def test_upload_download_file(
    mock_storage_client: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    bucket = mock_storage_client.bucket.return_value
    blob = MagicMock()
    bucket.blob.return_value = blob
    loader = GCSLoader("buck")
    p = tmp_path / "local.txt"
    p.write_text("x")
    uri = loader.upload_file(str(p), "remote/path.txt")
    assert uri.startswith("gs://buck/remote/")
    out = tmp_path / "out.txt"
    loader.download_file("remote/path.txt", str(out))
    blob.download_to_filename.assert_called_once()
