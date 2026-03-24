"""Tests for shared/config_loader.py."""

from __future__ import annotations

import pytest

import shared.config_loader as config_loader


@pytest.fixture(autouse=True)
def clear_config_cache() -> None:
    """Isolate tests that load YAML or env overrides."""
    config_loader.load_training_config.cache_clear()
    yield
    config_loader.load_training_config.cache_clear()


def test_load_training_config_crime_navigate() -> None:
    cfg = config_loader.load_training_config("crime_navigate_train")
    assert "model" in cfg
    assert cfg["model"].get("name") is not None


def test_load_training_config_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        config_loader.load_training_config("nonexistent_config_xyz")


def test_apply_env_overrides_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_VALIDATION_RMSE_GATE", "99.5")
    cfg = config_loader.load_training_config("crime_navigate_train")
    assert cfg["validation"]["rmse_gate"] == 99.5


def test_apply_env_overrides_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_TRAINING_RANDOM_SEED", "123")
    cfg = config_loader.load_training_config("crime_navigate_train")
    assert cfg["training"]["random_seed"] == 123


def test_apply_env_overrides_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_SCORING_SCALE_WITHIN_BUCKET", "false")
    cfg = config_loader.load_training_config("crime_navigate_train")
    assert cfg["scoring"]["scale_within_bucket"] is False


def test_apply_env_overrides_string_else_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DATA_BUCKET", "override-bucket-name")
    cfg = config_loader.load_training_config("crime_navigate_train")
    assert cfg["data"]["bucket"] == "override-bucket-name"


def test_reload_config() -> None:
    c1 = config_loader.load_training_config("crime_navigate_train")
    config_loader.reload_config("crime_navigate_train")
    c2 = config_loader.load_training_config("crime_navigate_train")
    assert c1["model"]["name"] == c2["model"]["name"]


def test_get_bucket_name_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    assert config_loader.get_bucket_name({"data": {"bucket": "my-bucket"}}) == "my-bucket"


def test_get_bucket_name_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCS_BUCKET", "env-bucket")
    assert config_loader.get_bucket_name({"data": {"bucket": "my-bucket"}}) == "env-bucket"


def test_get_bucket_name_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    assert "boston-pulse" in config_loader.get_bucket_name({})
