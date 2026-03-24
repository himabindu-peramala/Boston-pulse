"""
Boston Pulse ML - Configuration Loader.

Loads training configuration from YAML files in ml/configs/.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent / "configs"


@lru_cache(maxsize=8)
def load_training_config(config_name: str) -> dict[str, Any]:
    """
    Load a training config YAML from ml/configs/.

    Args:
        config_name: Config name without .yaml extension
                     e.g. "crime_navigate_train"

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist

    Usage:
        cfg = load_training_config("crime_navigate_train")
        rmse_gate = cfg["validation"]["rmse_gate"]
    """
    config_path = _get_config_dir() / f"{config_name}.yaml"

    if not config_path.exists():
        available = list(_get_config_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"Config not found: {config_path}\n" f"Available configs: {[p.stem for p in available]}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply environment overrides
    config = _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to config.

    Environment variables follow pattern: ML_<SECTION>_<KEY>
    e.g. ML_VALIDATION_RMSE_GATE=10.0 overrides config["validation"]["rmse_gate"]
    """
    env_prefix = "ML_"

    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue

        parts = key[len(env_prefix) :].lower().split("_")
        if len(parts) < 2:
            continue

        section = parts[0]
        subkey = "_".join(parts[1:])

        if section in config and isinstance(config[section], dict) and subkey in config[section]:
            original = config[section][subkey]
            if isinstance(original, bool):
                config[section][subkey] = value.lower() in ("true", "1", "yes")
            elif isinstance(original, int):
                config[section][subkey] = int(value)
            elif isinstance(original, float):
                config[section][subkey] = float(value)
            else:
                config[section][subkey] = value

    return config


def reload_config(config_name: str) -> dict[str, Any]:
    """Reload config, clearing the cache."""
    load_training_config.cache_clear()
    return load_training_config(config_name)


def get_bucket_name(config: dict[str, Any]) -> str:
    """Get GCS bucket name from config or environment."""
    return os.getenv(
        "GCS_BUCKET", config.get("data", {}).get("bucket", "boston-pulse-data-pipeline")
    )
