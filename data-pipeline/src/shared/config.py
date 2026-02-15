"""
Boston Pulse - Configuration Loader

Pydantic-based configuration management with:
- Environment-based configuration (dev/prod)
- YAML file loading with inheritance
- Environment variable overrides for secrets
- Type validation via Pydantic

Usage:
    from src.shared.config import get_config

    config = get_config()  # Uses BP_ENVIRONMENT env var
    config = get_config("dev")  # Explicit environment

    # Access config values
    bucket = config.storage.buckets.main
    threshold = config.validation.quality.max_null_ratio
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Configuration Models
# =============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "boston-pulse"
    version: str = "0.1.0"
    description: str = "Urban analytics platform for Boston public safety"


class BucketsConfig(BaseModel):
    """GCS bucket configuration."""

    main: str = "boston-pulse-data"
    dvc: str = "boston-pulse-dvc"
    temp: str = "boston-pulse-temp"


class StoragePathsConfig(BaseModel):
    """Storage path configuration."""

    raw: str = "raw"
    processed: str = "processed"
    features: str = "features"
    schemas: str = "schemas"
    model_cards: str = "model_cards"


class StorageEmulatorConfig(BaseModel):
    """Storage emulator configuration (for local dev)."""

    enabled: bool = False
    host: str = "http://localhost:4443"


class StorageConfig(BaseModel):
    """Storage configuration."""

    buckets: BucketsConfig = Field(default_factory=BucketsConfig)
    paths: StoragePathsConfig = Field(default_factory=StoragePathsConfig)
    emulator: StorageEmulatorConfig = Field(default_factory=StorageEmulatorConfig)
    formats: dict[str, str] = Field(
        default_factory=lambda: {"raw": "parquet", "processed": "parquet", "features": "parquet"}
    )


class RetriesConfig(BaseModel):
    """Retry configuration."""

    max_attempts: int = 3
    delay_seconds: int = 300
    exponential_backoff: bool = True


class WatermarkConfig(BaseModel):
    """Watermark configuration for incremental ingestion."""

    enabled: bool = True
    lookback_days: int = 7


class DatasetsConfig(BaseModel):
    """Dataset defaults configuration."""

    default_schedule: str = "0 2 * * *"
    retries: RetriesConfig = Field(default_factory=RetriesConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)


class SchemaValidationConfig(BaseModel):
    """Schema validation configuration."""

    strict_mode: bool = True
    allow_extra_columns: bool = False


class QualityConfig(BaseModel):
    """Data quality thresholds."""

    max_null_ratio: float = 0.05
    max_duplicate_ratio: float = 0.01
    min_row_count: int = 100


class GeoBoundsConfig(BaseModel):
    """Geographic bounds for Boston."""

    min_lat: float = 42.2
    max_lat: float = 42.4
    min_lon: float = -71.2
    max_lon: float = -70.9


class TemporalValidationConfig(BaseModel):
    """Temporal validation configuration."""

    max_future_days: int = 1
    max_past_years: int = 10


class ValidationConfig(BaseModel):
    """Validation configuration."""

    schema: SchemaValidationConfig = Field(default_factory=SchemaValidationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    geo_bounds: GeoBoundsConfig = Field(default_factory=GeoBoundsConfig)
    temporal: TemporalValidationConfig = Field(default_factory=TemporalValidationConfig)


class PSIConfig(BaseModel):
    """Population Stability Index thresholds."""

    warning: float = 0.1
    critical: float = 0.25


class DriftReferenceConfig(BaseModel):
    """Drift reference configuration."""

    window_days: int = 30
    min_samples: int = 1000


class DriftConfig(BaseModel):
    """Drift detection configuration."""

    psi: PSIConfig = Field(default_factory=PSIConfig)
    reference: DriftReferenceConfig = Field(default_factory=DriftReferenceConfig)


class FairnessThresholdConfig(BaseModel):
    """Fairness threshold configuration."""

    warning: float = 0.2
    critical: float = 0.4


class FairnessThresholdsConfig(BaseModel):
    """All fairness thresholds."""

    representation: FairnessThresholdConfig = Field(default_factory=FairnessThresholdConfig)
    outcome_disparity: FairnessThresholdConfig = Field(
        default_factory=lambda: FairnessThresholdConfig(warning=0.15, critical=0.3)
    )


class FairnessConfig(BaseModel):
    """Fairness configuration."""

    default_slices: list[str] = Field(
        default_factory=lambda: ["neighborhood", "hour_of_day", "day_of_week"]
    )
    thresholds: FairnessThresholdsConfig = Field(default_factory=FairnessThresholdsConfig)
    gate_enabled: bool = False


class AlertRoutingConfig(BaseModel):
    """Alert routing configuration."""

    info: list[str] = Field(default_factory=lambda: ["log"])
    warning: list[str] = Field(default_factory=lambda: ["log", "slack"])
    critical: list[str] = Field(default_factory=lambda: ["log", "slack", "email"])


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    max_alerts_per_hour: int = 10
    cooldown_minutes: int = 15


class AlertingConfig(BaseModel):
    """Alerting configuration."""

    levels: list[str] = Field(default_factory=lambda: ["info", "warning", "critical"])
    routing: AlertRoutingConfig = Field(default_factory=AlertRoutingConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    include_timestamp: bool = True
    include_trace_id: bool = True


class FeatureStoreConfig(BaseModel):
    """Feature store configuration."""

    collections: dict[str, str] = Field(
        default_factory=lambda: {
            "crime_features": "crime_features",
            "transit_features": "transit_features",
            "safety_scores": "safety_scores",
        }
    )
    ttl_hours: int = 24


class APIConfig(BaseModel):
    """API configuration."""

    base_url: str
    timeout_seconds: int = 60
    rate_limit_per_minute: int = 60
    api_key_env: str | None = None


class APIsConfig(BaseModel):
    """All API configurations."""

    analyze_boston: APIConfig = Field(
        default_factory=lambda: APIConfig(
            base_url="https://data.boston.gov/api/3/action",
            timeout_seconds=60,
            rate_limit_per_minute=60,
        )
    )
    mbta: APIConfig = Field(
        default_factory=lambda: APIConfig(
            base_url="https://api-v3.mbta.com",
            timeout_seconds=30,
            rate_limit_per_minute=100,
        )
    )


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """
    Main configuration class for Boston Pulse.

    Loads configuration from:
    1. YAML files in configs/environments/
    2. Environment variables (for secrets)

    Environment variables take precedence over YAML values.
    """

    model_config = SettingsConfigDict(
        env_prefix="BP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    environment: Literal["dev", "prod"] = "dev"

    # Configuration sections
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    apis: APIsConfig = Field(default_factory=APIsConfig)

    # Secrets (from environment variables only)
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")
    mbta_api_key: str | None = Field(default=None, alias="MBTA_API_KEY")
    gcp_project_id: str | None = Field(default=None, alias="GCP_PROJECT_ID")

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = {"dev", "prod"}
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of: {valid_envs}")
        return v


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Try relative path from data-pipeline root
    config_dir = Path(__file__).parent.parent.parent / "configs"
    if config_dir.exists():
        return config_dir

    # Try from current working directory
    config_dir = Path.cwd() / "configs"
    if config_dir.exists():
        return config_dir

    # Try from data-pipeline subdirectory
    config_dir = Path.cwd() / "data-pipeline" / "configs"
    if config_dir.exists():
        return config_dir

    raise FileNotFoundError(
        "Could not find configs directory. "
        "Ensure you're running from the project root or data-pipeline directory."
    )


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_for_environment(environment: str) -> dict[str, Any]:
    """Load and merge configuration for a specific environment."""
    config_dir = _get_config_dir()
    env_dir = config_dir / "environments"

    # Load base config
    base_config = _load_yaml_file(env_dir / "base.yaml")

    # Load environment-specific config
    env_config = _load_yaml_file(env_dir / f"{environment}.yaml")

    # Remove inheritance marker if present
    env_config.pop("_inherit", None)

    # Merge configs
    merged = _deep_merge(base_config, env_config)
    merged["environment"] = environment

    return merged


@lru_cache(maxsize=4)
def get_config(environment: str | None = None) -> Settings:
    """
    Get configuration for the specified environment.

    Args:
        environment: Environment name (dev, prod).
                    If None, uses BP_ENVIRONMENT env var, defaulting to "dev".

    Returns:
        Settings: Validated configuration object.

    Example:
        config = get_config()  # Uses BP_ENVIRONMENT or defaults to dev
        config = get_config("prod")  # Explicit production config

        # Access values
        bucket = config.storage.buckets.main
        threshold = config.validation.quality.max_null_ratio
    """
    if environment is None:
        environment = os.getenv("BP_ENVIRONMENT", "dev")

    # Load YAML configuration
    yaml_config = _load_config_for_environment(environment)

    # Create Settings object (also loads env vars)
    return Settings(**yaml_config)


def reload_config(environment: str | None = None) -> Settings:
    """
    Reload configuration, clearing the cache.

    Useful for testing or when config files have changed.
    """
    get_config.cache_clear()
    return get_config(environment)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_bucket_path(layer: str, dataset: str, config: Settings | None = None) -> str:
    """
    Get the full GCS path for a dataset layer.

    Args:
        layer: Data layer (raw, processed, features)
        dataset: Dataset name (crime, service_311, etc.)
        config: Optional config object (uses default if not provided)

    Returns:
        Full GCS path like "gs://boston-pulse-data/raw/crime"
    """
    if config is None:
        config = get_config()

    bucket = config.storage.buckets.main
    layer_path = getattr(config.storage.paths, layer, layer)
    return f"gs://{bucket}/{layer_path}/{dataset}"


def is_production() -> bool:
    """Check if running in production environment."""
    config = get_config()
    return config.environment == "prod"


def is_development() -> bool:
    """Check if running in development environment."""
    config = get_config()
    return config.environment == "dev"
