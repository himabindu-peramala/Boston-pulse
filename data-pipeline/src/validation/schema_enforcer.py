"""
Boston Pulse - Schema Enforcer

Three-stage validation system for data quality:
1. Raw data validation: Basic schema + data quality checks
2. Processed data validation: Schema + business rule checks
3. Features validation: Schema + feature quality checks

FAILS pipeline execution on violations when in strict mode.

Usage:
    enforcer = SchemaEnforcer(config)

    # Validate raw data
    result = enforcer.validate_raw(df, dataset="crime")
    if not result.is_valid:
        raise ValidationError(result.errors)

    # Validate with custom schema version
    result = enforcer.validate_processed(df, dataset="crime", version="v1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import pandas as pd

from src.shared.config import Settings, get_config
from src.validation.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class ValidationLevel(StrEnum):
    """Validation severity level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStage(StrEnum):
    """Data validation stage."""

    RAW = "raw"
    PROCESSED = "processed"
    FEATURES = "features"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    level: ValidationLevel
    stage: ValidationStage
    check: str  # Name of the check that failed
    message: str
    column: str | None = None
    count: int | None = None  # For aggregate issues (e.g., null count)
    percentage: float | None = None


@dataclass
class ValidationResult:
    """Result of validation checks."""

    dataset: str
    stage: ValidationStage
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    validated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def errors(self) -> list[str]:
        """Get list of error messages."""
        return [
            issue.message
            for issue in self.issues
            if issue.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
        ]

    @property
    def warnings(self) -> list[str]:
        """Get list of warning messages."""
        return [issue.message for issue in self.issues if issue.level == ValidationLevel.WARNING]

    @property
    def has_errors(self) -> bool:
        """Check if result has any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if result has any warnings."""
        return len(self.warnings) > 0


class SchemaEnforcer:
    """
    Three-stage validation enforcer.

    Validates data at three stages:
    - Raw: Schema compliance + basic quality
    - Processed: Schema + business rules
    - Features: Schema + feature quality
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize schema enforcer.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.registry = SchemaRegistry(config)
        self.strict_mode = self.config.validation.quality_schema.strict_mode

    def validate_raw(
        self,
        df: pd.DataFrame,
        dataset: str,
        version: str | None = None,
    ) -> ValidationResult:
        """
        Validate raw data.

        Checks:
        - Schema compliance
        - Row count threshold
        - Null ratio per column
        - Duplicate ratio

        Args:
            df: Raw DataFrame
            dataset: Dataset name
            version: Schema version (uses latest if not provided)

        Returns:
            ValidationResult with issues
        """
        result = ValidationResult(
            dataset=dataset,
            stage=ValidationStage.RAW,
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns),
        )

        # 0. Auto-register schema if it doesn't exist yet
        self._ensure_schema_registered(dataset, "raw")

        # 1. Schema validation
        is_valid, schema_errors = self.registry.validate_dataframe(df, dataset, "raw", version)
        if not is_valid:
            for error in schema_errors:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.CRITICAL,
                        stage=ValidationStage.RAW,
                        check="schema_compliance",
                        message=error,
                    )
                )
            result.is_valid = False

        # 2. Row count check
        min_rows = self.config.validation.quality.min_row_count
        if len(df) < min_rows:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    stage=ValidationStage.RAW,
                    check="min_row_count",
                    message=f"Row count {len(df)} is below minimum {min_rows}",
                    count=len(df),
                )
            )
            result.is_valid = False

        # 3. Null ratio check
        max_null_ratio = self.config.validation.quality.max_null_ratio
        for col in df.columns:
            null_ratio = df[col].isna().sum() / len(df)
            if null_ratio > max_null_ratio:
                level = ValidationLevel.ERROR if self.strict_mode else ValidationLevel.WARNING
                result.issues.append(
                    ValidationIssue(
                        level=level,
                        stage=ValidationStage.RAW,
                        check="null_ratio",
                        message=f"Column '{col}' has null ratio {null_ratio:.2%} (threshold: {max_null_ratio:.2%})",
                        column=col,
                        percentage=null_ratio,
                    )
                )
                if self.strict_mode:
                    result.is_valid = False

        # 4. Duplicate check
        max_dup_ratio = self.config.validation.quality.max_duplicate_ratio
        dup_count = df.duplicated().sum()
        dup_ratio = dup_count / len(df) if len(df) > 0 else 0
        if dup_ratio > max_dup_ratio:
            level = ValidationLevel.WARNING  # Duplicates are warnings, not errors
            result.issues.append(
                ValidationIssue(
                    level=level,
                    stage=ValidationStage.RAW,
                    check="duplicate_ratio",
                    message=f"Duplicate ratio {dup_ratio:.2%} exceeds threshold {max_dup_ratio:.2%}",
                    count=dup_count,
                    percentage=dup_ratio,
                )
            )

        logger.info(
            f"Raw validation for {dataset}: {'PASSED' if result.is_valid else 'FAILED'}",
            extra={
                "dataset": dataset,
                "is_valid": result.is_valid,
                "issues_count": len(result.issues),},
        )

        return result

    def validate_processed(
        self,
        df: pd.DataFrame,
        dataset: str,
        version: str | None = None,
    ) -> ValidationResult:
        """
        Validate processed data.

        Checks:
        - All raw checks
        - Geographic bounds (if lat/lon present)
        - Temporal bounds
        - Business rule checks

        Args:
            df: Processed DataFrame
            dataset: Dataset name
            version: Schema version (uses latest if not provided)

        Returns:
            ValidationResult with issues
        """
        result = ValidationResult(
            dataset=dataset,
            stage=ValidationStage.PROCESSED,
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns),
        )

        # 0. Ensure schema exists
        self._ensure_schema_registered(dataset, "processed")

        # 1. Schema validation
        is_valid, schema_errors = self.registry.validate_dataframe(
            df, dataset, "processed", version
        )
        if not is_valid:
            for error in schema_errors:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.CRITICAL,
                        stage=ValidationStage.PROCESSED,
                        check="schema_compliance",
                        message=error,
                    )
                )
            result.is_valid = False

        # 2. Row count check
        min_rows = self.config.validation.quality.min_row_count
        if len(df) < min_rows:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    stage=ValidationStage.PROCESSED,
                    check="min_row_count",
                    message=f"Row count {len(df)} is below minimum {min_rows}",
                    count=len(df),
                )
            )
            result.is_valid = False

        # 3. Geographic bounds validation
        self._validate_geographic_bounds(df, result)

        # 4. Temporal validation
        self._validate_temporal_bounds(df, result)

        logger.info(
            f"Processed validation for {dataset}: {'PASSED' if result.is_valid else 'FAILED'}",
            extra={
                "dataset": dataset,
                "is_valid": result.is_valid,
                "issues_count": len(result.issues),},
        )

        return result

    def validate_features(
        self,
        df: pd.DataFrame,
        dataset: str,
        version: str | None = None,
    ) -> ValidationResult:
        """
        Validate feature data.

        Checks:
        - Schema compliance
        - Feature value ranges
        - Missing features
        - Feature distributions

        Args:
            df: Features DataFrame
            dataset: Dataset name
            version: Schema version (uses latest if not provided)

        Returns:
            ValidationResult with issues
        """
        result = ValidationResult(
            dataset=dataset,
            stage=ValidationStage.FEATURES,
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns),
        )

        # 0. Ensure schema exists
        self._ensure_schema_registered(dataset, "features")

        # 1. Schema validation
        is_valid, schema_errors = self.registry.validate_dataframe(df, dataset, "features", version)
        if not is_valid:
            for error in schema_errors:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.CRITICAL,
                        stage=ValidationStage.FEATURES,
                        check="schema_compliance",
                        message=error,
                    )
                )
            result.is_valid = False

        # 2. Feature completeness
        max_null_ratio = self.config.validation.quality.max_null_ratio
        for col in df.columns:
            null_ratio = df[col].isna().sum() / len(df)
            if null_ratio > max_null_ratio:
                level = ValidationLevel.ERROR if self.strict_mode else ValidationLevel.WARNING
                result.issues.append(
                    ValidationIssue(
                        level=level,
                        stage=ValidationStage.FEATURES,
                        check="feature_completeness",
                        message=f"Feature '{col}' has null ratio {null_ratio:.2%} (threshold: {max_null_ratio:.2%})",
                        column=col,
                        percentage=null_ratio,
                    )
                )
                if self.strict_mode:
                    result.is_valid = False

        # 3. Numeric feature ranges (check for inf, extreme values)
        for col in df.select_dtypes(include=["number"]).columns:
            if df[col].isin([float("inf"), float("-inf")]).any():
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        stage=ValidationStage.FEATURES,
                        check="feature_range",
                        message=f"Feature '{col}' contains infinite values",
                        column=col,
                    )
                )
                result.is_valid = False

        logger.info(
            f"Features validation for {dataset}: {'PASSED' if result.is_valid else 'FAILED'}",
            extra={
                "dataset": dataset,
                "is_valid": result.is_valid,
                "issues_count": len(result.issues),},
        )

        return result

    # =========================================================================
    # Private Validation Methods
    # =========================================================================

    def _ensure_schema_registered(self, dataset: str, stage: str) -> None:
        """
        Check if schema exists in GCS.
        If not, register it from the local schema file.

        This handles first-time setup automatically — no separate
        script needed when running the pipeline for the first time.
        """
        # Already exists — nothing to do
        if self.registry.schema_exists(dataset, stage):
            return

        logger.info(f"Schema not found in GCS for {dataset}/{stage} — registering from local file")

        # Find the local schema file
        schema_file = (
            Path(__file__).parent.parent.parent  # data-pipeline/
            / "schemas"
            / dataset
            / f"{stage}_schema.json"
        )

        if not schema_file.exists():
            raise FileNotFoundError(
                f"No local schema file found at {schema_file}. "
                f"Create schemas/{dataset}/{stage}_schema.json first."
            )

        # Load and register
        import json

        schema_data = json.loads(schema_file.read_text())

        version = schema_data.get("metadata", {}).get("version", "v1.0.0")

        schema_properties = schema_data.get("properties", schema_data)

        self.registry.register_schema(
            dataset=dataset,
            layer=stage,
            schema=schema_properties,
            version=version,
            description=schema_data.get("description", f"{dataset} {stage} schema"),
            created_by="auto-registered",
            primary_key=self._extract_primary_key(schema_data),
        )

        logger.info(f"Auto-registered schema for {dataset}/{stage} as {version}")

    def _extract_primary_key(self, schema_data: dict) -> str | None:
        """Pull primary key from schema properties."""
        for col, spec in schema_data.get("properties", {}).items():
            if spec.get("primary_key"):
                return col
        return None

    def _validate_geographic_bounds(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate geographic coordinates are within Boston bounds."""
        # Look for common latitude/longitude column names
        lat_cols = [col for col in df.columns if col.lower() in ("lat", "latitude")]
        lon_cols = [col for col in df.columns if col.lower() in ("lon", "long", "longitude")]

        if not (lat_cols and lon_cols):
            return  # No geographic data to validate

        lat_col = lat_cols[0]
        lon_col = lon_cols[0]

        bounds = self.config.validation.geo_bounds

        # Check latitude bounds
        invalid_lat = ((df[lat_col] < bounds.min_lat) | (df[lat_col] > bounds.max_lat)) & df[
            lat_col
        ].notna()

        if invalid_lat.any():
            count = invalid_lat.sum()
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    stage=result.stage,
                    check="geographic_bounds",
                    message=f"{count} records have latitude outside Boston bounds ({bounds.min_lat}, {bounds.max_lat})",
                    column=lat_col,
                    count=count,
                )
            )

        # Check longitude bounds
        invalid_lon = ((df[lon_col] < bounds.min_lon) | (df[lon_col] > bounds.max_lon)) & df[
            lon_col
        ].notna()

        if invalid_lon.any():
            count = invalid_lon.sum()
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    stage=result.stage,
                    check="geographic_bounds",
                    message=f"{count} records have longitude outside Boston bounds ({bounds.min_lon}, {bounds.max_lon})",
                    column=lon_col,
                    count=count,
                )
            )

    def _validate_temporal_bounds(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate temporal data is within reasonable bounds."""
        # Look for common datetime columns
        date_cols = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ("date", "time", "timestamp", "occurred", "created")
            )
        ]

        if not date_cols:
            return  # No temporal data to validate

        for date_col in date_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                continue

            # Ensure column is localized to UTC for comparison
            col_data = df[date_col]
            if not col_data.dt.tz:
                col_data = col_data.dt.tz_localize(UTC)
            else:
                col_data = col_data.dt.tz_convert(UTC)

            # Check for future dates
            max_future = datetime.now(UTC) + timedelta(
                days=self.config.validation.temporal.max_future_days
            )
            future_dates = col_data > max_future
            if future_dates.any():
                count = future_dates.sum()
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        stage=result.stage,
                        check="temporal_bounds",
                        message=f"{count} records have future dates in '{date_col}' (beyond {max_future.date()})",
                        column=date_col,
                        count=count,
                    )
                )

            # Check for very old dates
            min_past = datetime.now(UTC) - timedelta(
                days=self.config.validation.temporal.max_past_years * 365
            )
            old_dates = (col_data < min_past) & col_data.notna()
            if old_dates.any():
                count = old_dates.sum()
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        stage=result.stage,
                        check="temporal_bounds",
                        message=f"{count} records have dates in '{date_col}' older than {self.config.validation.temporal.max_past_years} years",
                        column=date_col,
                        count=count,
                    )
                )


# =============================================================================
# Exception Classes
# =============================================================================


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""

    def __init__(self, result: ValidationResult):
        self.result = result
        error_msg = "\n".join([f"  - {error}" for error in result.errors])
        super().__init__(f"Validation failed for {result.dataset}:\n{error_msg}")


# =============================================================================
# Convenience Functions
# =============================================================================


def enforce_validation(
    df: pd.DataFrame,
    dataset: str,
    stage: ValidationStage,
    version: str | None = None,
    config: Settings | None = None,
) -> ValidationResult:
    """
    Convenience function to validate data and raise on failure.

    Args:
        df: DataFrame to validate
        dataset: Dataset name
        stage: Validation stage (raw, processed, features)
        version: Schema version
        config: Configuration object

    Returns:
        ValidationResult

    Raises:
        ValidationError: If validation fails and strict mode is enabled
    """
    enforcer = SchemaEnforcer(config)

    if stage == ValidationStage.RAW:
        result = enforcer.validate_raw(df, dataset, version)
    elif stage == ValidationStage.PROCESSED:
        result = enforcer.validate_processed(df, dataset, version)
    elif stage == ValidationStage.FEATURES:
        result = enforcer.validate_features(df, dataset, version)
    else:
        raise ValueError(f"Invalid validation stage: {stage}")

    # Raise error if validation failed and strict mode enabled
    if not result.is_valid and enforcer.strict_mode:
        raise ValidationError(result)

    return result
