"""Navigate schema validation — extends SchemaEnforcer with config-driven checks."""

from __future__ import annotations

import logging

import pandas as pd

from src.shared.config import get_dataset_config
from src.validation.schema_enforcer import (
    SchemaEnforcer,
    ValidationIssue,
    ValidationLevel,
    ValidationResult,
    ValidationStage,
)

logger = logging.getLogger(__name__)


def _cfg() -> dict:
    return get_dataset_config("crime_navigate")


class NavigateSchemaEnforcer(SchemaEnforcer):
    """
    Extends base SchemaEnforcer with crime_navigate-specific rules.

    Key overrides:
      validate_raw — completely replaces the base implementation because
        the base class null ratio check runs on ALL columns including
        OFFENSE_CODE_GROUP and UCR_PART which are always 100% null by
        design (documented in EDA). Calling super() would always fail.

      validate_navigate_processed / validate_navigate_features — call
        super() and add Navigate-specific checks on top. The base class
        behaviour is correct for these stages.
    """

    DATASET = "crime_navigate"

    # Columns that are always null in the raw API response by design.
    # Documented in configs/datasets/crime_navigate.yaml data_quality_notes.
    # Never check null ratios for these at the raw stage.
    _ALWAYS_NULL_RAW = frozenset(
        {
            "OFFENSE_CODE_GROUP",
            "UCR_PART",
        }
    )

    # -------------------------------------------------------------------------
    # Raw validation — full override, does NOT call super()
    # -------------------------------------------------------------------------

    def validate_raw(
        self,
        df: pd.DataFrame,
        dataset: str,
        _version: str | None = None,
    ) -> ValidationResult:
        """
        Override base validate_raw entirely for Navigate raw data.

        Why not call super():
          1. Base iterates ALL columns for null ratios — hits OFFENSE_CODE_GROUP
             and UCR_PART which are always 100% null (API design, not data error).
          2. Base reads min_row_count from global config (500) not dataset config (1).
          3. Base may reject extra columns depending on allow_extra_columns setting.
             Raw stage must accept all API columns — preprocessing drops the extras.

        All thresholds come from configs/datasets/crime_navigate.yaml.
        Nothing is hardcoded here.
        """
        raw_cfg = _cfg().get("validation", {}).get("raw", {})

        result = ValidationResult(
            dataset=dataset,
            stage=ValidationStage.RAW,
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns),
        )

        # ── 1. Required columns present ───────────────────────────────────────
        # Check only that the columns we USE are present.
        # Extra columns from the API are expected — do not reject them.
        required = raw_cfg.get(
            "required_columns",
            ["INCIDENT_NUMBER", "OCCURRED_ON_DATE"],
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    stage=ValidationStage.RAW,
                    check="required_columns",
                    message=f"Missing required columns: {missing}",
                )
            )
            result.is_valid = False

        # ── 2. Minimum row count ──────────────────────────────────────────────
        # Read from dataset config, not global default.
        # Incremental 3-day runs can legitimately return very few rows.
        min_rows = raw_cfg.get("min_row_count", 1)
        if len(df) < min_rows:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    stage=ValidationStage.RAW,
                    check="min_row_count",
                    message=f"Row count {len(df)} below minimum {min_rows}",
                    count=len(df),
                )
            )
            result.is_valid = False

        # ── 3. Null ratio — only for columns that should not be null ──────────
        # Explicitly skip _ALWAYS_NULL_RAW columns.
        # Threshold per column also comes from dataset config.
        null_thresholds = raw_cfg.get("max_null_ratio", {})
        default_null_threshold = raw_cfg.get("default_max_null_ratio", 0.20)

        columns_to_check = [c for c in df.columns if c not in self._ALWAYS_NULL_RAW]
        for col in columns_to_check:
            threshold = null_thresholds.get(col, default_null_threshold)
            null_ratio = df[col].isna().mean()
            # Also count empty strings as null for string columns
            if df[col].dtype == object:
                null_ratio = (df[col].isna() | (df[col].astype(str).str.strip() == "")).mean()
            if null_ratio > threshold:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        stage=ValidationStage.RAW,
                        check="null_ratio",
                        message=(
                            f"Column '{col}' null ratio {null_ratio:.1%} "
                            f"exceeds threshold {threshold:.1%}"
                        ),
                        column=col,
                        percentage=null_ratio,
                    )
                )
                # Null ratio at raw stage is WARNING only — not a pipeline failure.
                # The preprocessor handles nulls. Only missing required columns
                # or zero rows should fail the pipeline at this stage.

        # ── 4. Duplicate INCIDENT_NUMBER ─────────────────────────────────────
        # Duplicates are expected from API retries. Warning only.
        # Preprocessor deduplicates on INCIDENT_NUMBER keeping last.
        if "INCIDENT_NUMBER" in df.columns:
            dup_count = df["INCIDENT_NUMBER"].duplicated().sum()
            if dup_count > 0:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        stage=ValidationStage.RAW,
                        check="duplicate_incidents",
                        message=(
                            f"{dup_count} duplicate INCIDENT_NUMBER values "
                            f"— will be deduplicated in preprocessing"
                        ),
                        count=dup_count,
                    )
                )

        logger.info(
            f"Raw validation for {dataset}: "
            f"{'PASSED' if result.is_valid else 'FAILED'} "
            f"({len(result.errors)} errors, {len(result.warnings)} warnings)"
        )

        return result

    def validate_navigate_raw(self, df: pd.DataFrame) -> ValidationResult:
        """Entry point called by the DAG task. Delegates to validate_raw override."""
        return self.validate_raw(df, self.DATASET)

    # -------------------------------------------------------------------------
    # Processed validation — calls super() + Navigate-specific checks
    # -------------------------------------------------------------------------

    def validate_navigate_processed(self, df: pd.DataFrame) -> ValidationResult:
        """Processed validation: base checks + Navigate-specific column checks."""
        result = self.validate_processed(df, self.DATASET)

        if "h3_index" in df.columns and df["h3_index"].isna().all():
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    stage=ValidationStage.PROCESSED,
                    check="h3_index_all_null",
                    message="All h3_index values are null — no valid coordinates in input",
                )
            )

        if "hour_bucket" in df.columns:
            invalid = ~df["hour_bucket"].between(0, 5)
            if invalid.any():
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        stage=ValidationStage.PROCESSED,
                        check="hour_bucket_range",
                        message=(
                            f"hour_bucket must be 0-5, "
                            f"found invalid values: {df.loc[invalid, 'hour_bucket'].tolist()[:5]}"
                        ),
                    )
                )
                result.is_valid = False

        if "severity_weight" in df.columns and (df["severity_weight"] <= 0).any():
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    stage=ValidationStage.PROCESSED,
                    check="severity_weight_nonpositive",
                    message="Some severity_weight values are <= 0",
                )
            )

        return result

    # -------------------------------------------------------------------------
    # Features validation — calls super() + Navigate-specific checks
    # -------------------------------------------------------------------------

    def validate_navigate_features(self, df: pd.DataFrame) -> ValidationResult:
        """Features validation: base checks + Navigate-specific feature checks."""
        result = self.validate_features(df, self.DATASET)
        features_cfg = _cfg().get("validation", {}).get("features", {})

        if "h3_index" in df.columns:
            min_cells = features_cfg.get("min_h3_cells", 800)
            n_cells = df["h3_index"].nunique()
            if n_cells < min_cells:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        stage=ValidationStage.FEATURES,
                        check="min_h3_cells",
                        message=f"H3 cell count {n_cells} below minimum {min_cells}",
                    )
                )
                result.is_valid = False

        if "risk_score" in df.columns:
            bounds = features_cfg.get("score_bounds", {})
            lo = bounds.get("min", 0)
            hi = bounds.get("max", 100)
            if (df["risk_score"] < lo).any() or (df["risk_score"] > hi).any():
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        stage=ValidationStage.FEATURES,
                        check="risk_score_bounds",
                        message=f"risk_score values outside [{lo}, {hi}]",
                    )
                )
                result.is_valid = False

        return result
