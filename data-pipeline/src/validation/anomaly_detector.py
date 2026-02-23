"""
Boston Pulse - Anomaly Detector

Detect anomalies in data:
- Missing value patterns
- Outliers (statistical and domain-based)
- Coordinate bounds violations
- Temporal anomalies
- Categorical value anomalies

Usage:
    detector = AnomalyDetector(config)

    # Detect all anomalies
    anomalies = detector.detect_anomalies(df, dataset="crime")

    if anomalies.has_critical_anomalies:
        for anomaly in anomalies.critical_anomalies:
            print(f"CRITICAL: {anomaly.message}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    """Anomaly severity level."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Type of anomaly detected."""

    MISSING_VALUES = "missing_values"
    OUTLIER = "outlier"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    DUPLICATE = "duplicate"


@dataclass
class Anomaly:
    """Individual anomaly detection."""

    type: AnomalyType
    severity: AnomalySeverity
    feature: str
    message: str
    count: int | None = None
    percentage: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    dataset: str
    detected_at: datetime
    row_count: int
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.anomalies) > 0

    @property
    def has_critical_anomalies(self) -> bool:
        """Check if any critical anomalies were detected."""
        return any(a.severity == AnomalySeverity.CRITICAL for a in self.anomalies)

    @property
    def critical_anomalies(self) -> list[Anomaly]:
        """Get list of critical anomalies."""
        return [a for a in self.anomalies if a.severity == AnomalySeverity.CRITICAL]

    @property
    def warning_anomalies(self) -> list[Anomaly]:
        """Get list of warning anomalies."""
        return [a for a in self.anomalies if a.severity == AnomalySeverity.WARNING]

    @property
    def anomalies_by_type(self) -> dict[AnomalyType, list[Anomaly]]:
        """Group anomalies by type."""
        result: dict[AnomalyType, list[Anomaly]] = {}
        for anomaly in self.anomalies:
            if anomaly.type not in result:
                result[anomaly.type] = []
            result[anomaly.type].append(anomaly)
        return result


class AnomalyDetector:
    """
    Detect various types of anomalies in datasets.

    Uses statistical methods and domain knowledge to identify
    data quality issues that may require attention.
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize anomaly detector.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        dataset: str,
        check_outliers: bool = True,
        check_geographic: bool = True,
        check_temporal: bool = True,
        check_categorical: bool = True,
    ) -> AnomalyResult:
        """
        Detect all anomalies in a DataFrame.

        Args:
            df: DataFrame to check
            dataset: Dataset name
            check_outliers: Enable outlier detection
            check_geographic: Enable geographic bounds checking
            check_temporal: Enable temporal anomaly detection
            check_categorical: Enable categorical value checking

        Returns:
            AnomalyResult with all detected anomalies
        """
        logger.info(
            f"Detecting anomalies for {dataset}",
            extra={"dataset": dataset, "rows": len(df), "columns": len(df.columns)},
        )

        result = AnomalyResult(
            dataset=dataset,
            detected_at=datetime.now(UTC),
            row_count=len(df),
        )

        # 1. Missing value patterns
        result.anomalies.extend(self._detect_missing_patterns(df))

        # 2. Outliers (if enabled)
        if check_outliers:
            result.anomalies.extend(self._detect_outliers(df))

        # 3. Geographic anomalies (if enabled and applicable)
        if check_geographic:
            result.anomalies.extend(self._detect_geographic_anomalies(df))

        # 4. Temporal anomalies (if enabled and applicable)
        if check_temporal:
            result.anomalies.extend(self._detect_temporal_anomalies(df))

        # 5. Categorical anomalies (if enabled)
        if check_categorical:
            result.anomalies.extend(self._detect_categorical_anomalies(df))

        # 6. Duplicate detection
        result.anomalies.extend(self._detect_duplicates(df))

        # TODO: Add more anomaly detection methods (L_inf, JS divergence, etc.) using Evidently
        # Or identify anomalies using domain knowledge or anything relevant to the dataset

        logger.info(
            f"Anomaly detection complete for {dataset}: "
            f"{len(result.anomalies)} anomalies found "
            f"({len(result.critical_anomalies)} critical)",
            extra={
                "dataset": dataset,
                "total_anomalies": len(result.anomalies),
                "critical_count": len(result.critical_anomalies),
            },
        )

        return result

    # =========================================================================
    # Private Detection Methods
    # =========================================================================

    def _detect_missing_patterns(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect unusual missing value patterns."""
        anomalies = []

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_ratio = null_count / len(df)

            # High missing value ratio
            if null_ratio > 0.5:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.MISSING_VALUES,
                        severity=AnomalySeverity.CRITICAL,
                        feature=col,
                        message=f"Column '{col}' has {null_ratio:.1%} missing values",
                        count=null_count,
                        percentage=null_ratio,
                    )
                )
            elif null_ratio > 0.2:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.MISSING_VALUES,
                        severity=AnomalySeverity.WARNING,
                        feature=col,
                        message=f"Column '{col}' has {null_ratio:.1%} missing values",
                        count=null_count,
                        percentage=null_ratio,
                    )
                )

        return anomalies

    def _detect_outliers(self, df: pd.DataFrame, z_threshold: float = 3.5) -> list[Anomaly]:
        """Detect statistical outliers using z-score and IQR methods."""
        anomalies = []

        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()

            if len(col_data) < 10:  # Need sufficient data
                continue

            # Z-score method
            z_scores = np.abs(scipy_stats.zscore(col_data))
            outliers_z = (z_scores > z_threshold).sum()
            outlier_ratio_z = outliers_z / len(col_data)

            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            outlier_ratio_iqr = outliers_iqr / len(col_data)

            # Use the more sensitive estimate
            outlier_count = int(max(outliers_z, outliers_iqr))
            outlier_ratio = max(outlier_ratio_z, outlier_ratio_iqr)

            # Report if significant outliers found
            if outlier_ratio > 0.05:  # More than 5% outliers
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.OUTLIER,
                        severity=AnomalySeverity.WARNING,
                        feature=col,
                        message=f"Column '{col}' has {outlier_ratio:.1%} outliers",
                        count=outlier_count,
                        percentage=outlier_ratio,
                        details={
                            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                        },
                    )
                )

        return anomalies

    def _detect_geographic_anomalies(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect geographic coordinate anomalies."""
        anomalies = []

        # Look for latitude/longitude columns
        lat_cols = [col for col in df.columns if col.lower() in ("lat", "latitude")]
        lon_cols = [col for col in df.columns if col.lower() in ("lon", "long", "longitude")]

        if not (lat_cols and lon_cols):
            return anomalies  # No geographic data

        lat_col = lat_cols[0]
        lon_col = lon_cols[0]

        bounds = self.config.validation.geo_bounds

        # Check latitude bounds
        invalid_lat = ((df[lat_col] < bounds.min_lat) | (df[lat_col] > bounds.max_lat)) & df[
            lat_col
        ].notna()

        if invalid_lat.any():
            count = invalid_lat.sum()
            anomalies.append(
                Anomaly(
                    type=AnomalyType.GEOGRAPHIC,
                    severity=AnomalySeverity.WARNING,
                    feature=lat_col,
                    message=f"{count} records with latitude outside Boston bounds",
                    count=count,
                    percentage=count / len(df),
                    details={
                        "bounds": {"min": bounds.min_lat, "max": bounds.max_lat},
                        "actual_range": {
                            "min": float(df[lat_col].min()),
                            "max": float(df[lat_col].max()),
                        },
                    },
                )
            )

        # Check longitude bounds
        invalid_lon = ((df[lon_col] < bounds.min_lon) | (df[lon_col] > bounds.max_lon)) & df[
            lon_col
        ].notna()

        if invalid_lon.any():
            count = invalid_lon.sum()
            anomalies.append(
                Anomaly(
                    type=AnomalyType.GEOGRAPHIC,
                    severity=AnomalySeverity.WARNING,
                    feature=lon_col,
                    message=f"{count} records with longitude outside Boston bounds",
                    count=count,
                    percentage=count / len(df),
                    details={
                        "bounds": {"min": bounds.min_lon, "max": bounds.max_lon},
                        "actual_range": {
                            "min": float(df[lon_col].min()),
                            "max": float(df[lon_col].max()),
                        },
                    },
                )
            )

        # Check for impossible coordinates (0, 0)
        zero_coords = ((df[lat_col] == 0) & (df[lon_col] == 0)).sum()
        if zero_coords > 0:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.GEOGRAPHIC,
                    severity=AnomalySeverity.WARNING,
                    feature=f"{lat_col}/{lon_col}",
                    message=f"{zero_coords} records with (0, 0) coordinates",
                    count=zero_coords,
                    percentage=zero_coords / len(df),
                )
            )

        return anomalies

    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect temporal anomalies in datetime columns."""
        anomalies = []

        # Find datetime columns
        date_cols = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ("date", "time", "timestamp", "occurred", "created")
            )
        ]

        for date_col in date_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                continue

            col_data = df[date_col].dropna()
            if len(col_data) == 0:
                continue

            # Check for future dates
            now = datetime.now(UTC)
            # Ensure col_data is localized to UTC for comparison
            if not col_data.dt.tz:
                col_data = col_data.dt.tz_localize(UTC)
            else:
                col_data = col_data.dt.tz_convert(UTC)

            max_future = now + timedelta(days=self.config.validation.temporal.max_future_days)
            future_dates = (col_data > max_future).sum()

            if future_dates > 0:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.TEMPORAL,
                        severity=AnomalySeverity.WARNING,
                        feature=date_col,
                        message=f"{future_dates} records with future dates in '{date_col}'",
                        count=future_dates,
                        percentage=future_dates / len(df),
                    )
                )

            # Check for very old dates
            min_past = now - timedelta(days=self.config.validation.temporal.max_past_years * 365)
            old_dates = (col_data < min_past).sum()

            if old_dates > 0:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.TEMPORAL,
                        severity=AnomalySeverity.INFO,
                        feature=date_col,
                        message=f"{old_dates} records with dates older than {self.config.validation.temporal.max_past_years} years",
                        count=old_dates,
                        percentage=old_dates / len(df),
                    )
                )

        return anomalies

    def _detect_categorical_anomalies(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect anomalies in categorical columns."""
        anomalies = []

        for col in df.select_dtypes(include=["object", "category"]).columns:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            # Check for too many unique values (possible data quality issue)
            unique_count = col_data.nunique()
            unique_ratio = unique_count / len(col_data)

            # If almost all values are unique, might not be categorical
            if unique_ratio > 0.9 and unique_count > 100:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.CATEGORICAL,
                        severity=AnomalySeverity.INFO,
                        feature=col,
                        message=f"Column '{col}' has {unique_count} unique values ({unique_ratio:.1%} of rows)",
                        count=unique_count,
                        percentage=unique_ratio,
                    )
                )

            # Check for singleton values (categories that appear only once)
            value_counts = col_data.value_counts()
            singletons = (value_counts == 1).sum()
            singleton_ratio = singletons / len(value_counts)

            if singleton_ratio > 0.5 and unique_count > 20:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.CATEGORICAL,
                        severity=AnomalySeverity.INFO,
                        feature=col,
                        message=f"Column '{col}' has {singletons} singleton categories ({singleton_ratio:.1%})",
                        count=singletons,
                        percentage=singleton_ratio,
                    )
                )

        return anomalies

    def _detect_duplicates(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect duplicate rows."""
        anomalies = []

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_ratio = dup_count / len(df)
            if dup_ratio > 0.1:
                severity = AnomalySeverity.CRITICAL
            elif dup_ratio > 0.01:
                severity = AnomalySeverity.WARNING
            else:
                severity = AnomalySeverity.INFO

            anomalies.append(
                Anomaly(
                    type=AnomalyType.DUPLICATE,
                    severity=severity,
                    feature="__all__",
                    message=f"Found {dup_count} duplicate rows ({dup_ratio:.2%})",
                    count=dup_count,
                    percentage=dup_ratio,
                )
            )

        return anomalies


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_anomalies(
    df: pd.DataFrame,
    dataset: str,
    config: Settings | None = None,
) -> AnomalyResult:
    """
    Convenience function to detect anomalies.

    Args:
        df: DataFrame to check
        dataset: Dataset name
        config: Configuration object

    Returns:
        AnomalyResult
    """
    detector = AnomalyDetector(config)
    return detector.detect_anomalies(df, dataset)
