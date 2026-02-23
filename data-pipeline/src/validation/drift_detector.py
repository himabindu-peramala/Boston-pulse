"""
Boston Pulse - Drift Detector

Detect data drift using:
- Population Stability Index (PSI) for distribution changes
- Statistical tests for numerical features
- Categorical distribution comparisons
- Evidently library for advanced drift detection

Drift detection helps identify when data distributions change significantly,
which may indicate data quality issues or shifts in real-world patterns.

Usage:
    detector = DriftDetector(config)

    # Detect drift between current and reference data
    drift_result = detector.detect_drift(
        current_df=new_data,
        reference_df=baseline_data,
        dataset="crime"
    )

    if drift_result.has_critical_drift:
        print(f"CRITICAL DRIFT DETECTED: {drift_result.critical_features}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from src.shared.config import Settings, get_config
from src.validation.statistics_generator import StatisticsGenerator

logger = logging.getLogger(__name__)


class DriftSeverity(StrEnum):
    """Drift severity level."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FeatureDrift:
    """Drift metrics for a single feature."""

    feature_name: str
    psi: float  # Population Stability Index
    severity: DriftSeverity
    reference_stats: dict[str, Any]
    current_stats: dict[str, Any]
    drift_type: str  # "numerical" or "categorical"


@dataclass
class DriftResult:
    """Result of drift detection."""

    dataset: str
    reference_date: datetime
    current_date: datetime
    features_analyzed: int
    features_with_drift: list[FeatureDrift] = field(default_factory=list)

    @property
    def has_warning_drift(self) -> bool:
        """Check if any feature has warning-level drift."""
        return any(f.severity == DriftSeverity.WARNING for f in self.features_with_drift)

    @property
    def has_critical_drift(self) -> bool:
        """Check if any feature has critical-level drift."""
        return any(f.severity == DriftSeverity.CRITICAL for f in self.features_with_drift)

    @property
    def warning_features(self) -> list[str]:
        """Get list of features with warning drift."""
        return [
            f.feature_name for f in self.features_with_drift if f.severity == DriftSeverity.WARNING
        ]

    @property
    def critical_features(self) -> list[str]:
        """Get list of features with critical drift."""
        return [
            f.feature_name for f in self.features_with_drift if f.severity == DriftSeverity.CRITICAL
        ]


class DriftDetector:
    """
    Detect data drift using PSI and statistical tests.

    Compares current data against a reference (baseline) to identify
    distribution changes that may indicate data quality issues.
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize drift detector.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.stats_generator = StatisticsGenerator(config)

        # Thresholds from config
        self.psi_warning_threshold = self.config.drift.psi.warning
        self.psi_critical_threshold = self.config.drift.psi.critical

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        dataset: str,
    ) -> DriftResult:
        """
        Detect drift between current and reference data.

        Args:
            current_df: Current dataset
            reference_df: Reference (baseline) dataset
            dataset: Dataset name

        Returns:
            DriftResult with drift metrics for each feature

        Example:
            result = detector.detect_drift(new_data, baseline_data, "crime")
            if result.has_critical_drift:
                print(f"Critical drift in: {result.critical_features}")
        """
        logger.info(
            f"Detecting drift for {dataset}",
            extra={
                "dataset": dataset,
                "current_rows": len(current_df),
                "reference_rows": len(reference_df),},
        )

        drift_result = DriftResult(
            dataset=dataset,
            reference_date=datetime.now(UTC),  # Would be loaded from metadata
            current_date=datetime.now(UTC),
            features_analyzed=0,
        )

        # Analyze common columns
        common_columns = set(current_df.columns) & set(reference_df.columns)
        drift_result.features_analyzed = len(common_columns)

        for col in common_columns:
            try:
                feature_drift = self._detect_feature_drift(
                    current_df[col],
                    reference_df[col],
                    col,
                )

                if feature_drift.severity != DriftSeverity.NONE:
                    drift_result.features_with_drift.append(feature_drift)

            except Exception as e:
                logger.warning(
                    f"Failed to detect drift for column {col}: {e}",
                    extra={"dataset": dataset, "column": col},
                )

        logger.info(
            f"Drift detection complete for {dataset}: "
            f"{len(drift_result.warning_features)} warnings, "
            f"{len(drift_result.critical_features)} critical",
            extra={
                "dataset": dataset,
                "warning_count": len(drift_result.warning_features),
                "critical_count": len(drift_result.critical_features),},
        )

        return drift_result

    def detect_drift_with_evidently(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        dataset: str,
    ) -> DriftResult:
        """
        Detect drift using Evidently library for more sophisticated analysis.

        This method uses Evidently's built-in drift detection which includes:
        - Kolmogorov-Smirnov test for numerical features
        - Chi-squared test for categorical features
        - Jensen-Shannon divergence

        Args:
            current_df: Current dataset
            reference_df: Reference (baseline) dataset
            dataset: Dataset name

        Returns:
            DriftResult with drift metrics for each feature
        """
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        logger.info(
            f"Detecting drift with Evidently for {dataset}",
            extra={
                "dataset": dataset,
                "current_rows": len(current_df),
                "reference_rows": len(reference_df),},
        )

        # Create column mapping
        numerical_features = current_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = current_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        column_mapping = ColumnMapping(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        # Generate drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        # Extract results
        report_dict = report.as_dict()

        drift_result = DriftResult(
            dataset=dataset,
            reference_date=datetime.now(UTC),
            current_date=datetime.now(UTC),
            features_analyzed=len(current_df.columns),
        )

        # Parse Evidently results
        try:
            metrics = report_dict.get("metrics", [])
            for metric in metrics:
                if metric.get("metric") == "DataDriftTable":
                    drift_by_columns = metric.get("result", {}).get("drift_by_columns", {})

                    for col_name, col_data in drift_by_columns.items():
                        drift_detected = col_data.get("drift_detected", False)
                        drift_score = col_data.get("drift_score", 0.0)
                        stattest_name = col_data.get("stattest_name", "unknown")

                        if drift_detected:
                            # Map Evidently's drift score to our PSI-like metric
                            severity = self._determine_severity_from_evidently(
                                drift_score, stattest_name
                            )

                            if severity != DriftSeverity.NONE:
                                feature_drift = FeatureDrift(
                                    feature_name=col_name,
                                    psi=drift_score,
                                    severity=severity,
                                    reference_stats={"stattest": stattest_name},
                                    current_stats={"drift_score": drift_score},
                                    drift_type=(
                                        "numerical"
                                        if col_name in numerical_features
                                        else "categorical"
                                    ),
                                )
                                drift_result.features_with_drift.append(feature_drift)
        except Exception as e:
            logger.warning(f"Error parsing Evidently results: {e}")

        logger.info(
            f"Evidently drift detection complete for {dataset}: "
            f"{len(drift_result.warning_features)} warnings, "
            f"{len(drift_result.critical_features)} critical",
        )

        return drift_result

    def _determine_severity_from_evidently(
        self, drift_score: float, stattest_name: str
    ) -> DriftSeverity:
        """Determine severity from Evidently drift metrics."""
        # For p-value based tests (lower = more drift)
        if "chi2" in stattest_name.lower() or "ks" in stattest_name.lower():
            if drift_score < 0.01:
                return DriftSeverity.CRITICAL
            elif drift_score < 0.05:
                return DriftSeverity.WARNING
            return DriftSeverity.NONE

        # For distance-based tests (higher = more drift)
        if drift_score > self.psi_critical_threshold:
            return DriftSeverity.CRITICAL
        elif drift_score > self.psi_warning_threshold:
            return DriftSeverity.WARNING

        return DriftSeverity.NONE

    def _calculate_psi_from_stats(self, current: Any, reference: Any) -> float:
        """Calculate PSI-like metric from pre-computed statistics."""
        if current.mean is not None and reference.mean is not None:
            # Numerical: use mean shift normalized by std
            if reference.std and reference.std > 0:
                normalized_shift = abs(current.mean - reference.mean) / reference.std
                return float(min(normalized_shift * 0.1, 1.0))  # Scale to PSI-like range
            return 0.0

        # Categorical: compare unique counts as proxy
        if current.num_unique and reference.num_unique:
            ratio_change = abs(current.num_unique - reference.num_unique) / max(
                reference.num_unique, 1
            )
            return float(min(ratio_change, 1.0))

        return 0.0

    def _get_severity(self, psi: float) -> DriftSeverity:
        """Get severity level from PSI value."""
        if psi >= self.psi_critical_threshold:
            return DriftSeverity.CRITICAL
        elif psi >= self.psi_warning_threshold:
            return DriftSeverity.WARNING
        return DriftSeverity.NONE

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _detect_feature_drift(
        self,
        current_series: pd.Series,
        reference_series: pd.Series,
        feature_name: str,
    ) -> FeatureDrift:
        """Detect drift for a single feature."""
        # Determine if numerical or categorical
        is_numerical = pd.api.types.is_numeric_dtype(current_series)

        if is_numerical:
            return self._detect_numerical_drift(current_series, reference_series, feature_name)
        else:
            return self._detect_categorical_drift(current_series, reference_series, feature_name)

    def _detect_numerical_drift(
        self,
        current: pd.Series,
        reference: pd.Series,
        feature_name: str,
    ) -> FeatureDrift:
        """Detect drift for numerical features using PSI."""
        # Remove NaN values
        current_clean = current.dropna()
        reference_clean = reference.dropna()

        if len(current_clean) == 0 or len(reference_clean) == 0:
            return FeatureDrift(
                feature_name=feature_name,
                psi=0.0,
                severity=DriftSeverity.NONE,
                reference_stats={},
                current_stats={},
                drift_type="numerical",
            )

        # Calculate PSI using binning
        psi = self._calculate_psi_numerical(current_clean, reference_clean)

        # Determine severity
        if psi >= self.psi_critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif psi >= self.psi_warning_threshold:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE

        # Collect statistics
        reference_stats = {
            "mean": float(reference_clean.mean()),
            "std": float(reference_clean.std()),
            "min": float(reference_clean.min()),
            "max": float(reference_clean.max()),
            "median": float(reference_clean.median()),}

        current_stats = {
            "mean": float(current_clean.mean()),
            "std": float(current_clean.std()),
            "min": float(current_clean.min()),
            "max": float(current_clean.max()),
            "median": float(current_clean.median()),}

        return FeatureDrift(
            feature_name=feature_name,
            psi=psi,
            severity=severity,
            reference_stats=reference_stats,
            current_stats=current_stats,
            drift_type="numerical",
        )

    def _detect_categorical_drift(
        self,
        current: pd.Series,
        reference: pd.Series,
        feature_name: str,
    ) -> FeatureDrift:
        """Detect drift for categorical features using PSI."""
        # Get value counts
        current_counts = current.value_counts(normalize=True)
        reference_counts = reference.value_counts(normalize=True)

        # Calculate PSI for categories
        psi = self._calculate_psi_categorical(current_counts, reference_counts)

        # Determine severity
        if psi >= self.psi_critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif psi >= self.psi_warning_threshold:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE

        # Collect statistics
        reference_stats = {
            "unique_count": len(reference_counts),
            "top_values": reference_counts.head(5).to_dict(),}

        current_stats = {
            "unique_count": len(current_counts),
            "top_values": current_counts.head(5).to_dict(),}

        return FeatureDrift(
            feature_name=feature_name,
            psi=psi,
            severity=severity,
            reference_stats=reference_stats,
            current_stats=current_stats,
            drift_type="categorical",
        )

    def _calculate_psi_numerical(
        self,
        current: pd.Series,
        reference: pd.Series,
        num_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index for numerical features.

        PSI measures the difference between two distributions:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.25: Moderate change (warning)
        - PSI > 0.25: Significant change (critical)
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=num_bins)

        # Ensure bins cover the range of both distributions
        min_val = min(current.min(), reference.min())
        max_val = max(current.max(), reference.max())
        bin_edges[0] = min_val - 1e-6
        bin_edges[-1] = max_val + 1e-6

        # Calculate proportions in each bin
        reference_props, _ = np.histogram(reference, bins=bin_edges)
        current_props, _ = np.histogram(current, bins=bin_edges)

        # Normalize to proportions
        reference_props = reference_props / len(reference)
        current_props = current_props / len(current)

        # Calculate PSI
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        reference_props = np.maximum(reference_props, epsilon)
        current_props = np.maximum(current_props, epsilon)

        psi = np.sum((current_props - reference_props) * np.log(current_props / reference_props))

        return float(psi)

    def _calculate_psi_categorical(
        self,
        current_dist: pd.Series,
        reference_dist: pd.Series,
    ) -> float:
        """Calculate PSI for categorical features."""
        # Get all categories
        all_categories = set(current_dist.index) | set(reference_dist.index)

        # Initialize PSI
        psi = 0.0
        epsilon = 1e-6

        for category in all_categories:
            current_prop = current_dist.get(category, 0.0)
            reference_prop = reference_dist.get(category, 0.0)

            # Add epsilon to avoid log(0)
            current_prop = max(current_prop, epsilon)
            reference_prop = max(reference_prop, epsilon)

            psi += (current_prop - reference_prop) * np.log(current_prop / reference_prop)

        return float(psi)


# =============================================================================
# Convenience Functions
# =============================================================================


def check_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    dataset: str,
    config: Settings | None = None,
    use_evidently: bool = False,
) -> DriftResult:
    """
    Convenience function to check for drift.

    Args:
        current_df: Current data
        reference_df: Reference data
        dataset: Dataset name
        config: Configuration object
        use_evidently: Use Evidently library for drift detection

    Returns:
        DriftResult
    """
    detector = DriftDetector(config)

    if use_evidently:
        return detector.detect_drift_with_evidently(current_df, reference_df, dataset)

    return detector.detect_drift(current_df, reference_df, dataset)
