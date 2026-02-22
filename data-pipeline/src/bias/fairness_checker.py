"""
Boston Pulse - Fairness Checker

Evaluate fairness across data slices:
- Representation analysis (are groups fairly represented?)
- Outcome disparity (do groups have similar outcomes?)
- Statistical parity checks
- FairnessGate: Block pipeline if fairness violations exceed thresholds

Usage:
    checker = FairnessChecker(config)

    # Evaluate fairness for a dataset
    result = checker.evaluate_fairness(
        df=crime_df,
        dataset="crime",
        outcome_column="arrest_made"
    )

    # Check if fairness gate passes
    if not result.passes_fairness_gate:
        print(f"FAIRNESS GATE FAILED: {result.critical_violations}")
        raise FairnessViolationError(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd

from src.bias.data_slicer import DataSlice, DataSlicer
from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class FairnessSeverity(StrEnum):
    """Fairness violation severity."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class FairnessMetric(StrEnum):
    """Fairness metric type."""

    REPRESENTATION = "representation"
    OUTCOME_DISPARITY = "outcome_disparity"
    STATISTICAL_PARITY = "statistical_parity"


@dataclass
class FairnessViolation:
    """Individual fairness violation."""

    metric: FairnessMetric
    severity: FairnessSeverity
    dimension: str
    slice_value: Any
    message: str
    expected: float
    actual: float
    disparity: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FairnessResult:
    """Result of fairness evaluation."""

    dataset: str
    evaluated_at: datetime
    slices_evaluated: int
    violations: list[FairnessViolation] = field(default_factory=list)
    fairness_gate_enabled: bool = False

    @property
    def has_violations(self) -> bool:
        """Check if any violations were found."""
        return len(self.violations) > 0

    @property
    def has_critical_violations(self) -> bool:
        """Check if any critical violations were found."""
        return any(v.severity == FairnessSeverity.CRITICAL for v in self.violations)

    @property
    def critical_violations(self) -> list[FairnessViolation]:
        """Get list of critical violations."""
        return [v for v in self.violations if v.severity == FairnessSeverity.CRITICAL]

    @property
    def warning_violations(self) -> list[FairnessViolation]:
        """Get list of warning violations."""
        return [v for v in self.violations if v.severity == FairnessSeverity.WARNING]

    @property
    def passes_fairness_gate(self) -> bool:
        """Check if fairness gate passes."""
        if not self.fairness_gate_enabled:
            return True
        return not self.has_critical_violations


class FairnessChecker:
    """
    Evaluate fairness across data slices.

    Checks for:
    - Representation fairness: Are groups represented proportionally?
    - Outcome disparity: Do groups have similar outcomes?
    - Statistical parity: Are outcome rates similar across groups?
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize fairness checker.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.slicer = DataSlicer(config)

        # Thresholds from config
        self.representation_warning = self.config.fairness.thresholds.representation.warning
        self.representation_critical = self.config.fairness.thresholds.representation.critical
        self.outcome_warning = self.config.fairness.thresholds.outcome_disparity.warning
        self.outcome_critical = self.config.fairness.thresholds.outcome_disparity.critical
        self.gate_enabled = self.config.fairness.gate_enabled

    def evaluate_fairness(
        self,
        df: pd.DataFrame,
        dataset: str,
        outcome_column: str | None = None,
        dimensions: list[str] | None = None,
    ) -> FairnessResult:
        """
        Evaluate fairness across data slices.

        Args:
            df: Source DataFrame
            dataset: Dataset name
            outcome_column: Column containing binary outcome (for outcome disparity)
            dimensions: Dimensions to evaluate (uses defaults if not provided)

        Returns:
            FairnessResult with any violations found

        Example:
            result = checker.evaluate_fairness(
                crime_df,
                "crime",
                outcome_column="arrest_made"
            )
        """
        logger.info(
            f"Evaluating fairness for {dataset}",
            extra={"dataset": dataset, "rows": len(df)},
        )

        result = FairnessResult(
            dataset=dataset,
            evaluated_at=datetime.now(UTC),
            slices_evaluated=0,
            fairness_gate_enabled=self.gate_enabled,
        )

        # Get slices to evaluate
        if dimensions is None:
            slice_dict = self.slicer.get_default_slices(df, dataset)
        else:
            slice_dict = {}
            for dim in dimensions:
                if dim in df.columns:
                    slice_dict[dim] = self.slicer.slice_by_category(df, dim)

        # Evaluate each dimension
        for dimension, slices in slice_dict.items():
            result.slices_evaluated += len(slices)

            # 1. Check representation fairness
            result.violations.extend(self._check_representation_fairness(slices, dimension))

            # 2. Check outcome disparity (if outcome column provided)
            if outcome_column and outcome_column in df.columns:
                result.violations.extend(
                    self._check_outcome_disparity(slices, dimension, outcome_column)
                )

        logger.info(
            f"Fairness evaluation complete for {dataset}: "
            f"{len(result.violations)} violations "
            f"({len(result.critical_violations)} critical)",
            extra={
                "dataset": dataset,
                "violations": len(result.violations),
                "critical": len(result.critical_violations),
                "passes_gate": result.passes_fairness_gate,
            },
        )

        return result

    def evaluate_model_fairness(
        self,
        df: pd.DataFrame,
        predictions_column: str,
        protected_attributes: list[str],
        dataset: str,
    ) -> FairnessResult:
        """
        Evaluate fairness of model predictions.

        Args:
            df: DataFrame with predictions
            predictions_column: Column with model predictions (0/1)
            protected_attributes: List of protected attribute columns
            dataset: Dataset name

        Returns:
            FairnessResult
        """
        # Use predictions as the outcome column
        return self.evaluate_fairness(
            df,
            dataset,
            outcome_column=predictions_column,
            dimensions=protected_attributes,
        )

    # =========================================================================
    # Private Checking Methods
    # =========================================================================

    def _check_representation_fairness(
        self,
        slices: list[DataSlice],
        dimension: str,
    ) -> list[FairnessViolation]:
        """
        Check if groups are fairly represented.

        Representation fairness: Each group should be represented proportionally.
        Significant deviations may indicate sampling bias or data quality issues.
        """
        violations = []

        if len(slices) == 0:
            return violations

        # Calculate expected representation (uniform distribution)
        expected_percentage = 100.0 / len(slices)

        for slice_obj in slices:
            # Calculate disparity (deviation from expected)
            disparity = abs(slice_obj.percentage - expected_percentage) / expected_percentage

            # Check against thresholds
            if disparity >= self.representation_critical:
                severity = FairnessSeverity.CRITICAL
            elif disparity >= self.representation_warning:
                severity = FairnessSeverity.WARNING
            else:
                continue  # No violation

            violations.append(
                FairnessViolation(
                    metric=FairnessMetric.REPRESENTATION,
                    severity=severity,
                    dimension=dimension,
                    slice_value=slice_obj.value,
                    message=(
                        f"{dimension}={slice_obj.value} has {slice_obj.percentage:.1f}% "
                        f"representation (expected {expected_percentage:.1f}%, "
                        f"disparity: {disparity:.1%})"
                    ),
                    expected=expected_percentage,
                    actual=slice_obj.percentage,
                    disparity=disparity,
                    details={
                        "slice_size": slice_obj.size,
                        "total_slices": len(slices),
                    },
                )
            )

        return violations

    def _check_outcome_disparity(
        self,
        slices: list[DataSlice],
        dimension: str,
        outcome_column: str,
    ) -> list[FairnessViolation]:
        """
        Check for outcome disparity across slices using Fairlearn's MetricFrame.

        Outcome disparity: Different groups should have similar outcome rates.
        Large disparities may indicate bias in the data or process.

        Uses Fairlearn's MetricFrame and demographic_parity_difference for
        statistically grounded disparity measurement.
        """
        from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate

        violations = []

        if len(slices) == 0:
            return violations

        # Rebuild full DataFrame from slices
        all_data = pd.concat([s.data for s in slices])

        if outcome_column not in all_data.columns:
            return violations

        # Need at least 2 slices to compare
        if all_data[dimension].nunique() < 2:
            return violations

        y = all_data[outcome_column].astype(int)
        sensitive = all_data[dimension]

        # -------------------------------------------------------------------------
        # Fairlearn MetricFrame: per-group selection rates
        # -------------------------------------------------------------------------
        try:
            mf = MetricFrame(
                metrics={"selection_rate": selection_rate},
                y_true=y,
                y_pred=y,  # Data-level check (no model), so y_true == y_pred
                sensitive_features=sensitive,
            )

            logger.info(
                f"Fairlearn MetricFrame for {dimension}:\n{mf.by_group}",
                extra={"dataset": dimension, "metric": "selection_rate"},
            )

            # Overall demographic parity difference (max - min selection rate)
            dpd = demographic_parity_difference(
                y_true=y,
                y_pred=y,
                sensitive_features=sensitive,
            )

            logger.info(
                f"Demographic parity difference for {dimension}: {dpd:.4f}",
                extra={"dataset": dimension, "dpd": dpd},
            )

        except Exception as e:
            logger.warning(f"Fairlearn MetricFrame failed for {dimension}: {e}. Falling back to manual calculation.")
            mf = None
            dpd = None

        # -------------------------------------------------------------------------
        # Per-slice violation detection
        # -------------------------------------------------------------------------
        overall_rate = y.mean()

        for slice_obj in slices:
            if outcome_column not in slice_obj.data.columns or len(slice_obj.data) == 0:
                continue

            slice_y = slice_obj.data[outcome_column].astype(int)
            outcome_rate = slice_y.mean()

            # Use Fairlearn's per-group rate if available, else compute manually
            if mf is not None and slice_obj.value in mf.by_group.index:
                outcome_rate = mf.by_group.loc[slice_obj.value, "selection_rate"]

            # Calculate disparity from overall rate
            if overall_rate > 0:
                disparity = abs(outcome_rate - overall_rate) / overall_rate
            else:
                disparity = abs(outcome_rate)

            # Classify severity
            if disparity >= self.outcome_critical:
                severity = FairnessSeverity.CRITICAL
            elif disparity >= self.outcome_warning:
                severity = FairnessSeverity.WARNING
            else:
                continue  # No violation

            violations.append(
                FairnessViolation(
                    metric=FairnessMetric.OUTCOME_DISPARITY,
                    severity=severity,
                    dimension=dimension,
                    slice_value=slice_obj.value,
                    message=(
                        f"{dimension}={slice_obj.value} has {outcome_rate:.1%} outcome rate "
                        f"(overall: {overall_rate:.1%}, disparity: {disparity:.1%})"
                    ),
                    expected=overall_rate,
                    actual=outcome_rate,
                    disparity=disparity,
                    details={
                        "slice_size": slice_obj.size,
                        "outcome_column": outcome_column,
                        "demographic_parity_difference": round(dpd, 4) if dpd is not None else None,
                        "fairlearn_used": mf is not None,
                    },
                )
            )

        return violations

    def create_fairness_report(self, result: FairnessResult) -> str:
        """
        Create a human-readable fairness report.

        Args:
            result: FairnessResult to summarize

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"FAIRNESS EVALUATION REPORT - {result.dataset}")
        report.append("=" * 80)
        report.append(f"Evaluated at: {result.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Slices evaluated: {result.slices_evaluated}")
        report.append(f"Fairness gate enabled: {result.fairness_gate_enabled}")
        report.append(f"Gate status: {'PASS' if result.passes_fairness_gate else 'FAIL'}")
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total violations: {len(result.violations)}")
        report.append(f"Critical violations: {len(result.critical_violations)}")
        report.append(f"Warning violations: {len(result.warning_violations)}")
        report.append("")

        # Violations by severity
        if result.critical_violations:
            report.append("CRITICAL VIOLATIONS")
            report.append("-" * 80)
            for v in result.critical_violations:
                report.append(f"  [{v.metric.value}] {v.message}")
            report.append("")

        if result.warning_violations:
            report.append("WARNING VIOLATIONS")
            report.append("-" * 80)
            for v in result.warning_violations:
                report.append(f"  [{v.metric.value}] {v.message}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)


# =============================================================================
# Exception Classes
# =============================================================================


class FairnessViolationError(Exception):
    """Raised when fairness gate fails."""

    def __init__(self, result: FairnessResult):
        self.result = result
        violations = "\n".join([f"  - {v.message}" for v in result.critical_violations])
        super().__init__(f"Fairness gate failed for {result.dataset}:\n{violations}")


# =============================================================================
# Convenience Functions
# =============================================================================


def check_fairness(
    df: pd.DataFrame,
    dataset: str,
    outcome_column: str | None = None,
    config: Settings | None = None,
) -> FairnessResult:
    """
    Convenience function to check fairness.

    Args:
        df: Source DataFrame
        dataset: Dataset name
        outcome_column: Column with binary outcome
        config: Configuration object

    Returns:
        FairnessResult

    Raises:
        FairnessViolationError: If fairness gate is enabled and fails
    """
    checker = FairnessChecker(config)
    result = checker.evaluate_fairness(df, dataset, outcome_column)

    # Raise error if gate fails
    if not result.passes_fairness_gate:
        raise FairnessViolationError(result)

    return result
