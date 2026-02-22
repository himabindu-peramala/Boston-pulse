"""
Boston Pulse - Bias Mitigation

Mitigation strategies for detected fairness violations:
- Reweighting: Adjust sample weights to correct representation bias
- Stratified Sampling: Resample to balance underrepresented groups
- Threshold Adjustment: Per-group decision thresholds (model-level)

Usage:
    from src.bias.bias_mitigator import BiasMitigator

    mitigator = BiasMitigator()
    result = mitigator.mitigate(df, fairness_result, dimension="district")

    # Access mitigated DataFrame and report
    mitigated_df = result.mitigated_df
    print(result.report())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from src.bias.fairness_checker import FairnessResult
from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)

class MitigationStrategy(StrEnum):
    REWEIGHTING         = "reweighting"
    STRATIFIED_SAMPLING = "stratified_sampling"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"


@dataclass
class MitigationAction:
    """Record of a single mitigation action taken."""
    strategy: MitigationStrategy
    dimension: str
    slice_value: Any
    before_disparity: float
    after_disparity: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.after_disparity < self.before_disparity

    @property
    def improvement_pct(self) -> float:
        if self.before_disparity == 0:
            return 0.0
        return (self.before_disparity - self.after_disparity) / self.before_disparity * 100


@dataclass
class MitigationResult:
    """Result of mitigation process."""
    dataset: str
    strategy: MitigationStrategy
    dimension: str
    original_df: pd.DataFrame
    mitigated_df: pd.DataFrame
    actions: list[MitigationAction] = field(default_factory=list)
    mitigated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tradeoffs: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_before(self) -> int:
        return len(self.original_df)

    @property
    def rows_after(self) -> int:
        return len(self.mitigated_df)

    @property
    def total_improved(self) -> int:
        return sum(1 for a in self.actions if a.improved)

    def report(self) -> str:
        lines = [
            "=" * 80,
            f"BIAS MITIGATION REPORT - {self.dataset}",
            "=" * 80,
            f"Mitigated at  : {self.mitigated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Strategy      : {self.strategy.value}",
            f"Dimension     : {self.dimension}",
            f"Rows before   : {self.rows_before}",
            f"Rows after    : {self.rows_after}",
            f"Slices improved: {self.total_improved} / {len(self.actions)}",
            "",
            "ACTIONS TAKEN",
            "-" * 80,
        ]
        for a in self.actions:
            status = "✅ improved" if a.improved else "⚠️  no change"
            lines.append(
                f"  [{status}] {self.dimension}={a.slice_value} | "
                f"disparity: {a.before_disparity:.3f} → {a.after_disparity:.3f} "
                f"({a.improvement_pct:+.1f}%)"
            )

        if self.tradeoffs:
            lines += ["", "TRADE-OFFS", "-" * 80]
            for k, v in self.tradeoffs.items():
                lines.append(f"  {k}: {v}")

        lines.append("=" * 80)
        return "\n".join(lines)


# =============================================================================
# Mitigator
# =============================================================================

class BiasMitigator:
    """
    Apply bias mitigation strategies to correct fairness violations.

    Supports:
    - Reweighting        : Assigns inverse-frequency weights per group
    - Stratified Sampling: Over/under-samples to balance group sizes
    - Threshold Adjustment: Adjusts per-group decision thresholds
    """

    def __init__(self, config: Settings | None = None):
        self.config = config or get_config()

    # -------------------------------------------------------------------------
    # Public Entry Point
    # -------------------------------------------------------------------------

    def mitigate(
        self,
        df: pd.DataFrame,
        fairness_result: FairnessResult,
        dimension: str,
        strategy: MitigationStrategy = MitigationStrategy.REWEIGHTING,
        outcome_column: str | None = None,
    ) -> MitigationResult:
        """
        Apply mitigation for detected violations on a given dimension.

        Args:
            df             : Source DataFrame
            fairness_result: Result from FairnessChecker.evaluate_fairness()
            dimension      : Column to mitigate across (e.g. "district")
            strategy       : Which mitigation strategy to apply
            outcome_column : Required for threshold adjustment strategy

        Returns:
            MitigationResult with mitigated DataFrame and full report
        """
        logger.info(
            f"Starting mitigation for {fairness_result.dataset} "
            f"| strategy={strategy} | dimension={dimension}"
        )

        if strategy == MitigationStrategy.REWEIGHTING:
            return self._apply_reweighting(df, fairness_result, dimension)

        elif strategy == MitigationStrategy.STRATIFIED_SAMPLING:
            return self._apply_stratified_sampling(df, fairness_result, dimension)

        elif strategy == MitigationStrategy.THRESHOLD_ADJUSTMENT:
            if outcome_column is None:
                raise ValueError("outcome_column is required for threshold adjustment strategy")
            return self._apply_threshold_adjustment(df, fairness_result, dimension, outcome_column)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # -------------------------------------------------------------------------
    # Strategy 1: Reweighting
    # -------------------------------------------------------------------------

    def compute_sample_weights(self, df: pd.DataFrame, dimension: str) -> pd.Series:
        """
        Compute inverse-frequency weights to correct representation bias.

        Groups with fewer records get higher weights, over-represented
        groups get lower weights. Weights sum to len(df).

        Args:
            df        : Source DataFrame
            dimension : Column to balance across

        Returns:
            Series of float weights aligned to df.index
        """
        counts = df[dimension].value_counts()
        n_groups = len(counts)
        expected_per_group = len(df) / n_groups

        # Inverse frequency: expected / actual count
        weights = df[dimension].map(lambda x: expected_per_group / counts[x])

        # Normalize so weights sum to len(df)
        weights = weights / weights.sum() * len(df)

        logger.info(
            f"Computed reweighting for {dimension}:\n"
            + "\n".join(
                f"  {g}: count={counts[g]}, weight={expected_per_group/counts[g]:.3f}"
                for g in counts.index
            )
        )

        return weights

    def _apply_reweighting(
        self,
        df: pd.DataFrame,
        fairness_result: FairnessResult,
        dimension: str,
    ) -> MitigationResult:
        """
        Add a `sample_weight` column to the DataFrame.
        Does not change row count — weights are used downstream
        in model training or feature aggregation.
        """
        if dimension not in df.columns:
            raise ValueError(f"Column '{dimension}' not found in DataFrame")

        mitigated_df = df.copy()
        weights = self.compute_sample_weights(df, dimension)
        mitigated_df["sample_weight"] = weights

        # Build actions from violations
        actions = self._build_reweighting_actions(df, mitigated_df, fairness_result, dimension)

        tradeoffs = {
            "row_count_change"  : "None (reweighting preserves all rows)",
            "data_loss"         : "None",
            "weight_range"      : f"{weights.min():.3f} – {weights.max():.3f}",
            "recommendation"    : (
                "Pass 'sample_weight' column to model training. "
                "Verify that downstream aggregations use weighted means."
            ),
        }

        result = MitigationResult(
            dataset=fairness_result.dataset,
            strategy=MitigationStrategy.REWEIGHTING,
            dimension=dimension,
            original_df=df,
            mitigated_df=mitigated_df,
            actions=actions,
            tradeoffs=tradeoffs,
        )

        logger.info("Reweighting complete. Weight column added: 'sample_weight'")
        return result

    # -------------------------------------------------------------------------
    # Strategy 2: Stratified Sampling
    # -------------------------------------------------------------------------

    def _apply_stratified_sampling(
        self,
        df: pd.DataFrame,
        fairness_result: FairnessResult,
        dimension: str,
        random_state: int = 42,
    ) -> MitigationResult:
        """
        Resample the DataFrame so each group has equal representation.

        - Over-represented groups are downsampled
        - Under-represented groups are oversampled (with replacement)

        Target size per group = median group size across all groups.
        """
        if dimension not in df.columns:
            raise ValueError(f"Column '{dimension}' not found in DataFrame")

        counts = df[dimension].value_counts()
        target_size = int(counts.median())  # Use median as target per group

        logger.info(
            f"Stratified sampling: target {target_size} rows per {dimension} group"
        )

        resampled_groups = []
        actions = []

        for group_val, group_df in df.groupby(dimension):
            before_size = len(group_df)
            expected_pct = 100.0 / len(counts)
            before_disparity = abs(
                (before_size / len(df) * 100) - expected_pct
            ) / expected_pct

            if before_size > target_size:
                # Downsample (without replacement)
                resampled = group_df.sample(n=target_size, random_state=random_state)
                action_detail = f"downsampled {before_size} → {target_size}"
            elif before_size < target_size:
                # Oversample (with replacement for deficit)
                deficit = target_size - before_size
                extras = group_df.sample(n=deficit, replace=True, random_state=random_state)
                resampled = pd.concat([group_df, extras])
                action_detail = f"oversampled {before_size} → {target_size}"
            else:
                resampled = group_df
                action_detail = "unchanged"

            resampled_groups.append(resampled)

            # Compute after disparity
            total_after = target_size * len(counts)
            after_pct = target_size / total_after * 100
            after_disparity = abs(after_pct - expected_pct) / expected_pct

            actions.append(MitigationAction(
                strategy=MitigationStrategy.STRATIFIED_SAMPLING,
                dimension=dimension,
                slice_value=group_val,
                before_disparity=before_disparity,
                after_disparity=after_disparity,
                details={"action": action_detail, "target_size": target_size},
            ))

        mitigated_df = pd.concat(resampled_groups).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

        tradeoffs = {
            "row_count_change": f"{len(df)} → {len(mitigated_df)}",
            "data_loss"       : f"{max(0, len(df) - len(mitigated_df))} rows removed from over-represented groups",
            "synthetic_rows"  : f"{max(0, len(mitigated_df) - len(df))} rows added via oversampling",
            "recommendation"  : (
                "Oversampled rows are duplicates. If using for model training, "
                "consider adding slight noise to avoid overfitting on minority groups."
            ),
        }

        result = MitigationResult(
            dataset=fairness_result.dataset,
            strategy=MitigationStrategy.STRATIFIED_SAMPLING,
            dimension=dimension,
            original_df=df,
            mitigated_df=mitigated_df,
            actions=actions,
            tradeoffs=tradeoffs,
        )

        logger.info(
            f"Stratified sampling complete: {len(df)} → {len(mitigated_df)} rows"
        )
        return result

    # -------------------------------------------------------------------------
    # Strategy 3: Threshold Adjustment
    # -------------------------------------------------------------------------

    def _apply_threshold_adjustment(
        self,
        df: pd.DataFrame,
        fairness_result: FairnessResult,
        dimension: str,
        outcome_column: str,
        base_threshold: float = 0.5,
    ) -> MitigationResult:
        """
        Adjust per-group decision thresholds to equalize outcome rates.

        For groups with outcome rates below the overall rate, the threshold
        is lowered to increase positive predictions. For groups above, it
        is raised. Thresholds are stored in a column `adjusted_threshold`.

        Note: Requires a probability score column. If outcome_column is
        binary (0/1), thresholds are illustrative only.
        """
        if dimension not in df.columns:
            raise ValueError(f"Column '{dimension}' not found in DataFrame")
        if outcome_column not in df.columns:
            raise ValueError(f"Column '{outcome_column}' not found in DataFrame")

        mitigated_df = df.copy()
        overall_rate = df[outcome_column].mean()
        group_rates = df.groupby(dimension)[outcome_column].mean()
        actions = []

        thresholds = {}

        for group_val, group_rate in group_rates.items():
            # Adjust threshold inversely proportional to outcome rate gap
            rate_gap = overall_rate - group_rate
            adjusted_threshold = base_threshold - (rate_gap * 0.5)
            adjusted_threshold = float(np.clip(adjusted_threshold, 0.1, 0.9))
            thresholds[group_val] = adjusted_threshold

            before_disparity = abs(group_rate - overall_rate) / overall_rate if overall_rate > 0 else 0

            logger.info(
                f"  {dimension}={group_val}: rate={group_rate:.3f}, "
                f"threshold {base_threshold:.2f} → {adjusted_threshold:.2f}"
            )

            actions.append(MitigationAction(
                strategy=MitigationStrategy.THRESHOLD_ADJUSTMENT,
                dimension=dimension,
                slice_value=group_val,
                before_disparity=before_disparity,
                after_disparity=0.0,  # Theoretical — actual improvement depends on model scores
                details={
                    "original_threshold": base_threshold,
                    "adjusted_threshold": adjusted_threshold,
                    "group_outcome_rate": group_rate,
                    "overall_outcome_rate": overall_rate,
                },
            ))

        # Store adjusted threshold per row
        mitigated_df["adjusted_threshold"] = mitigated_df[dimension].map(thresholds)

        tradeoffs = {
            "row_count_change": "None (threshold adjustment preserves all rows)",
            "data_loss"       : "None",
            "threshold_range" : f"{min(thresholds.values()):.2f} – {max(thresholds.values()):.2f}",
            "recommendation"  : (
                "Apply 'adjusted_threshold' per group when converting model "
                "probability scores to binary predictions. Overall accuracy "
                "may decrease slightly to gain fairness across groups."
            ),
        }

        result = MitigationResult(
            dataset=fairness_result.dataset,
            strategy=MitigationStrategy.THRESHOLD_ADJUSTMENT,
            dimension=dimension,
            original_df=df,
            mitigated_df=mitigated_df,
            actions=actions,
            tradeoffs=tradeoffs,
        )

        logger.info("Threshold adjustment complete.")
        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_reweighting_actions(
        self,
        original_df: pd.DataFrame,
        _mitigated_df: pd.DataFrame,
        _fairness_result: FairnessResult,
        dimension: str,
    ) -> list[MitigationAction]:
        """Build MitigationAction list for reweighting strategy."""
        actions = []
        counts = original_df[dimension].value_counts()
        n_groups = len(counts)
        expected_pct = 100.0 / n_groups

        for group_val, count in counts.items():
            before_pct = count / len(original_df) * 100
            before_disparity = abs(before_pct - expected_pct) / expected_pct

            weight = (len(original_df) / n_groups) / count
            # After reweighting, effective disparity is near 0
            after_disparity = 0.0

            actions.append(MitigationAction(
                strategy=MitigationStrategy.REWEIGHTING,
                dimension=dimension,
                slice_value=group_val,
                before_disparity=before_disparity,
                after_disparity=after_disparity,
                details={
                    "count"         : int(count),
                    "weight"        : round(weight, 4),
                    "original_pct"  : round(before_pct, 2),
                    "expected_pct"  : round(expected_pct, 2),
                },
            ))

        return actions


# =============================================================================
# Convenience Function
# =============================================================================

def mitigate_bias(
    df: pd.DataFrame,
    fairness_result: FairnessResult,
    dimension: str,
    strategy: MitigationStrategy = MitigationStrategy.REWEIGHTING,
    outcome_column: str | None = None,
    config: Settings | None = None,
) -> MitigationResult:
    """
    Convenience function to apply bias mitigation.

    Args:
        df             : Source DataFrame
        fairness_result: Output from FairnessChecker.evaluate_fairness()
        dimension      : Column with representational bias (e.g. "district")
        strategy       : Mitigation strategy to apply
        outcome_column : Needed only for threshold_adjustment strategy
        config         : Optional config override

    Returns:
        MitigationResult

    Example:
        result = mitigate_bias(
            df=crime_df,
            fairness_result=fairness_result,
            dimension="district",
            strategy=MitigationStrategy.REWEIGHTING,
        )
        print(result.report())
        mitigated_df = result.mitigated_df
    """
    mitigator = BiasMitigator(config)
    return mitigator.mitigate(df, fairness_result, dimension, strategy, outcome_column)
