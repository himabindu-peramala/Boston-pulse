"""
Boston Pulse ML - Bias Detection Utilities.

Fairlearn wrappers and slice report serialization helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE between true and predicted values."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_slice_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    metrics: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """
    Compute metrics per slice using Fairlearn MetricFrame.

    Args:
        y_true: True labels
        y_pred: Predicted values
        sensitive_features: Slice dimension values
        metrics: Dict of metric name -> callable (default: {"rmse": rmse_metric})

    Returns:
        DataFrame with metrics per slice
    """
    if metrics is None:
        metrics = {"rmse": rmse_metric}

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return frame.by_group


def analyze_slice_fairness(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    max_deviation: float = 0.20,
    min_slice_size: int = 50,
    max_multiplier: float = 2.0,
) -> dict[str, Any]:
    """
    Analyze fairness across slices.

    Args:
        y_true: True labels
        y_pred: Predicted values
        sensitive_features: Slice dimension values
        max_deviation: Maximum allowed deviation from overall RMSE
        min_slice_size: Minimum samples for a slice to be evaluated
        max_multiplier: Maximum allowed RMSE multiplier vs overall

    Returns:
        Dictionary with:
            - overall_rmse: Overall RMSE across all data
            - slice_results: Dict of slice -> metrics
            - worst_slice: Name of worst performing slice
            - worst_deviation: Deviation of worst slice
            - passed: Whether all slices pass thresholds
    """
    overall_rmse = rmse_metric(y_true, y_pred)

    slice_df = compute_slice_metrics(y_true, y_pred, sensitive_features)

    slice_results = {}
    worst_slice = None
    worst_deviation = 0.0
    gate_failed = False

    for slice_val, row in slice_df.iterrows():
        slice_rmse = row["rmse"]
        slice_count = int((sensitive_features == slice_val).sum())

        deviation = abs(slice_rmse - overall_rmse) / overall_rmse if overall_rmse > 0 else 0
        too_small = slice_count < min_slice_size
        hard_fail = slice_rmse > overall_rmse * max_multiplier
        soft_fail = deviation > max_deviation

        passed_slice = not hard_fail and not soft_fail

        slice_results[str(slice_val)] = {
            "rmse": round(float(slice_rmse), 4),
            "deviation_pct": round(deviation * 100, 2),
            "count": slice_count,
            "too_small": too_small,
            "passed": passed_slice,
        }

        if deviation > worst_deviation:
            worst_deviation = deviation
            worst_slice = str(slice_val)

        if not too_small and not passed_slice:
            gate_failed = True
            logger.warning(
                f"Slice {slice_val}: rmse={slice_rmse:.4f}, "
                f"deviation={deviation:.1%} (threshold={max_deviation:.0%})"
            )

    return {
        "overall_rmse": round(overall_rmse, 4),
        "slice_results": slice_results,
        "worst_slice": worst_slice,
        "worst_deviation": round(worst_deviation, 4),
        "passed": not gate_failed,
    }


def format_bias_report(
    analysis: dict[str, Any],
    execution_date: str,
    slice_dimension: str,
) -> dict[str, Any]:
    """
    Format bias analysis into a report structure.

    Args:
        analysis: Output from analyze_slice_fairness
        execution_date: Execution date
        slice_dimension: Name of the slice dimension

    Returns:
        Formatted report dictionary
    """
    return {
        "execution_date": execution_date,
        "slice_dimension": slice_dimension,
        "overall_rmse": analysis["overall_rmse"],
        "gate_passed": analysis["passed"],
        "slice_results": analysis["slice_results"],
        "worst_slice": analysis["worst_slice"],
        "worst_deviation_pct": round(analysis["worst_deviation"] * 100, 2),
        "n_slices": len(analysis["slice_results"]),
        "n_failed_slices": sum(
            1 for s in analysis["slice_results"].values() if not s["passed"] and not s["too_small"]
        ),
    }
