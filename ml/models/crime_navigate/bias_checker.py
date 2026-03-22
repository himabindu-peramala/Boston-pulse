"""
Boston Pulse ML - Bias Checker.

Bias detection: does the model make equally good predictions across all districts?
Uses Fairlearn MetricFrame. Gates deployment on per-slice RMSE deviation.

If any slice has RMSE deviating more than max_slice_rmse_deviation from overall,
or RMSE > overall * max_slice_rmse_multiplier, the gate fails and deployment is blocked.

Small slices (< min_slice_size) are warned but do not fail the gate.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error

from shared.gcs_loader import GCSLoader
from shared.schemas import BiasResult

logger = logging.getLogger(__name__)


class BiasGateError(Exception):
    """Raised when bias gate fails."""

    pass


def check_bias(
    model: lgb.Booster,
    val_df: pd.DataFrame,
    execution_date: str,
    cfg: dict[str, Any],
    bucket: str,
    mlflow_run_id: str,
) -> BiasResult:
    """
    Check model for bias across slice dimensions.

    Args:
        model: trained LightGBM model
        val_df: validation data with district column
        execution_date: "YYYY-MM-DD"
        cfg: parsed crime_navigate_train.yaml
        bucket: GCS bucket name
        mlflow_run_id: MLflow run ID for logging

    Returns:
        BiasResult if gate passes

    Raises:
        BiasGateError if any significant slice fails the gate
    """
    feature_cols = cfg["features"]["input_columns"]
    target_col = cfg["features"]["target_column"]
    slice_dims = cfg["bias"]["slice_dimensions"]
    max_dev = cfg["bias"]["max_slice_rmse_deviation"]
    min_size = cfg["bias"]["min_slice_size"]
    max_mult = cfg["bias"]["max_slice_rmse_multiplier"]

    # Make predictions
    preds = model.predict(val_df[feature_cols])
    overall_rmse = float(np.sqrt(mean_squared_error(val_df[target_col], preds)))

    slice_results: dict[str, dict[str, Any]] = {}
    worst_slice, worst_dev = None, 0.0
    gate_failed = False

    for dim in slice_dims:
        if dim not in val_df.columns:
            logger.warning(f"Slice dimension '{dim}' not in val_df — skipping")
            continue

        # Use Fairlearn MetricFrame for slice-based evaluation
        frame = MetricFrame(
            metrics={"rmse": lambda y, yhat: float(np.sqrt(mean_squared_error(y, yhat)))},
            y_true=val_df[target_col],
            y_pred=preds,
            sensitive_features=val_df[dim].astype(str),
        )

        for sv, m in frame.by_group.iterrows():
            s_rmse = float(m["rmse"])
            dev = abs(s_rmse - overall_rmse) / overall_rmse if overall_rmse > 0 else 0.0
            cnt = int((val_df[dim].astype(str) == str(sv)).sum())

            # Determine if this slice fails the gate
            # Small slices are warned but don't fail
            is_small = cnt < min_size
            exceeds_deviation = dev > max_dev
            exceeds_multiplier = s_rmse > overall_rmse * max_mult
            fail = (not is_small) and (exceeds_deviation or exceeds_multiplier)

            slice_results[f"{dim}={sv}"] = {
                "rmse": round(s_rmse, 4),
                "deviation_pct": round(dev * 100, 2),
                "count": cnt,
                "passed": not fail,
                "is_small_slice": is_small,
            }

            if dev > worst_dev:
                worst_dev, worst_slice = dev, f"{dim}={sv}"

            if fail:
                gate_failed = True
                logger.warning(
                    f"Bias gate FAIL: {dim}={sv} rmse={s_rmse:.4f} "
                    f"deviation={dev:.1%} count={cnt}"
                )
            elif is_small and (exceeds_deviation or exceeds_multiplier):
                logger.warning(
                    f"Bias WARNING (small slice): {dim}={sv} rmse={s_rmse:.4f} "
                    f"deviation={dev:.1%} count={cnt}"
                )

    # Write bias report to GCS
    loader = GCSLoader(bucket)
    report_data = {
        "execution_date": execution_date,
        "overall_rmse": round(overall_rmse, 4),
        "gate_passed": not gate_failed,
        "slice_results": slice_results,
        "worst_slice": worst_slice,
        "worst_deviation_pct": round(worst_dev * 100, 2),
    }
    report_path = loader.write_json(
        report_data,
        prefix=cfg["data"]["bias_reports_prefix"],
        date=execution_date,
        filename="report.json",
    )

    # Log to MLflow
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(
            {
                "bias_overall_rmse": overall_rmse,
                "bias_worst_deviation_pct": worst_dev * 100,
                "bias_gate_passed": int(not gate_failed),
            }
        )

    if gate_failed:
        raise BiasGateError(
            f"Bias gate FAILED: worst={worst_slice} deviation={worst_dev:.1%}. "
            f"Report: {report_path}"
        )

    logger.info(
        f"Bias check PASSED: overall_rmse={overall_rmse:.4f}, " f"worst_deviation={worst_dev:.1%}"
    )

    return BiasResult(
        passed=True,
        overall_rmse=overall_rmse,
        slice_results=slice_results,
        worst_slice=worst_slice,
        worst_deviation_pct=worst_dev * 100,
        report_gcs_path=report_path,
    )


def get_slice_summary(bias_result: BiasResult) -> dict[str, Any]:
    """Get a summary of slice results for logging."""
    passed_slices = sum(1 for s in bias_result.slice_results.values() if s["passed"])
    failed_slices = sum(1 for s in bias_result.slice_results.values() if not s["passed"])
    return {
        "total_slices": len(bias_result.slice_results),
        "passed_slices": passed_slices,
        "failed_slices": failed_slices,
        "worst_slice": bias_result.worst_slice,
        "worst_deviation_pct": bias_result.worst_deviation_pct,
    }
