"""
Boston Pulse ML - Model Validator.

RMSE gate + SHAP analysis. Raises RuntimeError if gate fails — halts pipeline.

Gates:
- RMSE gate: val_rmse must be <= rmse_gate threshold
- Overfit gate: train_rmse / val_rmse must be <= overfit_ratio_gate

If either gate fails, the pipeline halts and the model is NOT pushed to registry.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import shap

from shared.schemas import TrainingResult, ValidationResult

logger = logging.getLogger(__name__)


class ValidationGateError(Exception):
    """Raised when validation gate fails."""

    pass


def validate_model(
    model: lgb.Booster,
    val_df: pd.DataFrame,
    training_result: TrainingResult,
    cfg: dict[str, Any],
    mlflow_run_id: str,
) -> ValidationResult:
    """
    Validate model against RMSE and overfit gates.

    Args:
        model: trained LightGBM model
        val_df: validation data
        training_result: result from trainer
        cfg: parsed crime_navigate_train.yaml
        mlflow_run_id: MLflow run ID for logging

    Returns:
        ValidationResult if gates pass

    Raises:
        ValidationGateError if any gate fails
    """
    feature_cols = cfg["features"]["input_columns"]
    target_col = cfg["features"]["target_column"]
    rmse_gate = cfg["validation"]["rmse_gate"]
    overfit_gate = cfg["validation"]["overfit_ratio_gate"]
    min_cells = cfg["validation"]["min_val_cells"]

    # Check minimum validation set size
    if val_df["h3_index"].nunique() < min_cells:
        raise ValidationGateError(
            f"Validation set has {val_df['h3_index'].nunique()} cells, "
            f"minimum required is {min_cells}"
        )

    # Compute validation RMSE
    preds = model.predict(val_df[feature_cols])
    rmse = float(np.sqrt(((preds - val_df[target_col].values) ** 2).mean()))

    # Compute overfit ratio (train_rmse / val_rmse)
    # If ratio > 1, model is overfitting (train error much lower than val)
    ratio = training_result.train_rmse / rmse if rmse > 0 else 0.0

    # Compute SHAP values for interpretability
    shap_path, feature_importance = _compute_shap(
        model, val_df[feature_cols], feature_cols, mlflow_run_id
    )

    # Log metrics to MLflow
    with mlflow.start_run(run_id=mlflow_run_id, nested=True):
        mlflow.log_metrics(
            {
                "rmse_val_final": rmse,
                "overfit_ratio": ratio,
                "validation_passed": int(rmse <= rmse_gate and ratio <= overfit_gate),
            }
        )

    # Check RMSE gate
    if rmse > rmse_gate:
        raise ValidationGateError(f"RMSE gate FAILED: {rmse:.4f} > {rmse_gate}. Pipeline halted.")

    # Check overfit gate
    if ratio > overfit_gate:
        raise ValidationGateError(
            f"Overfit gate FAILED: ratio={ratio:.2f} > {overfit_gate}. Pipeline halted."
        )

    logger.info(f"Validation PASSED: rmse={rmse:.4f}, overfit_ratio={ratio:.2f}")

    return ValidationResult(
        rmse_val=rmse,
        rmse_train=training_result.train_rmse,
        overfit_ratio=ratio,
        passed=True,
        shap_artifact_path=shap_path,
        feature_importance=feature_importance,
    )


def _compute_shap(
    model: lgb.Booster,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    run_id: str,
) -> tuple[str | None, dict[str, float]]:
    """
    Compute SHAP values and log artifacts to MLflow.

    Returns:
        (path to SHAP plot, feature importance dict)
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        tmp = Path(tempfile.mkdtemp())

        # Summary plot
        shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False)
        plot_path = str(tmp / "shap_summary.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=120)
        plt.close()

        # Feature importance (mean absolute SHAP value)
        importance = dict(
            sorted(
                {
                    c: float(np.abs(shap_values[:, i]).mean()) for i, c in enumerate(feature_cols)
                }.items(),
                key=lambda x: -x[1],
            )
        )

        json_path = str(tmp / "feature_importance.json")
        with open(json_path, "w") as f:
            json.dump(importance, f, indent=2)

        # Log to MLflow
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_artifact(plot_path, "shap")
            mlflow.log_artifact(json_path, "shap")

        logger.info(f"SHAP analysis complete. Top features: {list(importance.keys())[:5]}")
        return plot_path, importance

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None, {}
