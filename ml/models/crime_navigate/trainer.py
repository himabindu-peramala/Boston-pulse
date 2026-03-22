"""
Boston Pulse ML - Model Trainer.

Train LightGBM model with best hyperparameters from tuner.

Key design decisions:
- Random 80/20 split — this is a cross-sectional model, NOT time-series
- regression_l1 objective — robust to outlier cells with extreme danger rates
- Early stopping on validation set
- All params/metrics logged to MLflow
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.schemas import TrainingResult

logger = logging.getLogger(__name__)


def random_split(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random 80/20 split across cells.

    Cross-sectional model — random split is correct.
    Temporal split would be WRONG here because:
    - danger_rate is a structural property, not a time-series value
    - We want the model to generalise across cells, not across time
    """
    return train_test_split(
        df,
        test_size=cfg["training"]["val_fraction"],
        random_state=cfg["training"]["random_seed"],
    )


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    best_params: dict[str, Any],
    cfg: dict[str, Any],
    mlflow_run_id: str,
) -> tuple[lgb.Booster, str, TrainingResult]:
    """
    Train LightGBM model with best hyperparameters.

    Args:
        train_df: training data (from random split)
        val_df: validation data (from random split)
        best_params: best hyperparameters from tuner
        cfg: parsed crime_navigate_train.yaml
        mlflow_run_id: parent MLflow run ID

    Returns:
        (trained model, local model path, TrainingResult for XCom)
    """
    feature_cols = cfg["features"]["input_columns"]
    target_col = cfg["features"]["target_column"]
    cat_cols = cfg["features"]["categorical_columns"]

    X_tr, y_tr = train_df[feature_cols], train_df[target_col]
    X_v, y_v = val_df[feature_cols], val_df[target_col]

    # Pop n_estimators from params (used as num_boost_round)
    params = best_params.copy()
    n_est = params.pop("n_estimators", 300)

    lgb_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    lgb_v = lgb.Dataset(X_v, label=y_v, reference=lgb_tr)

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_params(
            {
                **params,
                "n_estimators": n_est,
                "n_train_rows": len(train_df),
                "n_val_rows": len(val_df),
                "label": target_col,
                "split_type": "random_80_20",
            }
        )
        mlflow.log_param("features", json.dumps(feature_cols))

        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=n_est,
            valid_sets=[lgb_v],
            callbacks=[
                lgb.early_stopping(cfg["training"]["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(cfg["training"]["verbose_eval"]),
            ],
        )

        # Compute RMSE on train and validation sets
        tr_rmse = float(np.sqrt(((model.predict(X_tr) - y_tr.values) ** 2).mean()))
        v_rmse = float(np.sqrt(((model.predict(X_v) - y_v.values) ** 2).mean()))

        mlflow.log_metrics(
            {
                "rmse_train": tr_rmse,
                "rmse_val": v_rmse,
                "best_iteration": model.best_iteration,
            }
        )
        mlflow.lightgbm.log_model(model, "model")

    # Save model locally for subsequent tasks
    path = str(Path(tempfile.mkdtemp()) / "model.lgb")
    model.save_model(path)

    logger.info(
        f"Model trained: train_rmse={tr_rmse:.4f}, val_rmse={v_rmse:.4f}, "
        f"best_iter={model.best_iteration}"
    )

    return (
        model,
        path,
        TrainingResult(
            model_path=path,
            train_rmse=tr_rmse,
            val_rmse=v_rmse,
            best_iteration=model.best_iteration,
            n_features=len(feature_cols),
            n_train_rows=len(train_df),
            n_val_rows=len(val_df),
            mlflow_run_id=mlflow_run_id,
            success=True,
        ),
    )


def load_model(model_path: str) -> lgb.Booster:
    """Load a saved LightGBM model."""
    return lgb.Booster(model_file=model_path)


def predict(model: lgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Make predictions with the model."""
    preds = model.predict(df[feature_cols])
    return np.maximum(preds, 0.0)  # danger_rate cannot be negative
