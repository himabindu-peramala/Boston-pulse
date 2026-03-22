"""
Boston Pulse ML - Hyperparameter Tuner.

Optuna-based hyperparameter search for LightGBM.
Each trial is logged as a nested MLflow run.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd

from shared.schemas import TuningResult

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    parent_run_id: str,
) -> tuple[dict[str, Any], TuningResult]:
    """
    Run Optuna hyperparameter search.

    Args:
        train_df: training data (from random split)
        val_df: validation data (from random split)
        cfg: parsed crime_navigate_train.yaml
        parent_run_id: parent MLflow run ID for nested runs

    Returns:
        (best_params dict, TuningResult for XCom)
    """
    feature_cols = cfg["features"]["input_columns"]
    target_col = cfg["features"]["target_column"]
    cat_cols = cfg["features"]["categorical_columns"]
    ss = cfg["tuning"]["search_space"]

    X_tr, y_tr = train_df[feature_cols], train_df[target_col]
    X_v, y_v = val_df[feature_cols], val_df[target_col]

    lgb_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    lgb_v = lgb.Dataset(X_v, label=y_v, reference=lgb_tr, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": cfg["model"]["objective"],
            "metric": cfg["model"]["metric"],
            "verbosity": -1,
            "num_leaves": trial.suggest_int("num_leaves", *ss["num_leaves"]),
            "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *ss["min_child_samples"]),
            "subsample": trial.suggest_float("subsample", *ss["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"]),
        }
        n_est = trial.suggest_int("n_estimators", *ss["n_estimators"])

        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
            tags={"parent_run_id": parent_run_id},
        ):
            mlflow.log_params({**params, "n_estimators": n_est})

            model = lgb.train(
                params,
                lgb_tr,
                num_boost_round=n_est,
                valid_sets=[lgb_v],
                callbacks=[
                    lgb.early_stopping(cfg["training"]["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(-1),  # suppress logging
                ],
            )

            rmse = float(np.sqrt(((model.predict(X_v) - y_v.values) ** 2).mean()))
            mlflow.log_metric("val_rmse", rmse)

        return rmse

    study = optuna.create_study(direction=cfg["tuning"]["direction"])
    study.optimize(
        objective,
        n_trials=cfg["tuning"]["n_trials"],
        timeout=cfg["tuning"]["timeout_seconds"],
    )

    # Build best params dict with model objective and metric
    best_params = {
        **study.best_trial.params,
        "objective": cfg["model"]["objective"],
        "metric": cfg["model"]["metric"],
        "verbosity": -1,
    }

    logger.info(
        f"Tuning complete: {len(study.trials)} trials, " f"best_rmse={study.best_value:.4f}"
    )

    return best_params, TuningResult(
        best_params=best_params,
        best_val_rmse=float(study.best_value),
        n_trials=len(study.trials),
        mlflow_parent_run_id=parent_run_id,
        success=True,
    )


def get_default_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Get default parameters if tuning is skipped."""
    return {
        "objective": cfg["model"]["objective"],
        "metric": cfg["model"]["metric"],
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_estimators": 200,
    }
