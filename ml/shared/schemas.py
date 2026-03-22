"""
Boston Pulse ML - Result schemas.

Pydantic-style dataclasses for all pipeline result objects.
These are the XCom payloads that flow between DAG tasks.
Typed, serialisable, self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureLoadResult:
    """Result of loading features from GCS."""

    rows: int
    h3_cells: int
    columns: list[str]
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "h3_cells": self.h3_cells,
            "columns": self.columns,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TargetBuildResult:
    """Result of building training targets (danger_rate label)."""

    rows: int
    h3_cells: int
    mean_danger_rate: float
    zero_rate_cells: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "h3_cells": self.h3_cells,
            "mean_danger_rate": self.mean_danger_rate,
            "zero_rate_cells": self.zero_rate_cells,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: dict[str, Any]
    best_val_rmse: float
    n_trials: int
    mlflow_parent_run_id: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_val_rmse": self.best_val_rmse,
            "n_trials": self.n_trials,
            "mlflow_parent_run_id": self.mlflow_parent_run_id,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TrainingResult:
    """Result of model training."""

    model_path: str
    train_rmse: float
    val_rmse: float
    best_iteration: int
    n_features: int
    n_train_rows: int
    n_val_rows: int
    mlflow_run_id: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "train_rmse": self.train_rmse,
            "val_rmse": self.val_rmse,
            "best_iteration": self.best_iteration,
            "n_features": self.n_features,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "mlflow_run_id": self.mlflow_run_id,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ValidationResult:
    """Result of model validation (RMSE gate + SHAP)."""

    rmse_val: float
    rmse_train: float
    overfit_ratio: float
    passed: bool
    shap_artifact_path: str | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmse_val": self.rmse_val,
            "rmse_train": self.rmse_train,
            "overfit_ratio": self.overfit_ratio,
            "passed": self.passed,
            "shap_artifact_path": self.shap_artifact_path,
            "feature_importance": self.feature_importance,
            "error": self.error,
        }


@dataclass
class BiasResult:
    """Result of bias detection."""

    passed: bool
    overall_rmse: float
    slice_results: dict[str, dict[str, Any]]
    worst_slice: str | None
    worst_deviation_pct: float
    report_gcs_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_rmse": self.overall_rmse,
            "slice_results": self.slice_results,
            "worst_slice": self.worst_slice,
            "worst_deviation_pct": self.worst_deviation_pct,
            "report_gcs_path": self.report_gcs_path,
            "error": self.error,
        }


@dataclass
class ScoringResult:
    """Result of model scoring."""

    rows_scored: int
    h3_cells: int
    output_gcs_path: str
    score_distribution: dict[str, int]
    model_version: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_scored": self.rows_scored,
            "h3_cells": self.h3_cells,
            "output_gcs_path": self.output_gcs_path,
            "score_distribution": self.score_distribution,
            "model_version": self.model_version,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PublishResult:
    """Result of publishing scores to Firestore."""

    rows_upserted: int
    firestore_collection: str
    duration_seconds: float
    model_version: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_upserted": self.rows_upserted,
            "firestore_collection": self.firestore_collection,
            "duration_seconds": self.duration_seconds,
            "model_version": self.model_version,
            "success": self.success,
            "error": self.error,
        }
