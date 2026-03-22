"""Tests for shared/schemas.py dataclass to_dict coverage."""

from __future__ import annotations

from shared.schemas import (
    BiasResult,
    FeatureLoadResult,
    PublishResult,
    ScoringResult,
    TargetBuildResult,
    TrainingResult,
    TuningResult,
    ValidationResult,
)


class TestSchemasToDict:
    def test_feature_load_result(self) -> None:
        r = FeatureLoadResult(rows=10, h3_cells=5, columns=["a"], success=True, error=None)
        d = r.to_dict()
        assert d["rows"] == 10 and d["success"] is True
        r2 = FeatureLoadResult(rows=0, h3_cells=0, columns=[], success=False, error="x")
        assert r2.to_dict()["error"] == "x"

    def test_target_build_result(self) -> None:
        r = TargetBuildResult(
            rows=1, h3_cells=1, mean_danger_rate=0.5, zero_rate_cells=0, success=True
        )
        assert "mean_danger_rate" in r.to_dict()

    def test_tuning_result(self) -> None:
        r = TuningResult(
            best_params={"a": 1},
            best_val_rmse=0.1,
            n_trials=3,
            mlflow_parent_run_id="p1",
            success=True,
        )
        assert r.to_dict()["mlflow_parent_run_id"] == "p1"

    def test_training_result(self) -> None:
        r = TrainingResult(
            model_path="/m.lgb",
            train_rmse=1.0,
            val_rmse=2.0,
            best_iteration=10,
            n_features=5,
            n_train_rows=100,
            n_val_rows=20,
            mlflow_run_id="r1",
            success=True,
        )
        assert r.to_dict()["n_val_rows"] == 20

    def test_validation_result(self) -> None:
        r = ValidationResult(
            rmse_val=1.0,
            rmse_train=0.5,
            overfit_ratio=2.0,
            passed=True,
            shap_artifact_path="gs://x",
            feature_importance={"f": 0.5},
        )
        d = r.to_dict()
        assert d["shap_artifact_path"] == "gs://x" and d["feature_importance"]["f"] == 0.5

    def test_bias_result(self) -> None:
        r = BiasResult(
            passed=True,
            overall_rmse=1.0,
            slice_results={},
            worst_slice="a",
            worst_deviation_pct=5.0,
            report_gcs_path="gs://r",
        )
        assert r.to_dict()["worst_slice"] == "a"

    def test_scoring_result(self) -> None:
        r = ScoringResult(
            rows_scored=10,
            h3_cells=2,
            output_gcs_path="gs://o",
            score_distribution={"low": 5},
            model_version="v1",
            success=True,
        )
        assert r.to_dict()["score_distribution"]["low"] == 5

    def test_publish_result(self) -> None:
        r = PublishResult(
            rows_upserted=5,
            firestore_collection="c",
            duration_seconds=1.0,
            model_version="v1",
            success=True,
        )
        assert r.to_dict()["firestore_collection"] == "c"
