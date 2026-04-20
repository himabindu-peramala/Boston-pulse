"""
Boston Pulse ML - Crime Navigate Training CLI.

Command-line interface for running the full training pipeline.
Called by the DAG inside a Docker container:

    python -m models.crime_navigate.cli train \
        --execution-date 2026-03-23 \
        --stage staging \
        --output-json /tmp/results.json

The CLI orchestrates the full pipeline:
    1. Load features from GCS
    2. Build targets (danger_rate label)
    3. Tune hyperparameters (Optuna)
    4. Train model (LightGBM)
    5. Validate model (RMSE gate, overfit gate, SHAP)
    6. Check bias (Fairlearn)
    7. Push to registry (AR + GCS)
    8. Score all cells
    9. Publish to Firestore

All results are written to --output-json for the DAG to parse.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum rows required to proceed with training
# Skip training if features have fewer rows (e.g., data pipeline failed or no new data)
MIN_ROWS_FOR_TRAINING = 1


def load_config() -> dict[str, Any]:
    """Load training configuration from YAML."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "crime_navigate_train.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training_pipeline(
    execution_date: str,
    stage: str,
    skip_tuning: bool = False,
    skip_publish: bool = False,
) -> dict[str, Any]:
    """
    Run the full training pipeline.

    Args:
        execution_date: Date string (YYYY-MM-DD)
        stage: Initial stage for model ("staging" or "production")
        skip_tuning: Skip hyperparameter tuning (use defaults)
        skip_publish: Skip Firestore publish

    Returns:
        Dict with all pipeline results
    """
    from models.crime_navigate.bias_checker import check_bias
    from models.crime_navigate.feature_loader import load_features
    from models.crime_navigate.publisher import publish_scores
    from models.crime_navigate.scorer import score_all_cells
    from models.crime_navigate.target_builder import build_targets
    from models.crime_navigate.trainer import random_split, train_model
    from models.crime_navigate.tuner import get_default_params, tune_hyperparams
    from models.crime_navigate.validator import validate_model
    from shared.alerting import (
        alert_gate_failure,
        alert_model_pushed,
        alert_scores_published,
        alert_training_complete,
        alert_training_start,
    )
    from shared.registry import ModelRegistry

    cfg = load_config()
    bucket = cfg["data"]["bucket"]
    dataset = "crime_navigate"
    dag_id = "crime_navigate_train"

    results: dict[str, Any] = {
        "execution_date": execution_date,
        "stage": stage,
        "status": "running",
        "steps": {},
    }

    alert_training_start(dataset, execution_date, dag_id)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow_run = mlflow.start_run(run_name=f"train_{execution_date}")
    mlflow_run_id = mlflow_run.info.run_id
    results["mlflow_run_id"] = mlflow_run_id

    try:
        logger.info("Step 1: Loading features...")
        features_df, feature_result = load_features(execution_date, cfg, bucket)
        results["steps"]["load_features"] = feature_result.to_dict()
        if not feature_result.success:
            raise RuntimeError(f"Feature loading failed: {feature_result.error}")
        logger.info(f"Loaded {feature_result.rows:,} rows, {feature_result.h3_cells:,} cells")

        # Skip training if insufficient data (e.g., pipeline failed or no new data)
        if feature_result.rows < MIN_ROWS_FOR_TRAINING:
            logger.info(
                f"Only {feature_result.rows} rows (< {MIN_ROWS_FOR_TRAINING} threshold) — "
                "skipping training to avoid wasting compute"
            )
            results["status"] = "skipped_insufficient_data"
            results["reason"] = (
                f"Feature rows ({feature_result.rows}) below minimum ({MIN_ROWS_FOR_TRAINING})"
            )
            mlflow.end_run()
            return results

        logger.info("Step 2: Building targets...")
        training_df, target_result = build_targets(features_df, execution_date, cfg, bucket)
        results["steps"]["build_targets"] = target_result.to_dict()
        if not target_result.success:
            raise RuntimeError(f"Target building failed: {target_result.error}")
        logger.info(
            f"Training matrix: {target_result.rows:,} rows, "
            f"mean_danger_rate={target_result.mean_danger_rate:.4f}"
        )

        logger.info("Step 3: Splitting data...")
        train_df, val_df = random_split(training_df, cfg)
        logger.info(f"Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")

        if skip_tuning:
            logger.info("Step 4: Using default hyperparameters (tuning skipped)")
            best_params = get_default_params(cfg)
            tuning_result_dict = {"skipped": True, "best_params": best_params}
        else:
            logger.info("Step 4: Tuning hyperparameters...")
            best_params, tuning_result = tune_hyperparams(train_df, val_df, cfg, mlflow_run_id)
            tuning_result_dict = tuning_result.to_dict()
            logger.info(f"Best RMSE: {tuning_result.best_val_rmse:.4f}")
        results["steps"]["tune"] = tuning_result_dict

        logger.info("Step 5: Training model...")
        model, model_path, training_result = train_model(
            train_df, val_df, best_params, cfg, mlflow_run_id
        )
        results["steps"]["train"] = training_result.to_dict()
        logger.info(
            f"Train RMSE: {training_result.train_rmse:.4f}, "
            f"Val RMSE: {training_result.val_rmse:.4f}"
        )

        logger.info("Step 6: Validating model...")
        try:
            validation_result = validate_model(model, val_df, training_result, cfg, mlflow_run_id)
            results["steps"]["validate"] = validation_result.to_dict()
            logger.info(f"Validation PASSED: RMSE={validation_result.rmse_val:.4f}")
        except Exception as e:
            alert_gate_failure(dataset, execution_date, "Validation", str(e), dag_id)
            raise

        logger.info("Step 7: Checking bias...")
        try:
            bias_result = check_bias(model, val_df, execution_date, cfg, bucket, mlflow_run_id)
            results["steps"]["bias"] = bias_result.to_dict()
            logger.info(
                f"Bias check PASSED: worst_deviation={bias_result.worst_deviation_pct:.1f}%"
            )
        except Exception as e:
            alert_gate_failure(dataset, execution_date, "Bias", str(e), dag_id)
            raise

        logger.info("Step 8: Pushing to registry...")
        version = datetime.strptime(execution_date, "%Y-%m-%d").strftime("%Y%m%d")
        registry = ModelRegistry(cfg)

        metadata = {
            "execution_date": execution_date,
            "train_rmse": training_result.train_rmse,
            "val_rmse": training_result.val_rmse,
            "best_iteration": training_result.best_iteration,
            "n_features": training_result.n_features,
            "n_train_rows": training_result.n_train_rows,
            "n_val_rows": training_result.n_val_rows,
            "bias_worst_deviation_pct": bias_result.worst_deviation_pct,
            "mlflow_run_id": mlflow_run_id,
            "git_sha": os.getenv("GIT_SHA", "unknown"),
            "ml_image": os.getenv("ML_IMAGE", "unknown"),
        }

        shap_path = validation_result.shap_artifact_path
        model_uri = registry.push(
            model_path=model_path,
            version=version,
            metadata=metadata,
            update_latest=True,
            shap_path=shap_path,
            stage=stage,
        )
        results["steps"]["push"] = {
            "version": version,
            "model_uri": model_uri,
            "stage": stage,
        }
        logger.info(f"Model pushed: {model_uri}")

        alert_model_pushed(
            dataset, execution_date, version, model_uri, training_result.val_rmse, dag_id
        )

        # Gate 3: Compare candidate to current production before promoting
        if stage == "staging":
            from shared.alerting import alert_model_promoted, alert_promotion_skipped

            promotion_cfg = cfg.get("promotion", {})
            force_promote = promotion_cfg.get("force_promote", False)
            tolerance = promotion_cfg.get("tolerance", 0.02)

            if force_promote:
                logger.warning("force_promote=true — skipping Gate 3 comparison")
                comparison: dict[str, Any] = {
                    "should_promote": True,
                    "reason": "force_promote enabled in config",
                    "candidate_rmse": training_result.val_rmse,
                }
            else:
                logger.info("Step 8b: Gate 3 — comparing candidate to production...")
                comparison = registry.compare_to_production(
                    candidate_val_rmse=training_result.val_rmse,
                    tolerance=tolerance,
                )
                logger.info(f"Gate 3 result: {comparison['reason']}")

            results["steps"]["push"]["comparison"] = comparison

            if comparison["should_promote"]:
                registry.promote_to_production(version)
                results["steps"]["push"]["promoted_to_production"] = True
                logger.info(f"Promoted {version} to production")
                alert_model_promoted(dataset, execution_date, version, comparison, dag_id)
            else:
                results["steps"]["push"]["promoted_to_production"] = False
                logger.warning(
                    f"Did NOT promote {version}. "
                    f"Candidate stays in staging. Production unchanged."
                )
                alert_promotion_skipped(dataset, execution_date, version, comparison, dag_id)

        # Only score and publish if model was promoted (or skip_publish is False)
        if results["steps"]["push"].get("promoted_to_production", False):
            logger.info("Step 9: Scoring all cells with newly-promoted model...")
            scores_df, scoring_result = score_all_cells(
                model, features_df, execution_date, cfg, bucket, version
            )
            results["steps"]["score"] = scoring_result.to_dict()
            logger.info(f"Scored {scoring_result.rows_scored:,} cells")

            if not skip_publish:
                logger.info("Step 10: Publishing to Firestore...")
                publish_result = publish_scores(scores_df, cfg, version, execution_date)
                results["steps"]["publish"] = publish_result.to_dict()
                logger.info(f"Published {publish_result.rows_upserted:,} rows to Firestore")

                alert_scores_published(
                    dataset,
                    execution_date,
                    publish_result.rows_upserted,
                    scoring_result.h3_cells,
                    publish_result.duration_seconds,
                    dag_id,
                )
            else:
                results["steps"]["publish"] = {"skipped": True, "reason": "skip_publish flag"}
        else:
            logger.info("Skipping scoring — candidate was not promoted; Firestore unchanged")
            results["steps"]["score"] = {"skipped": True, "reason": "gate_3_not_promoted"}
            results["steps"]["publish"] = {"skipped": True, "reason": "gate_3_not_promoted"}

        results["status"] = "success"

        alert_training_complete(
            dataset,
            execution_date,
            results["steps"]["train"],
            results["steps"]["validate"],
            results["steps"]["bias"],
            results["steps"]["score"],
            results["steps"].get("publish", {}),
            dag_id,
        )

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        raise
    finally:
        mlflow.end_run()

    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Boston Pulse ML - Crime Navigate Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run full training pipeline")
    train_parser.add_argument(
        "--execution-date",
        required=True,
        help="Execution date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--stage",
        default="staging",
        choices=["staging", "production"],
        help="Initial model stage (default: staging)",
    )
    train_parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    train_parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Skip Firestore publish",
    )
    train_parser.add_argument(
        "--output-json",
        help="Path to write results JSON",
    )

    args = parser.parse_args()

    if args.command == "train":
        try:
            results = run_training_pipeline(
                execution_date=args.execution_date,
                stage=args.stage,
                skip_tuning=args.skip_tuning,
                skip_publish=args.skip_publish,
            )

            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results written to {args.output_json}")

            print(json.dumps(results, indent=2, default=str))
            # Return 0 for success or intentional skip (insufficient data is not an error)
            return 0 if results["status"] in ("success", "skipped_insufficient_data") else 1

        except Exception as e:
            logger.error(f"Training failed: {e}")
            error_result = {"status": "failed", "error": str(e)}
            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(error_result, f, indent=2)
            print(json.dumps(error_result))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
