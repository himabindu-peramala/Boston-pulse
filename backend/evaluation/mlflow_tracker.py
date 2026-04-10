"""
Boston Pulse — MLflow Experiment Tracker
Logs RAG pipeline experiments: parameters, metrics, and artifacts.

Usage:
    from evaluation.mlflow_tracker import track_experiment
    track_experiment(eval_summary, params)
"""
import logging
import json
import os

import mlflow

from app.core.config import settings

logger = logging.getLogger(__name__)

# MLflow experiment name
EXPERIMENT_NAME = "boston-pulse-rag"


def setup_mlflow():
    """Initialize MLflow with GCS or local tracking."""
    gcs_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(gcs_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME} @ {gcs_uri}")


def track_experiment(
    eval_summary: dict,
    run_name: str = None,
    params: dict = None,
) -> str:
    """
    Log one complete experiment run to MLflow.

    Args:
        eval_summary: Output from eval_runner.run_evaluation()
        run_name: Human-readable name for this run
        params: Override parameters (uses defaults from settings if None)

    Returns:
        MLflow run_id
    """
    setup_mlflow()

    # Default parameters from current config
    default_params = {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.gemini_model,
        "llm_temperature": settings.gemini_temperature,
        "llm_max_tokens": settings.gemini_max_tokens,
        "top_k_results": settings.top_k_results,
        "chunk_strategy": "one_row_per_chunk",
        "chunk_overlap": 0,
        "vector_db": "chromadb",
        "similarity_metric": "cosine",
    }

    # Override with any custom params
    if params:
        default_params.update(params)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for key, value in default_params.items():
            mlflow.log_param(key, value)

        # Log overall metrics
        mlflow.log_metric("retrieval_accuracy", eval_summary["retrieval_accuracy"])
        mlflow.log_metric("avg_keyword_coverage", eval_summary["avg_keyword_coverage"])
        mlflow.log_metric("avg_similarity_score", eval_summary["avg_similarity_score"])
        mlflow.log_metric("avg_response_time_ms", eval_summary["avg_response_time_ms"])
        mlflow.log_metric("total_questions", eval_summary["total_questions"])

        # Log per-category metrics
        for cat, metrics in eval_summary.get("per_category", {}).items():
            mlflow.log_metric(
                f"retrieval_accuracy_{cat}",
                metrics["retrieval_accuracy"]
            )
            mlflow.log_metric(
                f"keyword_coverage_{cat}",
                metrics["avg_keyword_coverage"]
            )

        # Log per-difficulty metrics
        for diff, metrics in eval_summary.get("per_difficulty", {}).items():
            mlflow.log_metric(
                f"retrieval_accuracy_{diff}",
                metrics["retrieval_accuracy"]
            )

        # Save full results as artifact
        results_path = "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_summary, f, indent=2, default=str)
        mlflow.log_artifact(results_path)
        os.remove(results_path)

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {run_id}")
        return run_id


def compare_runs() -> list:
    """
    Fetch all runs from the experiment for comparison.

    Returns:
        List of dicts with run_id, params, and metrics.
    """
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        logger.warning("No experiment found")
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.retrieval_accuracy DESC"],
    )

    return runs.to_dict("records")


def get_best_run() -> dict:
    """Get the run with highest retrieval accuracy."""
    runs = compare_runs()
    if not runs:
        return {}
    return runs[0]


if __name__ == "__main__":
    # Show existing runs
    runs = compare_runs()
    if runs:
        print(f"\nFound {len(runs)} experiment runs:")
        for r in runs[:5]:
            print(f"  Run: {r.get('tags.mlflow.runName', 'unnamed')}")
            print(f"    Retrieval accuracy: {r.get('metrics.retrieval_accuracy', 'N/A')}")
            print(f"    Embedding model: {r.get('params.embedding_model', 'N/A')}")
            print()
    else:
        print("No experiment runs found yet. Run an evaluation first.")
