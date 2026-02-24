"""
Boston Pulse - Lineage DAG Utilities

Convenience functions for lineage tracking in Airflow DAGs.
These wrap the LineageTracker class for easy use in DAG task functions.

Usage in DAG:
    from dags.utils import record_pipeline_lineage

    def record_lineage(**context) -> dict:
        return record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=context,
        )

For debugging:
    from dags.utils import (
        get_lineage_for_date,
        compare_runs,
        get_restore_commands,
        print_lineage_summary,
    )

    lineage = get_lineage_for_date("crime", "2026-02-21")
    diff = compare_runs("crime", "2026-02-20", "2026-02-21")
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.config import Settings
from src.shared.lineage import LineageDiff, LineageRecord, LineageTracker

logger = logging.getLogger(__name__)


# =============================================================================
# DAG Task Functions
# =============================================================================


def record_pipeline_lineage(
    dataset: str,
    dag_id: str,
    context: dict[str, Any],
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Record lineage for a pipeline run - designed to be called from DAG task.

    This function:
    1. Extracts stats from upstream tasks via XCom
    2. Records lineage with GCS generation numbers
    3. Returns a summary dict for XCom

    Args:
        dataset: Dataset name (crime, service_311, etc.)
        dag_id: Airflow DAG ID
        context: Airflow context dict (passed from **context)
        config: Optional configuration object

    Returns:
        Dict with lineage recording results

    Example:
        def record_lineage(**context) -> dict:
            return record_pipeline_lineage(
                dataset="crime",
                dag_id="crime_pipeline",
                context=context,
            )
    """
    execution_date = context["ds"]
    run_id = context.get("run_id")
    ti = context.get("ti")

    # Extract stats from upstream tasks via XCom
    rows_ingested = None
    rows_processed = None
    features_generated = None
    drift_detected = None
    fairness_passed = None

    if ti is not None:
        # Try to get ingest stats
        ingest_result = ti.xcom_pull(task_ids="ingest_data")
        if ingest_result:
            rows_ingested = ingest_result.get("rows_fetched")

        # Try to get preprocess stats
        preprocess_result = ti.xcom_pull(task_ids="preprocess_data")
        if preprocess_result:
            rows_processed = preprocess_result.get("rows_output")

        # Try to get features stats
        features_result = ti.xcom_pull(task_ids="build_features")
        if features_result:
            features_generated = features_result.get("features_computed")

        # Try to get drift detection result
        drift_result = ti.xcom_pull(task_ids="detect_drift")
        if drift_result:
            drift_detected = drift_result.get("drift_detected")

        # Try to get fairness result
        fairness_result = ti.xcom_pull(task_ids="check_fairness")
        if fairness_result:
            fairness_passed = fairness_result.get("passes_fairness_gate")

    # Record lineage
    tracker = LineageTracker(config)
    lineage = tracker.record_lineage(
        dataset=dataset,
        execution_date=execution_date,
        dag_id=dag_id,
        run_id=run_id,
        rows_ingested=rows_ingested,
        rows_processed=rows_processed,
        features_generated=features_generated,
        drift_detected=drift_detected,
        fairness_passed=fairness_passed,
    )

    # Log summary
    logger.info(f"Lineage recorded for {dataset}/{execution_date}")
    logger.info(f"  Data versions: {list(lineage.data_versions.keys())}")
    logger.info(f"  Schema versions: {list(lineage.schema_versions.keys())}")

    for layer, version in lineage.data_versions.items():
        logger.info(f"    {layer}: generation={version.generation}")

    return {
        "lineage_recorded": True,
        "dataset": dataset,
        "execution_date": execution_date,
        "data_versions": {k: v.generation for k, v in lineage.data_versions.items()},
        "schema_versions": {k: v.generation for k, v in lineage.schema_versions.items()},
        "rows_ingested": rows_ingested,
        "rows_processed": rows_processed,
        "features_generated": features_generated,
        "drift_detected": drift_detected,
        "fairness_passed": fairness_passed,
    }


def create_record_lineage_task(
    dataset: str,
    dag_id: str,
    config: Settings | None = None,
):
    """
    Create a record_lineage task function for a specific dataset.

    This is a factory function that returns a task function
    pre-configured for a specific dataset.

    Args:
        dataset: Dataset name
        dag_id: DAG ID
        config: Optional configuration

    Returns:
        Task function that can be used with PythonOperator

    Example:
        record_lineage = create_record_lineage_task("crime", "crime_pipeline")

        t_record_lineage = PythonOperator(
            task_id="record_lineage",
            python_callable=record_lineage,
        )
    """

    def _record_lineage(**context) -> dict:
        return record_pipeline_lineage(
            dataset=dataset,
            dag_id=dag_id,
            context=context,
            config=config,
        )

    return _record_lineage


# =============================================================================
# Query Functions
# =============================================================================


def get_lineage_for_date(
    dataset: str,
    execution_date: str,
    config: Settings | None = None,
) -> LineageRecord | None:
    """
    Get lineage record for a specific date.

    Args:
        dataset: Dataset name
        execution_date: Date in YYYY-MM-DD format
        config: Optional configuration

    Returns:
        LineageRecord or None if not found
    """
    tracker = LineageTracker(config)
    return tracker.get_lineage(dataset, execution_date)


def get_dataset_lineage_history(
    dataset: str,
    limit: int = 30,
    config: Settings | None = None,
) -> list[LineageRecord]:
    """
    Get recent lineage records for a dataset.

    Args:
        dataset: Dataset name
        limit: Maximum records to return
        config: Optional configuration

    Returns:
        List of LineageRecord sorted newest first
    """
    tracker = LineageTracker(config)
    return tracker.get_dataset_history(dataset, limit)


def compare_runs(
    dataset: str,
    date_from: str,
    date_to: str,
    config: Settings | None = None,
) -> LineageDiff:
    """
    Compare lineage between two dates.

    Shows exactly what changed between two pipeline runs.

    Args:
        dataset: Dataset name
        date_from: Earlier date
        date_to: Later date
        config: Optional configuration

    Returns:
        LineageDiff showing all changes
    """
    tracker = LineageTracker(config)
    return tracker.compare_lineage(dataset, date_from, date_to)


def find_runs_with_schema(
    dataset: str,
    schema_generation: str,
    layer: str = "features",
    config: Settings | None = None,
) -> list[LineageRecord]:
    """
    Find all runs that used a specific schema version.

    Useful for identifying runs affected by a bad schema.

    Args:
        dataset: Dataset name
        schema_generation: GCS generation number
        layer: Schema layer to check
        config: Optional configuration

    Returns:
        List of LineageRecord using this schema
    """
    tracker = LineageTracker(config)
    return tracker.find_runs_by_schema_generation(dataset, schema_generation, layer)


# =============================================================================
# Recovery Functions
# =============================================================================


def get_restore_commands(
    dataset: str,
    execution_date: str,
    layers: list[str] | None = None,
    config: Settings | None = None,
) -> dict[str, str]:
    """
    Get gsutil commands to restore data to a specific state.

    Args:
        dataset: Dataset name
        execution_date: Date to restore to
        layers: Specific layers (default: all)
        config: Optional configuration

    Returns:
        Dict mapping layer to gsutil command
    """
    tracker = LineageTracker(config)
    return tracker.get_restore_commands(dataset, execution_date, layers)


def print_restore_commands(
    dataset: str,
    execution_date: str,
    layers: list[str] | None = None,
    config: Settings | None = None,
) -> None:
    """Print restore commands in a copy-paste friendly format."""
    commands = get_restore_commands(dataset, execution_date, layers, config)

    print(f"\n# Restore commands for {dataset}/{execution_date}")
    print("# Copy and run these commands to restore data:\n")

    for layer, command in commands.items():
        print(f"# {layer}")
        print(command)
        print()


# =============================================================================
# Display Functions
# =============================================================================


def print_lineage_summary(
    dataset: str,
    execution_date: str,
    config: Settings | None = None,
) -> None:
    """
    Print a human-readable lineage summary.

    Args:
        dataset: Dataset name
        execution_date: Date to display
        config: Optional configuration
    """
    tracker = LineageTracker(config)
    tracker.print_lineage_summary(dataset, execution_date)


def print_lineage_diff(
    dataset: str,
    date_from: str,
    date_to: str,
    config: Settings | None = None,
) -> None:
    """
    Print a human-readable diff between two dates.

    Args:
        dataset: Dataset name
        date_from: Earlier date
        date_to: Later date
        config: Optional configuration
    """
    diff = compare_runs(dataset, date_from, date_to, config)

    print(f"\n{'='*60}")
    print(f"LINEAGE DIFF: {dataset}")
    print(f"From: {date_from} → To: {date_to}")
    print(f"{'='*60}")

    if diff.data_changes:
        print("\n--- Data Changes ---")
        for layer, change in diff.data_changes.items():
            status = change.get("status", "unknown")
            if status == "changed":
                print(f"  {layer}: {change.get('from')} → {change.get('to')}")
            elif status == "added":
                print(f"  {layer}: [NEW] {change.get('to')}")
            elif status == "removed":
                print(f"  {layer}: [REMOVED] {change.get('from')}")
    else:
        print("\n--- Data Changes ---")
        print("  No changes")

    if diff.schema_changes:
        print("\n--- Schema Changes ---")
        for layer, change in diff.schema_changes.items():
            print(f"  {layer}: {change.get('from')} → {change.get('to')}")
    else:
        print("\n--- Schema Changes ---")
        print("  No changes")

    if diff.stats_changes:
        print("\n--- Stats Changes ---")
        for stat, change in diff.stats_changes.items():
            print(f"  {stat}: {change.get('from')} → {change.get('to')}")
    else:
        print("\n--- Stats Changes ---")
        print("  No changes")

    print(f"{'='*60}\n")
