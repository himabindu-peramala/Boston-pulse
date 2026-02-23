"""
Boston Pulse - Street Sweeping Schedules DAG

Complete Airflow DAG for the street sweeping schedules data pipeline.

Pipeline Stages:
    1. Ingest: Fetch data from Analyze Boston API
    2. Validate Raw: Check raw data schema and quality
    3. Preprocess: Clean and transform data
    4. Validate Processed: Check processed data schema and quality
    5. Build Features: Create engineered features
    6. Validate Features: Check feature schema and quality
    7. Detect Drift: Check for data distribution changes
    8. Check Fairness: Evaluate fairness metrics
    9. Generate Model Card: Create dataset documentation
    10. Update Watermark: Track last successful run
    11. Pipeline Complete: Send summary alert
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "street_sweeping_pipeline"
DATASET = "street_sweeping"
SCHEDULE = "@monthly"
START_DATE = datetime(2024, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(hours=1),
}


# =============================================================================
# Task Functions
# =============================================================================


def ingest_data(**context) -> dict:
    """Ingest street sweeping data from Analyze Boston API."""
    from dags.utils import write_data
    from src.datasets.street_sweeping.ingest import StreetSweepingIngester

    execution_date = context["ds"]

    ingester = StreetSweepingIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=None)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_raw(**context) -> dict:
    """Validate raw street sweeping data against schema."""
    from dags.utils import alert_validation_failure, read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "raw", execution_date)

    enforcer = SchemaEnforcer()
    result = enforcer.validate_raw(df, DATASET)

    if not result.is_valid:
        errors = [str(e) for e in result.errors]
        alert_validation_failure(
            dataset=DATASET,
            stage="raw",
            errors=errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_raw",
        )

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "raw",
    }


def preprocess_data(**context) -> dict:
    """Preprocess raw street sweeping data."""
    from dags.utils import read_data, write_data
    from src.datasets.street_sweeping.preprocess import StreetSweepingPreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)

    preprocessor = StreetSweepingPreprocessor()
    result = preprocessor.run(raw_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Preprocessing failed: {result.error_message}")

    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_processed(**context) -> dict:
    """Validate processed street sweeping data against schema."""
    from dags.utils import alert_validation_failure, read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    enforcer = SchemaEnforcer()
    result = enforcer.validate_processed(df, DATASET)

    if not result.is_valid:
        errors = [str(e) for e in result.errors]
        alert_validation_failure(
            dataset=DATASET,
            stage="processed",
            errors=errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_processed",
        )

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "processed",
    }


def build_features(**context) -> dict:
    """Build street sweeping features from processed data."""
    from dags.utils import read_data, write_data
    from src.datasets.street_sweeping.features import StreetSweepingFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)

    builder = StreetSweepingFeatureBuilder()
    result = builder.run(processed_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Feature building failed: {result.error_message}")

    df = builder.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "features", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_features(**context) -> dict:
    """Validate street sweeping features against schema."""
    from dags.utils import alert_validation_failure, read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "features", execution_date)

    enforcer = SchemaEnforcer()
    result = enforcer.validate_features(df, DATASET)

    if not result.is_valid:
        errors = [str(e) for e in result.errors]
        alert_validation_failure(
            dataset=DATASET,
            stage="features",
            errors=errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_features",
        )

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "features",
    }


def detect_drift(**context) -> dict:
    """Detect data drift in processed street sweeping data."""
    from dags.utils import alert_drift_detected, read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]
    current_df = read_data(DATASET, "processed", execution_date)

    try:
        from datetime import datetime, timedelta

        prev_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=30)
        reference_df = read_data(DATASET, "processed", prev_date.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        return {
            "drift_detected": False,
            "message": "No reference data available for drift detection",
            "drifted_features": [],
        }

    detector = DriftDetector()
    result = detector.detect_drift(current_df, reference_df, DATASET)

    if result.has_drift:
        severity = "warning" if result.severity.value == "warning" else "critical"
        alert_drift_detected(
            dataset=DATASET,
            drifted_features=result.drifted_features,
            psi_scores={f: r.psi_score for f, r in result.feature_results.items()},
            severity=severity,
            execution_date=execution_date,
            dag_id=DAG_ID,
        )

    return {
        "drift_detected": result.has_drift,
        "severity": result.severity.value,
        "drifted_features": result.drifted_features,
        "overall_psi": result.overall_psi,
    }


def check_fairness(**context) -> dict:
    """
    Check fairness metrics for street sweeping dataset.

    Evaluates whether street sweeping schedules are equitably
    distributed across different districts.
    """
    from dags.utils import alert_fairness_violation, read_data
    from src.bias.fairness_checker import FairnessChecker, FairnessViolationError

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if df.empty:
        return {
            "passes_fairness_gate": True,
            "has_violations": False,
            "violation_count": 0,
            "message": "No data to evaluate",
        }

    checker = FairnessChecker()
    result = checker.evaluate_fairness(
        df=df,
        dataset=DATASET,
        outcome_column=None,
        dimensions=["district"],
    )

    report = checker.create_fairness_report(result)
    print(report)

    if result.has_violations:
        violations_payload = [
            {
                "metric": v.metric.value,
                "severity": v.severity.value,
                "dimension": v.dimension,
                "slice_value": str(v.slice_value),
                "expected": round(v.expected, 4),
                "actual": round(v.actual, 4),
                "disparity": round(v.disparity, 4),
                "message": v.message,
            }
            for v in result.violations
        ]
        alert_fairness_violation(
            dataset=DATASET,
            violations=violations_payload,
            execution_date=execution_date,
            dag_id=DAG_ID,
        )

    if not result.passes_fairness_gate:
        raise FairnessViolationError(result)

    return {
        "passes_fairness_gate": result.passes_fairness_gate,
        "has_violations": result.has_violations,
        "has_critical_violations": result.has_critical_violations,
        "violation_count": len(result.violations),
        "critical_count": len(result.critical_violations),
        "warning_count": len(result.warning_violations),
        "slices_evaluated": result.slices_evaluated,
        "violations": [
            {
                "metric": v.metric.value,
                "severity": v.severity.value,
                "dimension": v.dimension,
                "message": v.message,
            }
            for v in result.violations
        ],
    }


def generate_model_card(**context) -> dict:
    """Generate model card for street sweeping dataset."""
    from dags.utils import read_data
    from src.bias import ModelCardGenerator

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    ti = context["ti"]
    validation_result = ti.xcom_pull(task_ids="validate_processed")
    drift_result = ti.xcom_pull(task_ids="detect_drift")
    fairness_result = ti.xcom_pull(task_ids="check_fairness")

    generator = ModelCardGenerator()
    card = generator.generate_model_card(
        dataset=DATASET,
        df=df,
        version=execution_date.replace("-", ""),
        description="Boston street sweeping schedules from Public Works Department",
        validation_result=validation_result,
        drift_result=drift_result,
        fairness_result=fairness_result,
    )

    output_path = generator.save_model_card(card, format="both")

    return {
        "model_card_generated": True,
        "output_path": output_path,
        "version": card.version,
    }


def update_watermark(**context) -> dict:
    """Update watermark after successful pipeline completion."""
    from dags.utils import set_watermark

    execution_date = context["ds"]
    set_watermark(DATASET, datetime.strptime(execution_date, "%Y-%m-%d"), execution_date)

    return {
        "watermark_updated": True,
        "new_watermark": execution_date,
    }


def pipeline_complete(**context) -> dict:
    """Final task to mark pipeline completion and send summary alert."""
    from dags.utils import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context["ti"]

    ingest_result = ti.xcom_pull(task_ids="ingest_data")
    preprocess_result = ti.xcom_pull(task_ids="preprocess_data")
    features_result = ti.xcom_pull(task_ids="build_features")

    stats = {
        "rows_ingested": ingest_result.get("rows_fetched", 0) if ingest_result else 0,
        "rows_processed": preprocess_result.get("rows_output", 0) if preprocess_result else 0,
        "features_generated": features_result.get("features_computed", 0) if features_result else 0,
    }

    duration = (
        (ingest_result.get("duration_seconds", 0) if ingest_result else 0)
        + (preprocess_result.get("duration_seconds", 0) if preprocess_result else 0)
        + (features_result.get("duration_seconds", 0) if features_result else 0)
    )

    alert_pipeline_complete(
        dataset=DATASET,
        execution_date=execution_date,
        duration_seconds=duration,
        stats=stats,
        dag_id=DAG_ID,
    )

    return {"status": "complete", "stats": stats}


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Street sweeping schedules data pipeline with validation and fairness checks",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["boston-pulse", "street-sweeping", "public-works"],
    max_active_runs=1,
) as dag:
    from dags.utils import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        on_failure_callback=on_task_failure,
    )

    t_validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=validate_raw,
        on_failure_callback=on_task_failure,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        on_failure_callback=on_task_failure,
    )

    t_validate_processed = PythonOperator(
        task_id="validate_processed",
        python_callable=validate_processed,
        on_failure_callback=on_task_failure,
    )

    t_build_features = PythonOperator(
        task_id="build_features",
        python_callable=build_features,
        on_failure_callback=on_task_failure,
    )

    t_validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
        on_failure_callback=on_task_failure,
    )

    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
        on_failure_callback=on_task_failure,
    )

    t_check_fairness = PythonOperator(
        task_id="check_fairness",
        python_callable=check_fairness,
        on_failure_callback=on_task_failure,
    )

    t_generate_model_card = PythonOperator(
        task_id="generate_model_card",
        python_callable=generate_model_card,
        on_failure_callback=on_task_failure,
    )

    t_update_watermark = PythonOperator(
        task_id="update_watermark",
        python_callable=update_watermark,
        on_failure_callback=on_task_failure,
    )

    t_pipeline_complete = PythonOperator(
        task_id="pipeline_complete",
        python_callable=pipeline_complete,
        trigger_rule="all_success",
    )

    # Define task dependencies
    (
        t_ingest
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> t_build_features
        >> t_validate_features
        >> [t_detect_drift, t_check_fairness]
        >> t_generate_model_card
        >> t_update_watermark
        >> t_pipeline_complete
    )
