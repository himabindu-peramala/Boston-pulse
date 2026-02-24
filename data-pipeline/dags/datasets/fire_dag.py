"""
Boston Pulse - Fire Dataset DAG

Complete Airflow DAG for the fire incident data pipeline.

Pipeline Stages:
    1. Ingest
    2. Validate Raw
    3. Preprocess
    4. Validate Processed
    5. Build Features
    6. Validate Features
    7. Detect Drift
    8. Check Fairness
    9. Mitigate Bias
    10. Generate Model Card
    11. Update Watermark
    12. Pipeline Complete
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "fire_pipeline"
DATASET = "fire"
SCHEDULE = "0 2 * * *"
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
# TASK FUNCTIONS
# =============================================================================

def ingest_data(**context) -> dict:
    from dags.utils import get_effective_watermark, write_data
    from src.datasets.fire import FireIngester

    execution_date = context["ds"]
    watermark = get_effective_watermark(DATASET, lookback_days=365)  # ✅ fixed from default 7

    ingester = FireIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        result.output_path = write_data(df, DATASET, "raw", execution_date)

    return result.to_dict()


def validate_raw(**context) -> dict:
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

        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Raw validation failed: {errors[:5]}")

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
    }


def preprocess_data(**context) -> dict:
    from dags.utils import read_data, write_data
    from src.datasets.fire import FirePreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)

    preprocessor = FirePreprocessor()
    result = preprocessor.run(raw_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Preprocessing failed: {result.error_message}")

    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        result.output_path = write_data(df, DATASET, "processed", execution_date)

    return result.to_dict()


def validate_processed(**context) -> dict:
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

        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Processed validation failed: {errors[:5]}")

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
    }


def build_features(**context) -> dict:
    from dags.utils import read_data, write_data
    from src.datasets.fire import FireFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)

    builder = FireFeatureBuilder()
    result = builder.run(processed_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Feature building failed: {result.error_message}")

    df = builder.get_data()
    if df is not None and len(df) > 0:
        result.output_path = write_data(df, DATASET, "features", execution_date)

    return result.to_dict()


def detect_drift(**context) -> dict:
    from dags.utils import read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]
    current_df = read_data(DATASET, "processed", execution_date)

    try:
        prev_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=1)
        reference_df = read_data(DATASET, "processed", prev_date.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        return {"drift_detected": False, "message": "No reference data"}

    detector = DriftDetector()
    result = detector.detect_drift(current_df, reference_df, DATASET)

    return {
        "drift_detected": result.has_drift,
        "severity": result.severity.value,
        "drifted_features": result.drifted_features,
        "overall_psi": result.overall_psi,
    }


def check_fairness(**context) -> dict:
    from dags.utils import read_data
    from src.bias.fairness_checker import FairnessChecker

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if df.empty:
        return {"passes_fairness_gate": True, "has_violations": False}

    checker = FairnessChecker()
    result = checker.evaluate_fairness(
        df=df,
        dataset=DATASET,
        outcome_column=None,
        dimensions=["district"],
    )

    return {
        "passes_fairness_gate": result.passes_fairness_gate,
        "has_violations": result.has_violations,
        "violation_count": len(result.violations),
    }


def update_watermark(**context) -> dict:
    import pandas as pd

    from dags.utils import read_data, set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if "alarm_date" in df.columns and len(df) > 0:  # ✅ fixed from incident_datetime
        max_date = pd.to_datetime(df["alarm_date"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True}

    return {"watermark_updated": False}


def pipeline_complete(**context) -> dict:
    return {"status": "complete"}


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Fire incident data pipeline with validation and fairness checks",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["fire", "dataset"],
    max_active_runs=1,
) as dag:

    t_ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    t_validate_raw = PythonOperator(task_id="validate_raw", python_callable=validate_raw)
    t_preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t_validate_processed = PythonOperator(task_id="validate_processed", python_callable=validate_processed)
    t_build_features = PythonOperator(task_id="build_features", python_callable=build_features)
    t_detect_drift = PythonOperator(task_id="detect_drift", python_callable=detect_drift)
    t_check_fairness = PythonOperator(task_id="check_fairness", python_callable=check_fairness)
    t_update_watermark = PythonOperator(task_id="update_watermark", python_callable=update_watermark)
    t_pipeline_complete = PythonOperator(task_id="pipeline_complete", python_callable=pipeline_complete)

    (
        t_ingest
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> t_build_features
        >> [t_detect_drift, t_check_fairness]
        >> t_update_watermark
        >> t_pipeline_complete
    )
