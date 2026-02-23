"""
Boston Pulse - CityScore DAG

Complete Airflow DAG for the CityScore data pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "cityscore_pipeline"
DATASET = "cityscore"
SCHEDULE = "0 4 * * *"  # Daily at 4 AM UTC
START_DATE = datetime(2024, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# =============================================================================
# Task Functions
# =============================================================================


def ingest_data(**context) -> dict:
    """Ingest CityScore data."""
    from airflow.exceptions import AirflowSkipException

    from dags.utils import get_effective_watermark, write_data
    from src.datasets.cityscore import CityScoreIngester

    execution_date = context["ds"]
    watermark = get_effective_watermark(DATASET, lookback_days=30)

    ingester = CityScoreIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path
    else:
        raise AirflowSkipException(f"No new data found for {DATASET} on {execution_date}")

    return result.to_dict()


def validate_raw(**context) -> dict:
    """Validate CityScore raw data."""
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

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def preprocess_data(**context) -> dict:
    """Preprocess CityScore data."""
    from dags.utils import read_data, write_data
    from src.datasets.cityscore import CityScorePreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)

    preprocessor = CityScorePreprocessor()
    result = preprocessor.run(raw_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Preprocessing failed: {result.error_message}")

    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_processed(**context) -> dict:
    """Validate CityScore processed data."""
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

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def build_features(**context) -> dict:
    """Build CityScore features."""
    from dags.utils import read_data, write_data
    from src.datasets.cityscore import CityScoreFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)

    builder = CityScoreFeatureBuilder()
    result = builder.run(processed_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Feature building failed: {result.error_message}")

    df = builder.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "features", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_features(**context) -> dict:
    """Validate CityScore features."""
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
        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Features validation failed: {errors[:5]}")

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def update_watermark(**context) -> dict:
    """Update watermark for CityScore."""
    from dags.utils import read_data, set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if "timestamp" in df.columns and len(df) > 0:
        import pandas as pd

        max_date = pd.to_datetime(df["timestamp"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True, "new_watermark": max_date.isoformat()}
    return {"watermark_updated": False}


def pipeline_complete(**context) -> dict:
    """Pipeline complete alert."""
    from dags.utils import alert_pipeline_complete

    execution_date = context["ds"]
    alert_pipeline_complete(
        dataset=DATASET, execution_date=execution_date, duration_seconds=0, stats={}, dag_id=DAG_ID
    )
    return {"status": "complete"}


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="CityScore data pipeline",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["cityscore", "dataset"],
) as dag:
    from dags.utils import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t_ingest = PythonOperator(
        task_id="ingest_data", python_callable=ingest_data, on_failure_callback=on_task_failure
    )
    t_validate_raw = PythonOperator(
        task_id="validate_raw", python_callable=validate_raw, on_failure_callback=on_task_failure
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
    t_update_watermark = PythonOperator(
        task_id="update_watermark",
        python_callable=update_watermark,
        on_failure_callback=on_task_failure,
    )
    t_pipeline_complete = PythonOperator(
        task_id="pipeline_complete", python_callable=pipeline_complete, trigger_rule="all_success"
    )

    (
        t_ingest
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> t_build_features
        >> t_validate_features
        >> t_update_watermark
        >> t_pipeline_complete
    )
