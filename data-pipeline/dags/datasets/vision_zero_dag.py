"""
Boston Pulse - Vision Zero Safety Concerns DAG

Complete Airflow DAG for the Vision Zero data pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "vision_zero_pipeline"
DATASET = "vision_zero"
SCHEDULE = "0 4 * * *"  # Daily at 4 AM UTC
START_DATE = datetime(2025, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def ingest_data(**context) -> dict:
    from dags.utils.gcs_io import write_data
    from dags.utils.watermark import get_effective_watermark
    from src.datasets.vision_zero.ingest import VisionZeroIngester

    execution_date = context["ds"]
    watermark = get_effective_watermark(DATASET)

    ingester = VisionZeroIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path
    else:
        from airflow.exceptions import AirflowSkipException

        raise AirflowSkipException(f"No new data for {DATASET}")

    return result.to_dict()


def validate_raw(**context) -> dict:
    from dags.utils.gcs_io import read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "raw", execution_date)
    enforcer = SchemaEnforcer()
    result = enforcer.validate_raw(df, DATASET)
    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def preprocess_data(**context) -> dict:
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.vision_zero.preprocess import VisionZeroPreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)
    preprocessor = VisionZeroPreprocessor()
    result = preprocessor.run(raw_df, execution_date)
    df = preprocessor.get_data()
    output_path = write_data(df, DATASET, "processed", execution_date)
    result.output_path = output_path
    return result.to_dict()


def validate_processed(**context) -> dict:
    from dags.utils.gcs_io import read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    enforcer = SchemaEnforcer()
    result = enforcer.validate_processed(df, DATASET)
    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def build_features(**context) -> dict:
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.vision_zero.features import VisionZeroFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)
    builder = VisionZeroFeatureBuilder()
    result = builder.run(processed_df, execution_date)
    df = builder.get_data()
    output_path = write_data(df, DATASET, "features", execution_date)
    result.output_path = output_path
    return result.to_dict()


def update_watermark(**context) -> dict:
    from dags.utils.gcs_io import read_data
    from dags.utils.watermark import set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    if "creation_date" in df.columns and len(df) > 0:
        import pandas as pd

        max_date = pd.to_datetime(df["creation_date"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True, "new_watermark": max_date.isoformat()}
    return {"watermark_updated": False}


def pipeline_complete(**context) -> dict:
    from dags.utils.alerting import alert_pipeline_complete

    execution_date = context["ds"]
    alert_pipeline_complete(
        dataset=DATASET, execution_date=execution_date, duration_seconds=0, stats={}, dag_id=DAG_ID
    )
    return {"status": "complete"}


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Vision Zero Safety Concerns data pipeline",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["safety", "vision_zero"],
) as dag:
    from dags.utils.callbacks import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t1 = PythonOperator(
        task_id="ingest_data", python_callable=ingest_data, on_failure_callback=on_task_failure
    )
    t2 = PythonOperator(
        task_id="validate_raw", python_callable=validate_raw, on_failure_callback=on_task_failure
    )
    t3 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        on_failure_callback=on_task_failure,
    )
    t4 = PythonOperator(
        task_id="validate_processed",
        python_callable=validate_processed,
        on_failure_callback=on_task_failure,
    )
    t5 = PythonOperator(
        task_id="build_features",
        python_callable=build_features,
        on_failure_callback=on_task_failure,
    )
    t6 = PythonOperator(
        task_id="update_watermark",
        python_callable=update_watermark,
        on_failure_callback=on_task_failure,
    )
    t7 = PythonOperator(
        task_id="pipeline_complete", python_callable=pipeline_complete, trigger_rule="all_success"
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
