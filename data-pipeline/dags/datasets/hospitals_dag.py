"""
Boston Pulse - Hospital Locations DAG

Complete Airflow DAG for the Hospital Locations data pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "hospitals_pipeline"
DATASET = "hospitals"
SCHEDULE = "@yearly"  # Very static reference data
START_DATE = datetime(2025, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def ingest_data(**context) -> dict:
    from dags.utils.gcs_io import write_data
    from src.datasets.hospitals.ingest import HospitalIngester

    execution_date = context["ds"]
    ingester = HospitalIngester()
    df = ingester.fetch_data()

    if df.empty:
        raise RuntimeError(f"No data found for {DATASET}")

    output_path = write_data(df, DATASET, "raw", execution_date)
    result = ingester.run(execution_date)
    result.output_path = output_path
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
    from src.datasets.hospitals.preprocess import HospitalPreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)
    preprocessor = HospitalPreprocessor()
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
    from src.datasets.hospitals.features import HospitalFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)
    builder = HospitalFeatureBuilder()
    result = builder.run(processed_df, execution_date)
    df = builder.get_data()
    output_path = write_data(df, DATASET, "features", execution_date)
    result.output_path = output_path
    return result.to_dict()


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
    description="Hospital Locations data pipeline",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["healthcare", "reference"],
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
        task_id="pipeline_complete", python_callable=pipeline_complete, trigger_rule="all_success"
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
