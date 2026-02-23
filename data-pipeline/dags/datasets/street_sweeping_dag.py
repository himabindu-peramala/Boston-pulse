"""
Boston Pulse - Street Sweeping Schedules Airflow DAG

Orchestrates ingestion, preprocessing, and feature engineering
for Street Sweeping Schedules data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.datasets.street_sweeping.features import build_street_sweeping_features
from src.datasets.street_sweeping.ingest import ingest_street_sweeping_data
from src.datasets.street_sweeping.preprocess import preprocess_street_sweeping_data

DEFAULT_ARGS = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="street_sweeping_pipeline",
    default_args=DEFAULT_ARGS,
    description="Ingest and process Boston Street Sweeping Schedules data",
    schedule_interval="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["boston-pulse", "street-sweeping", "public-works"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_street_sweeping",
        python_callable=ingest_street_sweeping_data,
        op_kwargs={
            "execution_date": "{{ ds }}",
            "watermark_start": None,
        },
    )

    preprocess = PythonOperator(
        task_id="preprocess_street_sweeping",
        python_callable=preprocess_street_sweeping_data,
        op_kwargs={
            "df": "{{ ti.xcom_pull(task_ids='ingest_street_sweeping') }}",
            "execution_date": "{{ ds }}",
        },
    )

    build_features = PythonOperator(
        task_id="build_street_sweeping_features",
        python_callable=build_street_sweeping_features,
        op_kwargs={
            "df": "{{ ti.xcom_pull(task_ids='preprocess_street_sweeping') }}",
            "execution_date": "{{ ds }}",
        },
    )

    ingest >> preprocess >> build_features