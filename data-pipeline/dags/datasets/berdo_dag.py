"""
Boston Pulse - BERDO Airflow DAG

Orchestrates ingestion, preprocessing, and feature engineering
for BERDO building energy and emissions data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.datasets.berdo.features import build_berdo_features
from src.datasets.berdo.ingest import ingest_berdo_data
from src.datasets.berdo.preprocess import preprocess_berdo_data

DEFAULT_ARGS = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="berdo_pipeline",
    default_args=DEFAULT_ARGS,
    description="Ingest and process Boston BERDO building energy and emissions data",
    schedule_interval="@yearly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["boston-pulse", "berdo", "emissions", "housing"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_berdo",
        python_callable=ingest_berdo_data,
        op_kwargs={
            "execution_date": "{{ ds }}",
            "watermark_start": None,
        },
    )

    preprocess = PythonOperator(
        task_id="preprocess_berdo",
        python_callable=preprocess_berdo_data,
        op_kwargs={
            "df": "{{ ti.xcom_pull(task_ids='ingest_berdo') }}",
            "execution_date": "{{ ds }}",
        },
    )

    build_features = PythonOperator(
        task_id="build_berdo_features",
        python_callable=build_berdo_features,
        op_kwargs={
            "df": "{{ ti.xcom_pull(task_ids='preprocess_berdo') }}",
            "execution_date": "{{ ds }}",
        },
    )

    ingest >> preprocess >> build_features
