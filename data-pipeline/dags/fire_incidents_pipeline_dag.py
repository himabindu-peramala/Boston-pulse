"""
Fire Incidents Data Pipeline DAG
Automated pipeline for ingesting, preprocessing, and validating Boston Fire Incident data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ingest_fire import download_fire_data, validate_fire_data
from preprocess_fire import preprocess_fire_data, generate_statistics
from bias_detection_fire import FireBiasDetector

default_args = {
    'owner': 'boston-pulse-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fire_incidents_pipeline',
    default_args=default_args,
    description='Automated pipeline for Boston Fire Incident data',
    schedule_interval='@monthly',
    start_date=datetime(2026, 2, 18),
    catchup=False,
    tags=['boston-pulse', 'fire-incidents', 'mlops'],
)


def download_data_task():
    print("Starting fire incident data download...")
    file_path = download_fire_data(year="2013", nrows=5000)
    print(f"Download complete: {file_path}")
    return file_path


download_task = PythonOperator(
    task_id='download_fire_data',
    python_callable=download_data_task,
    dag=dag,
)


def validate_data_task():
    print("Starting data validation...")
    validation = validate_fire_data("data/raw/fire_incidents.csv")
    if validation['critical_issues']:
        raise ValueError(f"Data validation failed: {validation['critical_issues']}")
    print(f"Validation passed: {validation['total_rows']} records found")
    return validation


validate_task = PythonOperator(
    task_id='validate_fire_data',
    python_callable=validate_data_task,
    dag=dag,
)


def preprocess_data_task():
    print("Starting data preprocessing...")
    output_file = preprocess_fire_data()
    print(f"Preprocessing complete: {output_file}")
    return output_file


preprocess_task = PythonOperator(
    task_id='preprocess_fire_data',
    python_callable=preprocess_data_task,
    dag=dag,
)


def generate_stats_task():
    print("Generating data statistics...")
    stats = generate_statistics()
    print(f"\n=== Pipeline Statistics ===")
    print(f"Total records: {stats['total_records']}")
    print(f"Severity distribution: {stats['severity_distribution']}")
    print(f"Total property loss: ${stats['total_property_loss']:,.0f}")
    return stats


stats_task = PythonOperator(
    task_id='generate_statistics',
    python_callable=generate_stats_task,
    dag=dag,
)


def bias_detection_task():
    print("Running bias detection analysis...")
    detector = FireBiasDetector()
    report = detector.generate_bias_report()
    detector.visualize_bias()
    print("Bias detection complete.")
    return report


bias_task = PythonOperator(
    task_id='bias_detection',
    python_callable=bias_detection_task,
    dag=dag,
)


download_task >> validate_task >> preprocess_task >> stats_task >> bias_task