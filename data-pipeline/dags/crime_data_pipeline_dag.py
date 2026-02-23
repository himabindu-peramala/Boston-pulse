"""
Crime Data Pipeline DAG
Automated pipeline for ingesting, preprocessing, and validating crime data
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ingest_crime import download_crime_data, validate_crime_data
from preprocess_crime import generate_statistics, preprocess_crime_data

# Default arguments for the DAG
default_args = {
    "owner": "boston-pulse-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Define the DAG
dag = DAG(
    "crime_data_pipeline",
    default_args=default_args,
    description="Automated pipeline for Boston crime data",
    schedule_interval="@weekly",  # Run every week
    start_date=datetime(2026, 2, 18),
    catchup=False,
    tags=["boston-pulse", "crime", "mlops"],
)


# Task 1: Download crime data
def download_data_task():
    """Download latest crime data from Boston Open Data Portal"""
    print("Starting crime data download...")
    file_path = download_crime_data(year="2023", nrows=None)
    print(f"Download complete: {file_path}")
    return file_path


download_task = PythonOperator(
    task_id="download_crime_data",
    python_callable=download_data_task,
    dag=dag,
)


# Task 2: Validate downloaded data
def validate_data_task():
    """Validate the downloaded crime data"""
    print("Starting data validation...")
    validation = validate_crime_data("data/raw/crime_data.csv")

    # Check for critical issues
    if validation["critical_issues"]:
        raise ValueError(f"Data validation failed: {validation['critical_issues']}")

    print(f"Validation passed: {validation['total_rows']} rows")
    return validation


validate_task = PythonOperator(
    task_id="validate_crime_data",
    python_callable=validate_data_task,
    dag=dag,
)


# Task 3: Preprocess data
def preprocess_data_task():
    """Preprocess and clean crime data"""
    print("Starting data preprocessing...")
    output_file = preprocess_crime_data()
    print(f"Preprocessing complete: {output_file}")
    return output_file


preprocess_task = PythonOperator(
    task_id="preprocess_crime_data",
    python_callable=preprocess_data_task,
    dag=dag,
)


# Task 4: Generate statistics
def generate_stats_task():
    """Generate statistics about processed data"""
    print("Generating data statistics...")
    stats = generate_statistics()

    print("\n=== Pipeline Statistics ===")
    print(f"Total records: {stats['total_records']}")
    print(f"Date range: {stats['date_range']}")
    print(f"Top offenses: {list(stats['top_offenses'].keys())[:3]}")

    return stats


stats_task = PythonOperator(
    task_id="generate_statistics",
    python_callable=generate_stats_task,
    dag=dag,
)


# # Task 5: Run tests
# test_task = BashOperator(
#     task_id='run_tests',
#     bash_command='cd .. && pytest tests/ -v',
#     dag=dag,
# )


# Define task dependencies (pipeline flow)
download_task >> validate_task >> preprocess_task >> stats_task
