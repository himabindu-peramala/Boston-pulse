"""
Police Stations Data Pipeline DAG
Automated pipeline for ingesting, preprocessing, and validating Boston Police Stations data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ingest_stations import download_stations_data, validate_stations_data
from preprocess_stations import preprocess_stations_data, generate_statistics
from bias_detection_stations import StationsBiasDetector

# Default arguments
default_args = {
    'owner': 'boston-pulse-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'police_stations_pipeline',
    default_args=default_args,
    description='Automated pipeline for Boston Police Stations data',
    schedule_interval='@monthly',  # Stations data changes infrequently
    start_date=datetime(2026, 2, 18),
    catchup=False,
    tags=['boston-pulse', 'police-stations', 'mlops'],
)


# Task 1: Download police stations data
def download_data_task():
    """Download latest police stations data from Boston Open Data Portal"""
    print("Starting police stations data download...")
    file_path = download_stations_data()
    print(f"Download complete: {file_path}")
    return file_path


download_task = PythonOperator(
    task_id='download_stations_data',
    python_callable=download_data_task,
    dag=dag,
)


# Task 2: Validate downloaded data
def validate_data_task():
    """Validate the downloaded police stations data"""
    print("Starting data validation...")
    validation = validate_stations_data("data/raw/police_stations.csv")

    if validation['critical_issues']:
        raise ValueError(f"Data validation failed: {validation['critical_issues']}")

    print(f"Validation passed: {validation['total_rows']} stations found")
    return validation


validate_task = PythonOperator(
    task_id='validate_stations_data',
    python_callable=validate_data_task,
    dag=dag,
)


# Task 3: Preprocess data
def preprocess_data_task():
    """Preprocess and feature-engineer police stations data"""
    print("Starting data preprocessing...")
    output_file = preprocess_stations_data()
    print(f"Preprocessing complete: {output_file}")
    return output_file


preprocess_task = PythonOperator(
    task_id='preprocess_stations_data',
    python_callable=preprocess_data_task,
    dag=dag,
)


# Task 4: Generate statistics
def generate_stats_task():
    """Generate statistics about processed data"""
    print("Generating data statistics...")
    stats = generate_statistics()

    print("\n=== Pipeline Statistics ===")
    print(f"Total stations: {stats['total_stations']}")
    print(f"Zone distribution: {stats['zone_distribution']}")
    print(f"Size distribution: {stats['size_distribution']}")
    print(f"Avg distance from center: {stats['avg_dist_from_center_km']:.2f} km")

    return stats


stats_task = PythonOperator(
    task_id='generate_statistics',
    python_callable=generate_stats_task,
    dag=dag,
)


# Task 5: Run bias detection
def bias_detection_task():
    """Run bias detection and generate report"""
    print("Running bias detection analysis...")
    detector = StationsBiasDetector()
    report = detector.generate_bias_report()
    detector.visualize_bias()
    print("Bias detection complete. Report saved.")
    return report


bias_task = PythonOperator(
    task_id='bias_detection',
    python_callable=bias_detection_task,
    dag=dag,
)


# Pipeline flow
download_task >> validate_task >> preprocess_task >> stats_task >> bias_task