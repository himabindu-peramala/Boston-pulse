"""
Boston Pulse - Fire Dataset DAG

Complete Airflow DAG for the fire incident data pipeline.

Pipeline Stages:
    1. Ingest
    2. Validate Raw
    3. Preprocess
    4. Validate Processed
    5. Detect Drift
    6. Check Fairness
    7. Generate Model Card
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# =============================================================================
# DAG Configuration
# =============================================================================

DAG_ID = "fire_pipeline"
DATASET = "fire"
SCHEDULE = "@monthly"
START_DATE = datetime(2024, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=1),
}

# =============================================================================
# Task Functions
# =============================================================================


def ingest_data(**context):
    from dags.utils import get_effective_watermark, write_data
    from src.datasets.fire import FireIngester

    execution_date = context["ds"]

    watermark = get_effective_watermark(DATASET)

    ingester = FireIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()

    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_raw(**context):
    from dags.utils import read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]

    df = read_data(DATASET, "raw", execution_date)

    enforcer = SchemaEnforcer()
    result = enforcer.validate_raw(df, DATASET)

    if not result.is_valid and enforcer.config.validation.schema.strict_mode:
        raise RuntimeError("Raw validation failed")

    return result.to_dict()


def preprocess_data(**context):
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
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_processed(**context):
    from dags.utils import read_data
    from src.validation import SchemaEnforcer

    execution_date = context["ds"]

    df = read_data(DATASET, "processed", execution_date)

    enforcer = SchemaEnforcer()
    result = enforcer.validate_processed(df, DATASET)

    if not result.is_valid and enforcer.config.validation.schema.strict_mode:
        raise RuntimeError("Processed validation failed")

    return result.to_dict()


def detect_drift(**context):
    from dags.utils import read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]

    current_df = read_data(DATASET, "processed", execution_date)

    detector = DriftDetector()
    result = detector.detect_drift(current_df, None, DATASET)

    return result.to_dict()


def check_fairness(**context):
    from dags.utils import read_data
    from src.bias.fairness_checker import FairnessChecker

    execution_date = context["ds"]

    df = read_data(DATASET, "processed", execution_date)

    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        df=df,
        dataset=DATASET,
        outcome_column=None,
        dimensions=["district"],
    )

    if not result.passes_fairness_gate:
        raise RuntimeError("Fairness gate failed")

    return result.to_dict()


def generate_model_card(**context):
    from dags.utils import read_data
    from src.bias import ModelCardGenerator

    execution_date = context["ds"]

    df = read_data(DATASET, "processed", execution_date)

    generator = ModelCardGenerator()

    card = generator.generate_model_card(
        dataset=DATASET,
        df=df,
        version=execution_date.replace("-", ""),
        description="Boston fire incident reports",
    )

    output_path = generator.save_model_card(card, format="both")

    return {"model_card_generated": True, "output_path": output_path}


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Fire incident data pipeline",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["fire", "dataset"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    t_validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=validate_raw,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t_validate_processed = PythonOperator(
        task_id="validate_processed",
        python_callable=validate_processed,
    )

    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    t_check_fairness = PythonOperator(
        task_id="check_fairness",
        python_callable=check_fairness,
    )

    t_generate_model_card = PythonOperator(
        task_id="generate_model_card",
        python_callable=generate_model_card,
    )

    (
        t_ingest
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> [t_detect_drift, t_check_fairness]
        >> t_generate_model_card
    )