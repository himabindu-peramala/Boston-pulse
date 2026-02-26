"""
Boston Pulse - Crime Dataset DAG

Complete Airflow DAG for the crime incident data pipeline.
This serves as the reference implementation for all other dataset DAGs.

Pipeline Stages:
    1. Ingest: Fetch data from Analyze Boston API
    2. Detect Anomalies: Check for data quality and statistical outliers
    3. Validate Raw: Check raw data schema and quality
    4. Preprocess: Clean and transform data
    5. Validate Processed: Check processed data schema and quality
    6. Build Features: Create aggregated features
    7. Validate Features: Check feature schema and quality
    8. Detect Drift: Check for data distribution changes
    9. Check Fairness: Evaluate fairness metrics
    10. Mitigate Bias: Apply corrections if needed
    11. Generate Model Card: Create dataset documentation
    12. Update Watermark: Track incremental data windows
    13. Record Lineage: Trace data provenance
    14. Pipeline Complete: Final notification
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "crime_pipeline"
DATASET = "crime"
SCHEDULE = "0 2 * * *"  # Daily at 2 AM UTC
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
# Task Functions
# =============================================================================


def ingest_data(**context) -> dict:
    """Ingest crime data from Analyze Boston API."""
    from dags.utils.gcs_io import write_data
    from dags.utils.watermark import get_effective_watermark
    from src.datasets.crime import CrimeIngester

    execution_date = context["ds"]
    watermark = get_effective_watermark(DATASET)

    ingester = CrimeIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path

    return result.to_dict()


def detect_anomalies(**context) -> dict:
    """Detect anomalies in raw data after ingestion."""
    from dags.utils.alerting import alert_anomaly_detected
    from dags.utils.gcs_io import read_data
    from src.validation import AnomalyDetector

    execution_date = context["ds"]
    df = read_data(DATASET, "raw", execution_date)
    detector = AnomalyDetector()
    result = detector.detect_anomalies(df, DATASET)

    if result.has_critical_anomalies:
        for anomaly in result.critical_anomalies:
            alert_anomaly_detected(
                dataset=DATASET,
                anomaly_type=anomaly.type.value,
                details=anomaly.message,
                severity="critical",
                execution_date=execution_date,
                dag_id=DAG_ID,
            )
    return {
        "has_anomalies": result.has_anomalies,
        "anomaly_count": len(result.anomalies),
        "critical_count": len(result.critical_anomalies),
    }


def validate_raw(**context) -> dict:
    """Validate raw data against schema."""
    from dags.utils.alerting import alert_validation_failure
    from dags.utils.gcs_io import read_data
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
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "raw"}


def preprocess_data(**context) -> dict:
    """Preprocess raw crime data."""
    from dags.utils.alerting import alert_preprocessing_complete
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.crime import CrimePreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)
    preprocessor = CrimePreprocessor()
    result = preprocessor.run(raw_df, execution_date)
    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    if result.rows_dropped > 0:
        drop_rate = result.rows_dropped / result.rows_input if result.rows_input > 0 else 0
        if drop_rate > 0.1:
            alert_preprocessing_complete(
                dataset=DATASET,
                rows_input=result.rows_input,
                rows_output=result.rows_output,
                rows_dropped=result.rows_dropped,
                execution_date=execution_date,
                dag_id=DAG_ID,
            )
    return result.to_dict()


def validate_processed(**context) -> dict:
    """Validate processed data against schema."""
    from dags.utils.alerting import alert_validation_failure
    from dags.utils.gcs_io import read_data
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
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "processed"}


def build_features(**context) -> dict:
    """Build crime features from processed data."""
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.crime import CrimeFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)
    builder = CrimeFeatureBuilder()
    result = builder.run(processed_df, execution_date)
    df = builder.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "features", execution_date)
        result.output_path = output_path
    return result.to_dict()


def validate_features(**context) -> dict:
    """Validate features against schema."""
    from dags.utils.alerting import alert_validation_failure
    from dags.utils.gcs_io import read_data
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
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "features"}


def detect_drift(**context) -> dict:
    """Detect data drift in processed data."""
    from dags.utils.alerting import alert_drift_detected
    from dags.utils.gcs_io import read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]
    current_df = read_data(DATASET, "processed", execution_date)

    try:
        from datetime import datetime, timedelta

        prev_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=1)
        reference_df = read_data(DATASET, "processed", prev_date.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        return {"drift_detected": False, "message": "No reference data"}

    detector = DriftDetector()
    result = detector.detect_drift(current_df, reference_df, DATASET)
    if result.has_drift:
        alert_drift_detected(
            dataset=DATASET,
            drifted_features=result.drifted_features,
            psi_scores={f: r.psi for f, r in result.feature_results.items()},
            severity="warning",
            execution_date=execution_date,
            dag_id=DAG_ID,
        )
    return {"drift_detected": result.has_drift, "drifted_features": result.drifted_features}


def check_fairness(**context) -> dict:
    """Standardized fairness check with detailed reporting."""
    from dags.utils.alerting import alert_fairness_violation
    from dags.utils.gcs_io import read_data
    from src.bias.fairness_checker import FairnessChecker, FairnessViolationError

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if df.empty:
        return {
            "passes_fairness_gate": True,
            "has_violations": False,
            "violation_count": 0,
            "message": "No data to evaluate",
        }

    checker = FairnessChecker()
    result = checker.evaluate_fairness(df, DATASET, dimensions=["district"])

    report = checker.create_fairness_report(result)
    print(report)

    if result.has_violations:
        violations_payload = [
            {
                "metric": v.metric.value,
                "severity": v.severity.value,
                "dimension": v.dimension,
                "slice_value": str(v.slice_value),
                "expected": round(v.expected, 4),
                "actual": round(v.actual, 4),
                "disparity": round(v.disparity, 4),
                "message": v.message,
            }
            for v in result.violations
        ]
        alert_fairness_violation(
            dataset=DATASET,
            violations=violations_payload,
            execution_date=execution_date,
            dag_id=DAG_ID,
        )

    if result.has_critical_violations:
        print(f"CRITICAL FAIRNESS VIOLATIONS ({len(result.critical_violations)}):")
        for v in result.critical_violations:
            print(f"  [{v.dimension}={v.slice_value}] {v.message}")

    if not result.passes_fairness_gate:
        raise FairnessViolationError(result)

    return {
        "passes_fairness_gate": result.passes_fairness_gate,
        "has_violations": result.has_violations,
        "has_critical_violations": result.has_critical_violations,
        "violation_count": len(result.violations),
        "critical_count": len(result.critical_violations),
    }


def mitigate_bias_task(**context) -> dict:
    """Standardized bias mitigation task logic."""
    from dags.utils.gcs_io import read_data, write_data
    from src.bias.bias_mitigator import BiasMitigator, MitigationStrategy

    execution_date = context["ds"]
    ti = context["ti"]
    fairness_xcom = ti.xcom_pull(task_ids="check_fairness")

    if not fairness_xcom or not fairness_xcom.get("has_violations"):
        return {"mitigation_applied": False, "reason": "No violations"}

    df = read_data(DATASET, "processed", execution_date)
    if df.empty:
        return {"mitigation_applied": False, "reason": "Empty DF"}

    from dataclasses import dataclass

    from src.bias.fairness_checker import FairnessMetric, FairnessSeverity, FairnessViolation

    @dataclass
    class _SlimFairnessResult:
        dataset: str
        violations: list

        @property
        def has_violations(self):
            return bool(self.violations)

    slim_violations = [
        FairnessViolation(
            metric=FairnessMetric(v["metric"]),
            severity=FairnessSeverity(v["severity"]),
            dimension=v["dimension"],
            slice_value=v.get("slice_value", ""),
            expected=v.get("expected", 0.0),
            actual=v.get("actual", 0.0),
            disparity=v.get("disparity", 0.0),
            message=v.get("message", ""),
        )
        for v in fairness_xcom.get("violations", [])
    ]
    slim_result = _SlimFairnessResult(dataset=DATASET, violations=slim_violations)

    strategy = MitigationStrategy.REWEIGHTING
    mitigator = BiasMitigator()
    mitigation_result = mitigator.mitigate(
        df=df, fairness_result=slim_result, dimension="district", strategy=strategy
    )
    write_data(mitigation_result.mitigated_df, DATASET, "mitigated", execution_date)

    return {
        "mitigation_applied": True,
        "strategy": strategy.value,
        "rows_before": mitigation_result.rows_before,
    }


def generate_model_card(**context) -> dict:
    from dags.utils.gcs_io import read_data
    from src.bias import ModelCardGenerator

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    ti = context["ti"]

    generator = ModelCardGenerator()
    card = generator.generate_model_card(
        dataset=DATASET,
        df=df,
        version=execution_date.replace("-", ""),
        description="Boston crime incident reports documentation",
        validation_result=ti.xcom_pull(task_ids="validate_processed"),
        drift_result=ti.xcom_pull(task_ids="detect_drift"),
        fairness_result=ti.xcom_pull(task_ids="check_fairness"),
        mitigation_result=ti.xcom_pull(task_ids="mitigate_bias"),
    )
    output_path = generator.save_model_card(card, format="both")
    return {"model_card_generated": True, "output_path": output_path}


def record_lineage(**context) -> dict:
    from dags.utils.lineage_utils import record_pipeline_lineage

    return record_pipeline_lineage(dataset=DATASET, dag_id=DAG_ID, context=context)


def update_watermark(**context) -> dict:
    from dags.utils.gcs_io import read_data
    from dags.utils.watermark import set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    if "occurred_on_date" in df.columns and len(df) > 0:
        import pandas as pd

        max_date = pd.to_datetime(df["occurred_on_date"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True, "new_watermark": max_date.isoformat()}
    return {"watermark_updated": False}


def pipeline_complete(**context) -> dict:
    from dags.utils.alerting import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context["ti"]
    ingest_result = ti.xcom_pull(task_ids="ingest_data")
    stats = {"rows_ingested": ingest_result.get("rows_fetched", 0) if ingest_result else 0}
    alert_pipeline_complete(
        dataset=DATASET,
        execution_date=execution_date,
        duration_seconds=0,
        stats=stats,
        dag_id=DAG_ID,
    )
    return {"status": "complete"}


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Crime incident data pipeline with standard compliance steps",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["crime", "reference"],
    max_active_runs=1,
) as dag:
    from dags.utils.callbacks import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t_ingest = PythonOperator(
        task_id="ingest_data", python_callable=ingest_data, on_failure_callback=on_task_failure
    )
    t_detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        on_failure_callback=on_task_failure,
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
    t_detect_drift = PythonOperator(
        task_id="detect_drift", python_callable=detect_drift, on_failure_callback=on_task_failure
    )
    t_check_fairness = PythonOperator(
        task_id="check_fairness",
        python_callable=check_fairness,
        on_failure_callback=on_task_failure,
    )
    t_mitigate_bias = PythonOperator(
        task_id="mitigate_bias",
        python_callable=mitigate_bias_task,
        on_failure_callback=on_task_failure,
    )
    t_generate_model_card = PythonOperator(
        task_id="generate_model_card",
        python_callable=generate_model_card,
        on_failure_callback=on_task_failure,
    )
    t_update_watermark = PythonOperator(
        task_id="update_watermark",
        python_callable=update_watermark,
        on_failure_callback=on_task_failure,
    )
    t_record_lineage = PythonOperator(
        task_id="record_lineage",
        python_callable=record_lineage,
        on_failure_callback=on_task_failure,
    )
    t_pipeline_complete = PythonOperator(
        task_id="pipeline_complete", python_callable=pipeline_complete, trigger_rule="all_success"
    )

    (
        t_ingest
        >> t_detect_anomalies
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> t_build_features
        >> t_validate_features
        >> [t_detect_drift, t_check_fairness]
        >> t_mitigate_bias
        >> t_generate_model_card
        >> t_update_watermark
        >> t_record_lineage
        >> t_pipeline_complete
    )
