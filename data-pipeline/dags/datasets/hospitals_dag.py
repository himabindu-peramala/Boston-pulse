"""
Boston Pulse - Hospital Locations DAG

Complete Airflow DAG for the Hospital Locations data pipeline.

Pipeline Stages:
    1. Ingest: Fetch data from Analyze Boston API
    2. Detect Anomalies: Check for data quality issues
    3. Validate Raw: Check raw data schema and quality
    4. Preprocess: Clean and transform data
    5. Validate Processed: Check processed data schema and quality
    6. Build Features: Create engineered features
    7. Validate Features: Check feature schema and quality
    8. Detect Drift: Check for distribution changes
    9. Check Fairness: Evaluate fairness metrics
    10. Mitigate Bias: Apply mitigation strategies
    11. Generate Model Card: Create dataset documentation
    12. Update Watermark: Track last successful run
    13. Record Lineage: Capture artifact generations
    14. Pipeline Complete: Send summary alert
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


# =============================================================================
# Task Functions
# =============================================================================


def ingest_data(**context) -> dict:
    """Ingest hospital locations data."""
    from dags.utils import write_data
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


def detect_anomalies(**context) -> dict:
    """Detect anomalies in raw data."""
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
    """Preprocess raw hospital data."""
    from dags.utils.alerting import alert_preprocessing_complete
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.hospitals.preprocess import HospitalPreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)
    preprocessor = HospitalPreprocessor()
    result = preprocessor.run(raw_df, execution_date)
    df = preprocessor.get_data()
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
    """Build features from processed data."""
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
    """Detect data drift (stubbed for yearly reference)."""
    return {"drift_detected": False, "message": "Reference dataset"}


def check_fairness(**context) -> dict:
    """Standardized fairness check."""
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
    result = checker.evaluate_fairness(df, DATASET, dimensions=["neighborhood"])

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

    if not result.passes_fairness_gate:
        raise FairnessViolationError(result)

    return {
        "passes_fairness_gate": result.passes_fairness_gate,
        "has_violations": result.has_violations,
        "violation_count": len(result.violations),
        "slices_evaluated": result.slices_evaluated,
        "violations": [
            {
                "metric": v.metric.value,
                "severity": v.severity.value,
                "dimension": v.dimension,
                "message": v.message,
            }
            for v in result.violations
        ],
    }


def mitigate_bias_task(**context) -> dict:
    """Standardized bias mitigation task logic."""
    from dags.utils.gcs_io import read_data, write_data
    from src.bias.bias_mitigator import BiasMitigator, MitigationStrategy

    execution_date = context["ds"]
    ti = context["ti"]
    fairness_xcom = ti.xcom_pull(task_ids="check_fairness")

    if not fairness_xcom or not fairness_xcom.get("has_violations"):
        return {
            "mitigation_applied": False,
            "reason": "No fairness violations detected",
        }

    df = read_data(DATASET, "processed", execution_date)

    if df.empty:
        return {
            "mitigation_applied": False,
            "reason": "Empty DataFrame",
        }

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
        df=df, fairness_result=slim_result, dimension="neighborhood", strategy=strategy
    )

    print(mitigation_result.report())
    output_path = write_data(mitigation_result.mitigated_df, DATASET, "mitigated", execution_date)

    return {
        "mitigation_applied": True,
        "strategy": strategy.value,
        "rows_before": mitigation_result.rows_before,
        "rows_after": mitigation_result.rows_after,
        "slices_improved": mitigation_result.total_improved,
        "output_path": output_path,
        "actions": [
            {
                "slice_value": a.slice_value,
                "improvement_pct": round(a.improvement_pct, 2),
            }
            for a in mitigation_result.actions
        ],
    }


def generate_model_card(**context) -> dict:
    """Generate model card."""
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
        description="Hospital Locations documentation",
        validation_result=ti.xcom_pull(task_ids="validate_processed"),
        drift_result=ti.xcom_pull(task_ids="detect_drift"),
        fairness_result=ti.xcom_pull(task_ids="check_fairness"),
        mitigation_result=ti.xcom_pull(task_ids="mitigate_bias"),
    )
    output_path = generator.save_model_card(card, format="both")
    return {"model_card_generated": True, "output_path": output_path}


def update_watermark(**context) -> dict:
    """Update watermark after successful run."""
    from dags.utils import set_watermark

    execution_date = context["ds"]
    set_watermark(DATASET, datetime.strptime(execution_date, "%Y-%m-%d"), execution_date)
    return {"watermark_updated": True, "new_watermark": execution_date}


def record_lineage(**context) -> dict:
    """Record data lineage."""
    from dags.utils.lineage_utils import record_pipeline_lineage

    return record_pipeline_lineage(dataset=DATASET, dag_id=DAG_ID, context=context)


def pipeline_complete(**context) -> dict:
    """Final reporting task."""
    from dags.utils.alerting import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context["ti"]

    ingest_result = ti.xcom_pull(task_ids="ingest_data")
    preprocess_result = ti.xcom_pull(task_ids="preprocess_data")
    features_result = ti.xcom_pull(task_ids="build_features")
    lineage_result = ti.xcom_pull(task_ids="record_lineage")

    stats = {
        "rows_ingested": ingest_result.get("rows_fetched", 0) if ingest_result else 0,
        "rows_processed": preprocess_result.get("rows_output", 0) if preprocess_result else 0,
        "features_generated": features_result.get("features_computed", 0) if features_result else 0,
        "lineage_recorded": lineage_result.get("lineage_recorded", False)
        if lineage_result
        else False,
    }

    duration = (
        (ingest_result.get("duration_seconds", 0) if ingest_result else 0)
        + (preprocess_result.get("duration_seconds", 0) if preprocess_result else 0)
        + (features_result.get("duration_seconds", 0) if features_result else 0)
    )

    alert_pipeline_complete(
        dataset=DATASET,
        execution_date=execution_date,
        duration_seconds=duration,
        stats=stats,
        dag_id=DAG_ID,
    )
    return {"status": "complete", "stats": stats, "duration": duration}


# =============================================================================
# DAG Definition
# =============================================================================


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Hospital Locations data pipeline with standard tasks",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["healthcare", "reference"],
    max_active_runs=1,
) as dag:
    from dags.utils.callbacks import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        on_failure_callback=on_task_failure,
    )
    t_detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        on_failure_callback=on_task_failure,
    )
    t_validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=validate_raw,
        on_failure_callback=on_task_failure,
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
        task_id="detect_drift",
        python_callable=detect_drift,
        on_failure_callback=on_task_failure,
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
        task_id="pipeline_complete",
        python_callable=pipeline_complete,
        trigger_rule="all_success",
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
