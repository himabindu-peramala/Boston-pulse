"""
Boston Pulse - BERDO Dataset DAG

Complete Airflow DAG for the BERDO building energy and emissions data pipeline.

Pipeline Stages:
    1. Ingest: Download BERDO Excel data from Analyze Boston
    2. Detect Anomalies: Check for data quality issues
    3. Validate Raw: Check raw data schema and quality
    4. Preprocess: Clean and transform data
    5. Validate Processed: Check processed data schema and quality
    6. Build Features: Create engineered features
    7. Validate Features: Check feature schema and quality
    8. Detect Drift: Check for data distribution changes
    9. Check Fairness: Evaluate fairness metrics
    10. Mitigate Bias: Apply bias mitigation strategies
    11. Generate Model Card: Create dataset documentation
    12. Update Watermark: Track last successful run
    13. Record Lineage: Capture GCS artifact generations
    14. Pipeline Complete: Send summary alert
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "berdo_pipeline"
DATASET = "berdo"
SCHEDULE = "@yearly"
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
        "anomalies_by_type": {k.value: len(v) for k, v in result.anomalies_by_type.items()},
    }


def ingest_data(**context) -> dict:
    """Ingest BERDO data from Analyze Boston via direct file download."""
    from dags.utils import write_data
    from src.datasets.berdo.ingest import BerdoIngester

    execution_date = context["ds"]

    ingester = BerdoIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=None)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_raw(**context) -> dict:
    """Validate raw BERDO data against schema."""
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

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "raw",
    }


def preprocess_data(**context) -> dict:
    """Preprocess raw BERDO data."""
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.berdo.preprocess import BerdoPreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)

    preprocessor = BerdoPreprocessor()
    result = preprocessor.run(raw_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Preprocessing failed: {result.error_message}")

    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_processed(**context) -> dict:
    """Validate processed BERDO data against schema."""
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

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "processed",
    }


def build_features(**context) -> dict:
    """Build BERDO features from processed data."""
    from dags.utils.gcs_io import read_data, write_data
    from src.datasets.berdo.features import BerdoFeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)

    builder = BerdoFeatureBuilder()
    result = builder.run(processed_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Feature building failed: {result.error_message}")

    df = builder.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "features", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_features(**context) -> dict:
    """Validate BERDO features against schema."""
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

    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "stage": "features",
    }


def detect_drift(**context) -> dict:
    """Detect data drift in processed BERDO data."""
    from dags.utils.alerting import alert_drift_detected
    from dags.utils.gcs_io import read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]
    current_df = read_data(DATASET, "processed", execution_date)

    try:
        from datetime import datetime, timedelta

        prev_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=365)
        reference_df = read_data(DATASET, "processed", prev_date.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        return {
            "drift_detected": False,
            "message": "No reference data available for drift detection",
            "drifted_features": [],
        }

    detector = DriftDetector()
    result = detector.detect_drift(current_df, reference_df, DATASET)

    if result.has_drift:
        severity = "warning" if result.severity.value == "warning" else "critical"
        alert_drift_detected(
            dataset=DATASET,
            drifted_features=result.drifted_features,
            psi_scores={f: r.psi_score for f, r in result.feature_results.items()},
            severity=severity,
            execution_date=execution_date,
            dag_id=DAG_ID,
        )

    return {
        "drift_detected": result.has_drift,
        "severity": result.severity.value,
        "drifted_features": result.drifted_features,
        "overall_psi": result.overall_psi,
    }


def check_fairness(**context) -> dict:
    """
    Check fairness metrics for BERDO dataset.

    Evaluates whether building emissions reporting is equitable
    across different property types and neighborhoods.
    """
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
    result = checker.evaluate_fairness(
        df=df,
        dataset=DATASET,
        outcome_column=None,
        dimensions=["property_type"],
    )

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
        "has_critical_violations": result.has_critical_violations,
        "violation_count": len(result.violations),
        "critical_count": len(result.critical_violations),
        "warning_count": len(result.warning_violations),
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
    """
    Apply bias mitigation when fairness violations are detected.

    Reads processed data, reconstructs violations from XCom, applies
    reweighting, and saves the mitigated DataFrame as the 'mitigated' stage.
    """
    from dataclasses import dataclass

    from dags.utils.gcs_io import read_data, write_data
    from src.bias.bias_mitigator import BiasMitigator, MitigationStrategy
    from src.bias.fairness_checker import (
        FairnessMetric,
        FairnessSeverity,
        FairnessViolation,
    )

    execution_date = context["ds"]
    ti = context["ti"]

    fairness_xcom: dict = ti.xcom_pull(task_ids="check_fairness")

    if not fairness_xcom or not fairness_xcom.get("has_violations"):
        return {
            "mitigation_applied": False,
            "reason": "No fairness violations detected",
            "rows_before": None,
            "rows_after": None,
        }

    df = read_data(DATASET, "processed", execution_date)

    if df.empty:
        return {
            "mitigation_applied": False,
            "reason": "Empty DataFrame — skipping mitigation",
        }

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
        df=df,
        fairness_result=slim_result,
        dimension="property_type",
        strategy=strategy,
    )

    print(mitigation_result.report())

    output_path = write_data(mitigation_result.mitigated_df, DATASET, "mitigated", execution_date)

    return {
        "mitigation_applied": True,
        "strategy": strategy.value,
        "dimension": "property_type",
        "rows_before": mitigation_result.rows_before,
        "rows_after": mitigation_result.rows_after,
        "slices_improved": mitigation_result.total_improved,
        "total_slices": len(mitigation_result.actions),
        "output_path": output_path,
        "weight_range": mitigation_result.tradeoffs.get("weight_range"),
        "actions": [
            {
                "slice_value": a.slice_value,
                "before_disparity": round(a.before_disparity, 4),
                "after_disparity": round(a.after_disparity, 4),
                "improvement_pct": round(a.improvement_pct, 2),
                "details": a.details,
            }
            for a in mitigation_result.actions
        ],
    }


def generate_model_card(**context) -> dict:
    """Generate model card for BERDO dataset."""
    from dags.utils.gcs_io import read_data
    from src.bias import ModelCardGenerator

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    ti = context["ti"]
    validation_result = ti.xcom_pull(task_ids="validate_processed")
    drift_result = ti.xcom_pull(task_ids="detect_drift")
    fairness_result = ti.xcom_pull(task_ids="check_fairness")
    mitigation_result = ti.xcom_pull(task_ids="mitigate_bias")

    generator = ModelCardGenerator()
    card = generator.generate_model_card(
        dataset=DATASET,
        df=df,
        version=execution_date.replace("-", ""),
        description="Boston BERDO building energy and emissions data from Environment Department",
        validation_result=validation_result,
        drift_result=drift_result,
        fairness_result=fairness_result,
        mitigation_result=mitigation_result,
    )

    output_path = generator.save_model_card(card, format="both")

    return {
        "model_card_generated": True,
        "output_path": output_path,
        "version": card.version,
    }


def update_watermark(**context) -> dict:
    """Update watermark after successful pipeline completion."""
    from dags.utils import set_watermark

    execution_date = context["ds"]
    set_watermark(DATASET, datetime.strptime(execution_date, "%Y-%m-%d"), execution_date)

    return {
        "watermark_updated": True,
        "new_watermark": execution_date,
    }


def record_lineage(**context) -> dict:
    """
    Record data lineage for this pipeline run.
    Captures exact GCS generation numbers for all artifacts,
    enabling precise point-in-time recovery and debugging.
    """
    from dags.utils.lineage_utils import record_pipeline_lineage

    return record_pipeline_lineage(
        dataset=DATASET,
        dag_id=DAG_ID,
        context=context,
    )


def pipeline_complete(**context) -> dict:
    """Final task to mark pipeline completion and send summary alert."""
    from dags.utils.alerting import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context["ti"]

    ingest_result = ti.xcom_pull(task_ids="ingest_data")
    preprocess_result = ti.xcom_pull(task_ids="preprocess_data")
    features_result = ti.xcom_pull(task_ids="build_features")

    stats = {
        "rows_ingested": ingest_result.get("rows_fetched", 0) if ingest_result else 0,
        "rows_processed": preprocess_result.get("rows_output", 0) if preprocess_result else 0,
        "features_generated": features_result.get("features_computed", 0) if features_result else 0,
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

    return {"status": "complete", "stats": stats}


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="BERDO building energy and emissions data pipeline with validation and fairness checks",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["boston-pulse", "berdo", "emissions", "housing"],
    max_active_runs=1,
) as dag:
    from dags.utils.callbacks import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

    t_detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        on_failure_callback=on_task_failure,
    )
    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
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

    # Define task dependencies
    # Define task dependencies: ingest -> [anomalies, validate] -> preprocess -> validate -> features -> validate
    #                          -> [drift, fairness] -> mitigate -> model_card -> watermark -> lineage -> complete
    (
        t_ingest
        >> [t_detect_anomalies, t_validate_raw]
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
