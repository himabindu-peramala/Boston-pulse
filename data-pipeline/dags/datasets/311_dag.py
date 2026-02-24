"""
Boston Pulse - 311 Service Requests DAG

Complete Airflow DAG for the 311 service request data pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DAG Configuration
DAG_ID = "311_pipeline"
DATASET = "service_311"
SCHEDULE = "0 3 * * *"  # Daily at 3 AM UTC
START_DATE = datetime(2025, 1, 1)

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
    """Ingest 311 data from Analyze Boston API."""
    from airflow.exceptions import AirflowSkipException

    from dags.utils import get_effective_watermark, write_data
    from src.datasets.service_311 import Service311Ingester

    execution_date = context["ds"]
    watermark = get_effective_watermark(DATASET)

    ingester = Service311Ingester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)

    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")

    df = ingester.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "raw", execution_date)
        result.output_path = output_path
    else:
        raise AirflowSkipException(f"No new data found for {DATASET} on {execution_date}")

    return result.to_dict()


def validate_raw(**context) -> dict:
    """Validate raw data against schema."""
    from dags.utils import alert_validation_failure, read_data
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
        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Raw validation failed: {errors[:5]}")

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def preprocess_data(**context) -> dict:
    """Preprocess raw 311 data."""
    from dags.utils import read_data, write_data
    from src.datasets.service_311 import Service311Preprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)

    preprocessor = Service311Preprocessor()
    result = preprocessor.run(raw_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Preprocessing failed: {result.error_message}")

    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "processed", execution_date)
        result.output_path = output_path

    return result.to_dict()


def validate_processed(**context) -> dict:
    """Validate processed data against schema."""
    from dags.utils import alert_validation_failure, read_data
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
        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Processed validation failed: {errors[:5]}")

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def build_features(**context) -> dict:
    """Build 311 features from processed data."""
    from dags.utils import read_data, write_data
    from src.datasets.service_311 import Service311FeatureBuilder

    execution_date = context["ds"]
    processed_df = read_data(DATASET, "processed", execution_date)

    builder = Service311FeatureBuilder()
    result = builder.run(processed_df, execution_date)

    if not result.success:
        raise RuntimeError(f"Feature building failed: {result.error_message}")

    df = builder.get_data()
    if df is not None and len(df) > 0:
        output_path = write_data(df, DATASET, "features", execution_date)
        result.output_path = output_path

    return result.to_dict()


def check_fairness(**context) -> dict:
    """
    Check fairness metrics for the 311 dataset.

    Evaluates representation and outcome disparity across neighborhoods.
    ALERTS on violations but does NOT fail the pipeline.
    """
    from dags.utils import alert_fairness_violation, read_data
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
        dimensions=["neighborhood"],  # 311-specific: neighborhood vs district for crime
    )

    print(checker.create_fairness_report(result))

    if result.has_violations:
        alert_fairness_violation(
            dataset=DATASET,
            violations=[
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
            ],
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
        "warning_count": len(result.warning_violations),
        "slices_evaluated": result.slices_evaluated,
        "violations": [
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
        ],
    }


def mitigate_bias_task(**context) -> dict:
    """
    Apply bias mitigation when fairness violations are detected.

    Reads processed data, reconstructs violations from XCom, applies
    reweighting, and saves the mitigated DataFrame as the 'mitigated' stage.
    """
    from dataclasses import dataclass

    from dags.utils import read_data, write_data
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
        dimension="neighborhood",  # 311-specific dimension
        strategy=strategy,
    )

    print(mitigation_result.report())

    output_path = write_data(mitigation_result.mitigated_df, DATASET, "mitigated", execution_date)

    return {
        "mitigation_applied": True,
        "strategy": strategy.value,
        "dimension": "neighborhood",
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


def detect_drift(**context) -> dict:
    """
    Detect data drift in processed 311 data.

    Compares today's processed data against the previous day's as reference.
    Skips gracefully if no reference data is available (e.g. first run).
    """
    from dags.utils import alert_drift_detected, read_data
    from src.validation import DriftDetector

    execution_date = context["ds"]

    current_df = read_data(DATASET, "processed", execution_date)

    try:
        from datetime import datetime, timedelta

        prev_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=1)
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
            psi_scores={f: r.psi for f, r in result.feature_results.items()},
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


def validate_features(**context) -> dict:
    """Validate features against schema."""
    from dags.utils import alert_validation_failure, read_data
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
        if enforcer.config.validation.quality_schema.strict_mode:
            raise RuntimeError(f"Features validation failed: {errors[:5]}")

    return {"is_valid": result.is_valid, "error_count": len(result.errors)}


def update_watermark(**context) -> dict:
    """Update watermark after success."""
    from dags.utils import read_data, set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)

    if "open_date" in df.columns and len(df) > 0:
        import pandas as pd

        max_date = pd.to_datetime(df["open_date"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True, "new_watermark": max_date.isoformat()}
    return {"watermark_updated": False}


def record_lineage(**context) -> dict:
    """
    Record data lineage for this pipeline run.

    Captures exact GCS generation numbers for all artifacts,
    enabling precise point-in-time recovery and debugging.
    """
    from dags.utils import record_pipeline_lineage

    return record_pipeline_lineage(
        dataset=DATASET,
        dag_id=DAG_ID,
        context=context,
    )


def pipeline_complete(**context) -> dict:
    """Final task."""
    from dags.utils import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context["ti"]
    ingest_result = ti.xcom_pull(task_ids="ingest_data")
    lineage_result = ti.xcom_pull(task_ids="record_lineage")
    stats = {
        "rows_ingested": ingest_result.get("rows_fetched", 0) if ingest_result else 0,
        "lineage_recorded": (
            lineage_result.get("lineage_recorded", False) if lineage_result else False
        ),
    }
    alert_pipeline_complete(
        dataset=DATASET,
        execution_date=execution_date,
        duration_seconds=0,
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
    description="311 Service Request data pipeline",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["311", "dataset"],
) as dag:
    from dags.utils import on_dag_failure, on_dag_success, on_task_failure

    dag.on_failure_callback = on_dag_failure
    dag.on_success_callback = on_dag_success

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
    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
        on_failure_callback=on_task_failure,
    )
    t_validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
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

    # Flow: ingest -> validate -> preprocess -> validate -> features -> validate
    #       -> [drift, fairness] -> mitigate -> watermark -> lineage -> complete
    (
        t_ingest
        >> t_validate_raw
        >> t_preprocess
        >> t_validate_processed
        >> t_build_features
        >> t_validate_features
        >> [t_detect_drift, t_check_fairness]
        >> t_mitigate_bias
        >> t_update_watermark
        >> t_record_lineage
        >> t_pipeline_complete
    )
