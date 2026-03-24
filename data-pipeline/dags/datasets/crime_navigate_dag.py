"""
Navigate Crime Scoring DAG — every 3 days at 2 AM UTC.

Ingestion: watermark-based, only new incidents since last run.
Features: computed from last 90 days of processed data in GCS.
All paths under gs://boston-pulse/navigate/. Independent of crime_dag.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "crime_navigate_pipeline"
DATASET = "crime_navigate"
SCHEDULE = "0 2 */3 * *"  # every 3 days at 2 AM UTC
START_DATE = datetime(2024, 1, 1)

default_args = {
    "owner": "boston-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def _read_features(execution_date: str):
    """Read features parquet (filename=features.parquet)."""
    from dags.utils.gcs_io import GCSDataIO

    gcs = GCSDataIO()
    return gcs.read_parquet(DATASET, "features", execution_date, filename="features.parquet")


def _write_features(df, execution_date: str):
    """Write features parquet (filename=features.parquet)."""
    from dags.utils.gcs_io import GCSDataIO

    gcs = GCSDataIO()
    return gcs.write_parquet(df, DATASET, "features", execution_date, filename="features.parquet")


def ingest_data(**context) -> dict:
    from dags.utils import get_watermark, write_data
    from src.datasets.crime_navigate import CrimeNavigateIngester

    execution_date = context["ds"]
    watermark = get_watermark(DATASET)
    ingester = CrimeNavigateIngester()
    result = ingester.run(execution_date=execution_date, watermark_start=watermark)
    if not result.success:
        raise RuntimeError(f"Ingestion failed: {result.error_message}")
    df = ingester.get_data()
    if df is not None and len(df) > 0:
        write_data(df, DATASET, "raw", execution_date)
    return result.to_dict()


def validate_raw(**context) -> dict:
    from dags.utils import alert_validation_failure, read_data
    from src.datasets.crime_navigate.validators import NavigateSchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "raw", execution_date)
    enforcer = NavigateSchemaEnforcer()
    result = enforcer.validate_navigate_raw(df)
    if not result.is_valid:
        alert_validation_failure(
            dataset=DATASET,
            stage="raw",
            errors=result.errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_raw",
        )
        raise RuntimeError(f"Raw validation failed: {result.errors[:5]}")
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "raw"}


def preprocess_data(**context) -> dict:
    from dags.utils import read_data, write_data
    from src.datasets.crime_navigate import CrimeNavigatePreprocessor

    execution_date = context["ds"]
    raw_df = read_data(DATASET, "raw", execution_date)
    preprocessor = CrimeNavigatePreprocessor()
    result = preprocessor.run(raw_df, execution_date)
    df = preprocessor.get_data()
    if df is not None and len(df) > 0:
        write_data(df, DATASET, "processed", execution_date)
    return result


def validate_processed(**context) -> dict:
    from dags.utils import alert_validation_failure, read_data
    from src.datasets.crime_navigate.validators import NavigateSchemaEnforcer

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    enforcer = NavigateSchemaEnforcer()
    result = enforcer.validate_navigate_processed(df)
    if not result.is_valid:
        alert_validation_failure(
            dataset=DATASET,
            stage="processed",
            errors=result.errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_processed",
        )
        raise RuntimeError(f"Processed validation failed: {result.errors[:5]}")
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "processed"}


def build_features(**context) -> dict:
    from datetime import timedelta

    import pandas as pd

    from dags.utils import read_data
    from dags.utils.gcs_io import GCSDataIO
    from src.datasets.crime_navigate.features import build_navigate_features
    from src.shared.config import get_dataset_config

    execution_date = context["ds"]
    cfg = get_dataset_config(DATASET)
    long_days = cfg.get("feature_windows", {}).get("long_days", 90)
    gcs = GCSDataIO()
    dates = gcs.list_execution_dates(DATASET, "processed")
    ref_dt = datetime.strptime(execution_date, "%Y-%m-%d")
    start_dt = ref_dt - timedelta(days=long_days)
    to_load = [d for d in dates if start_dt <= datetime.strptime(d, "%Y-%m-%d") <= ref_dt]
    to_load = sorted(to_load)[-long_days:]

    dfs = []
    for d in to_load:
        try:
            df = read_data(DATASET, "processed", d)
            if df is not None and len(df) > 0:
                dfs.append(df)
        except FileNotFoundError:
            continue
    if not dfs:
        raise RuntimeError(f"No processed data found for {DATASET} in last {long_days} days")
    processed_df = pd.concat(dfs, ignore_index=True)

    yesterday_features = None
    features_7d_ago = None
    try:
        prev = (ref_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday_features = _read_features(prev)
    except Exception:
        pass
    try:
        past7 = (ref_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        features_7d_ago = _read_features(past7)
    except Exception:
        pass

    features_df = build_navigate_features(
        processed_df,
        execution_date,
        reference_date=ref_dt,
        yesterday_features=yesterday_features,
        features_7d_ago=features_7d_ago,
    )
    if features_df is not None and len(features_df) > 0:
        _write_features(features_df, execution_date)
    return {
        "h3_cells": int(features_df["h3_index"].nunique()) if features_df is not None else 0,
        "total_rows": len(features_df) if features_df is not None else 0,
        "feature_columns": list(features_df.columns) if features_df is not None else [],
    }


def validate_features(**context) -> dict:
    from dags.utils import alert_validation_failure
    from src.datasets.crime_navigate.validators import NavigateSchemaEnforcer

    execution_date = context["ds"]
    try:
        df = _read_features(execution_date)
    except Exception:
        from dags.utils import read_data

        df = read_data(DATASET, "features", execution_date)
    enforcer = NavigateSchemaEnforcer()
    result = enforcer.validate_navigate_features(df)
    if not result.is_valid:
        alert_validation_failure(
            dataset=DATASET,
            stage="features",
            errors=result.errors,
            execution_date=execution_date,
            dag_id=DAG_ID,
            task_id="validate_features",
        )
        raise RuntimeError(f"Features validation failed: {result.errors[:5]}")
    return {"is_valid": result.is_valid, "error_count": len(result.errors), "stage": "features"}


def detect_drift(**context) -> dict:
    from datetime import timedelta

    from dags.utils import alert_drift_detected
    from src.datasets.crime_navigate.validators import NavigateDriftDetector

    execution_date = context["ds"]
    ref_dt = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=3)
    ref_date = ref_dt.strftime("%Y-%m-%d")

    # Load current features
    try:
        current_df = _read_features(execution_date)
    except Exception:
        try:
            from dags.utils import read_data

            current_df = read_data(DATASET, "features", execution_date)
        except Exception:
            current_df = None

    # Load reference features — missing is expected on first few runs
    try:
        reference_df = _read_features(ref_date)
    except Exception:
        try:
            from dags.utils import read_data

            reference_df = read_data(DATASET, "features", ref_date)
        except Exception:
            reference_df = None  # ← now actually reaches the None guard below

    if current_df is None or reference_df is None or current_df.empty or reference_df.empty:
        return {
            "drift_detected": False,
            "message": f"No reference data for {ref_date} — skipping drift detection",
            "drifted_columns": [],
        }

    detector = NavigateDriftDetector()
    result = detector.detect_navigate_drift(current_df, reference_df)
    if result.has_drift:
        severity = "critical" if getattr(result, "has_critical_drift", False) else "warning"
        alert_drift_detected(
            dataset=DATASET,
            drifted_features=[f.feature_name for f in result.features_with_drift],
            psi_scores={f.feature_name: f.psi for f in result.features_with_drift},
            severity=severity,
            execution_date=execution_date,
            dag_id=DAG_ID,
        )
    return {
        "drift_detected": result.has_drift,
        "severity": getattr(result, "overall_severity", "none"),
        "drifted_columns": [f.feature_name for f in result.features_with_drift],
    }


def check_fairness(**context) -> dict:
    from dags.utils import alert_fairness_violation
    from src.bias.fairness_checker import FairnessViolationError
    from src.datasets.crime_navigate.validators import NavigateFairnessChecker

    execution_date = context["ds"]
    try:
        df = _read_features(execution_date)
    except Exception:
        from dags.utils import read_data

        df = read_data(DATASET, "features", execution_date)
    if df is None or df.empty:
        return {"passes_gate": True, "violations": 0, "slices_evaluated": 0}
    checker = NavigateFairnessChecker()
    result = checker.evaluate_navigate_fairness(df)
    if result.has_violations:
        violations_payload = [
            {
                "metric": v.metric.value,
                "severity": v.severity.value,
                "dimension": v.dimension,
                "slice_value": str(v.slice_value),
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
        "passes_gate": result.passes_fairness_gate,
        "violations": len(result.violations),
        "slices_evaluated": result.slices_evaluated,
    }


def mitigate_bias(**context) -> dict:
    ti = context.get("ti")
    fairness_xcom = ti.xcom_pull(task_ids="check_fairness") if ti else None
    if not fairness_xcom or not fairness_xcom.get("violations", 0):
        return {"mitigation_applied": False, "reason": "No fairness violations"}
    return {"mitigation_applied": False, "reason": "Mitigation optional; not applied"}


def generate_model_card(**context) -> dict:
    from dags.utils import read_data
    from src.bias import ModelCardGenerator

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    if df is None:
        import pandas as pd

        df = pd.DataFrame()
    # XCom returns dicts; ModelCardGenerator expects result objects — pass None for optional summaries
    generator = ModelCardGenerator()
    card = generator.generate_model_card(
        dataset=DATASET,
        df=df,
        version=execution_date.replace("-", ""),
        description="Navigate crime scoring features (H3, hour buckets, severity weights)",
        validation_result=None,
        drift_result=None,
        fairness_result=None,
        mitigation_result=None,
    )
    paths = generator.save_model_card(card, format="both")
    return {"model_card_generated": True, "output_path": str(paths), "version": card.version}


def update_watermark(**context) -> dict:
    from dags.utils import read_data, set_watermark

    execution_date = context["ds"]
    df = read_data(DATASET, "processed", execution_date)
    if df is not None and "occurred_on_date" in df.columns and len(df) > 0:
        import pandas as pd

        max_date = pd.to_datetime(df["occurred_on_date"]).max()
        if pd.notna(max_date):
            set_watermark(DATASET, max_date.to_pydatetime(), execution_date)
            return {"watermark_updated": True, "new_watermark": max_date.isoformat()}
    return {"watermark_updated": False, "reason": "No valid dates"}


def record_lineage(**context) -> dict:
    from dags.utils import record_pipeline_lineage

    return record_pipeline_lineage(dataset=DATASET, dag_id=DAG_ID, context=context)


def pipeline_complete(**context) -> dict:
    from dags.utils import alert_pipeline_complete

    execution_date = context["ds"]
    ti = context.get("ti")
    ingest_result = ti.xcom_pull(task_ids="ingest_data") if ti else {}
    preprocess_result = ti.xcom_pull(task_ids="preprocess_data") if ti else {}
    features_result = ti.xcom_pull(task_ids="build_features") if ti else {}
    lineage_result = ti.xcom_pull(task_ids="record_lineage") if ti else {}
    stats = {
        "rows_ingested": ingest_result.get("rows_fetched", 0) or 0,
        "rows_processed": preprocess_result.get("rows_output", 0) or 0,
        "features_generated": features_result.get("total_rows", 0) or 0,
        "lineage_recorded": lineage_result.get("lineage_recorded", False) or False,
    }
    alert_pipeline_complete(
        dataset=DATASET,
        execution_date=execution_date,
        duration_seconds=0,
        stats=stats,
        dag_id=DAG_ID,
    )
    return {"status": "complete", "stats": stats}


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Navigate crime scoring — incremental ingest, H3 features, risk placeholders",
    schedule_interval=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    tags=["crime", "navigate", "scoring"],
    max_active_runs=1,
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
        python_callable=mitigate_bias,
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
