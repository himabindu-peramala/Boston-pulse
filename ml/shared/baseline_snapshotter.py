"""
Boston Pulse ML - Training Baseline Snapshotter.

Captures feature distributions of the training data whenever a new model
is promoted to production. Used by monitoring to detect serving-vs-training drift.

Baselines stored at:
    gs://{artifact_bucket}/monitoring/baselines/{model_name}/{version}/feature_sample.parquet
    gs://{artifact_bucket}/monitoring/baselines/{model_name}/{version}/feature_stats.json
    gs://{artifact_bucket}/monitoring/baselines/{model_name}/latest/  (copy of most recent)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)

BASELINE_SAMPLE_SIZE = 10_000


def snapshot_training_baseline(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    version: str,
    cfg: dict[str, Any],
) -> dict[str, str]:
    """
    Save a baseline snapshot of the training features + target.

    Args:
        training_df: The training DataFrame (post target-join)
        feature_cols: List of feature column names to snapshot
        target_col: Target column name
        version: Model version this baseline corresponds to
        cfg: Training config

    Returns:
        Dict with GCS paths to the sample and stats files
    """
    registry_cfg = cfg.get("registry", {})
    # ARTIFACT_BUCKET env var (set by deploy_app.sh from Terraform output)
    # overrides the YAML so a fresh project can use its own bucket.
    bucket_name = os.environ.get(
        "ARTIFACT_BUCKET",
        registry_cfg.get("artifact_bucket", "boston-pulse-mlflow-artifacts"),
    )
    model_name = registry_cfg.get("package", "crime-risk").split("/")[-1]

    # Columns to snapshot
    cols = [c for c in feature_cols if c in training_df.columns]
    if target_col in training_df.columns:
        cols = cols + [target_col]

    # Sample for Evidently (reference data)
    sample_df = training_df[cols].sample(
        n=min(BASELINE_SAMPLE_SIZE, len(training_df)),
        random_state=42,
    )

    # Stats JSON for lightweight drift checks
    stats: dict[str, Any] = {
        "version": version,
        "n_rows_total": int(len(training_df)),
        "n_rows_sampled": int(len(sample_df)),
        "feature_stats": {},
    }
    for c in cols:
        series = training_df[c]
        if pd.api.types.is_numeric_dtype(series):
            stats["feature_stats"][c] = {
                "dtype": "numeric",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "null_pct": float(series.isna().mean()),
            }
        else:
            vc = series.value_counts(normalize=True).head(20).to_dict()
            stats["feature_stats"][c] = {
                "dtype": "categorical",
                "n_unique": int(series.nunique()),
                "top_values": {str(k): float(v) for k, v in vc.items()},
                "null_pct": float(series.isna().mean()),
            }

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    versioned_prefix = f"monitoring/baselines/{model_name}/{version}"
    latest_prefix = f"monitoring/baselines/{model_name}/latest"

    sample_path = f"/tmp/baseline_sample_{version}.parquet"
    sample_df.to_parquet(sample_path, index=False)

    sample_blob = bucket.blob(f"{versioned_prefix}/feature_sample.parquet")
    sample_blob.upload_from_filename(sample_path)

    stats_blob = bucket.blob(f"{versioned_prefix}/feature_stats.json")
    stats_blob.upload_from_string(json.dumps(stats, indent=2), content_type="application/json")

    # Copy to latest/ (atomic pointer update)
    bucket.copy_blob(sample_blob, bucket, f"{latest_prefix}/feature_sample.parquet")
    bucket.copy_blob(stats_blob, bucket, f"{latest_prefix}/feature_stats.json")

    paths = {
        "sample_uri": f"gs://{bucket_name}/{versioned_prefix}/feature_sample.parquet",
        "stats_uri": f"gs://{bucket_name}/{versioned_prefix}/feature_stats.json",
        "latest_sample_uri": f"gs://{bucket_name}/{latest_prefix}/feature_sample.parquet",
    }
    logger.info(f"Saved training baseline v{version}: {paths['sample_uri']}")
    return paths
