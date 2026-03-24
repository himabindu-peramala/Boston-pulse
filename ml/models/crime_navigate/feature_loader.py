"""
Boston Pulse ML - Feature Loader.

Load latest features per (h3_index, hour_bucket) from GCS.

Design:
- One row per (h3_index, hour_bucket) — the latest computed values.
- Load the most recent N days of partitions, dedup to keep latest per cell.
- This is NOT a time-series load. No temporal stacking.
- Drop computed_date after dedup — not a training feature.

The training matrix has ~90,000 rows (15,000 cells × 6 hour buckets).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from shared.gcs_loader import GCSLoader
from shared.schemas import FeatureLoadResult

logger = logging.getLogger(__name__)


def load_features(
    execution_date: str,
    cfg: dict[str, Any],
    bucket: str,
) -> tuple[pd.DataFrame, FeatureLoadResult]:
    """
    Load latest features per (h3_index, hour_bucket) from GCS.

    Args:
        execution_date: "YYYY-MM-DD" — the DAG run date
        cfg: parsed crime_navigate_train.yaml
        bucket: GCS bucket name

    Returns:
        (DataFrame with one row per cell-bucket, FeatureLoadResult for XCom)

    The returned DataFrame has columns:
        h3_index, hour_bucket, weighted_score_3d, weighted_score_30d, ...
        (all columns from features.input_columns in config)
    """
    loader = GCSLoader(bucket)

    features_prefix = cfg["data"]["features_prefix"]
    input_columns = cfg["features"]["input_columns"]
    lookback_days = cfg["scoring"]["active_cell_lookback_days"]

    exec_dt = datetime.strptime(execution_date, "%Y-%m-%d")
    since_date = (exec_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.info(f"Loading features from {since_date} to {execution_date}")

    try:
        raw_df = loader.read_all_partitions(
            prefix=features_prefix,
            filename="features.parquet",
            after=since_date,
            before=execution_date,
        )
    except FileNotFoundError as e:
        logger.error(f"No feature partitions found: {e}")
        return pd.DataFrame(), FeatureLoadResult(
            rows=0,
            h3_cells=0,
            columns=[],
            success=False,
            error=str(e),
        )

    if raw_df.empty:
        raise RuntimeError(f"No feature partitions found in last {lookback_days} days")

    # Dedup: keep latest computed_date per (h3_index, hour_bucket)
    # This gives us one row per cell-bucket with the most recent feature values
    sort_col = "computed_date" if "computed_date" in raw_df.columns else "date"
    raw_df["_sort"] = pd.to_datetime(raw_df[sort_col], errors="coerce")

    features_df = (
        raw_df.sort_values("_sort", ascending=False)
        .drop_duplicates(subset=["h3_index", "hour_bucket"], keep="first")
        .drop(columns=["_sort"], errors="ignore")
        .reset_index(drop=True)
    )

    # Validate required columns are present
    missing = [c for c in input_columns if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"Feature columns missing from parquet: {missing}\n"
            f"Available: {list(features_df.columns)}\n"
            f"Check that data-pipeline features.py output matches config input_columns."
        )

    logger.info(
        f"Loaded {len(features_df):,} cell-bucket pairs "
        f"({features_df['h3_index'].nunique():,} H3 cells)"
    )

    return features_df, FeatureLoadResult(
        rows=len(features_df),
        h3_cells=features_df["h3_index"].nunique(),
        columns=list(features_df.columns),
        success=True,
    )
