"""
Boston Pulse ML - Target Builder.

Build danger_rate label for each (h3_index, hour_bucket).

Steps:
  1. Load all processed incidents from history_start to execution_date
  2. Compute danger_rate = total_severity / days_active per cell-bucket
  3. Left-join onto features_df — cells with no incidents get danger_rate = 0.0

No spatial smoothing is applied. The neighbour signal is already captured in
the training features (neighbor_weighted_score_30d, neighbor_trend_3v10,
neighbor_gun_count_30d) — the model learns spatial relationships from those.

The danger_rate is the raw historical danger level for each cell at each hour.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from shared.gcs_loader import GCSLoader
from shared.schemas import TargetBuildResult

logger = logging.getLogger(__name__)


def build_targets(
    features_df: pd.DataFrame,
    execution_date: str,
    cfg: dict[str, Any],
    bucket: str,
) -> tuple[pd.DataFrame, TargetBuildResult]:
    """
    Build danger_rate label and join with features.

    Args:
        features_df: DataFrame from feature_loader (one row per cell-bucket)
        execution_date: "YYYY-MM-DD" — the DAG run date
        cfg: parsed crime_navigate_train.yaml
        bucket: GCS bucket name

    Returns:
        (training_df with features + danger_rate label, TargetBuildResult for XCom)

    The danger_rate is computed as:
        total_severity / days_active

    Where:
        - total_severity = sum of severity_weight for all incidents in cell-bucket
        - days_active = number of unique days with incidents in cell-bucket

    This normalises for cells with short history — a cell with 2 incidents in 2 days
    is as dangerous as one with 30 incidents in 30 days.
    """
    loader = GCSLoader(bucket)

    history_start = cfg["data"]["history_start"]
    processed_prefix = cfg["data"]["processed_prefix"]
    target_col = cfg["features"]["target_column"]  # "danger_rate"

    logger.info(f"Loading processed data from {history_start} → {execution_date}")

    try:
        processed_df = loader.read_all_partitions(
            prefix=processed_prefix,
            filename="data.parquet",
            after=history_start,
            before=execution_date,
            columns=[
                "h3_index",
                "hour_bucket",
                "occurred_on_date",
                "severity_weight",
                "incident_number",
                "district",
            ],
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"No processed data found — cannot build label: {e}") from e

    if processed_df.empty:
        raise RuntimeError("No processed data found — cannot build label") from None

    # Parse dates and clean
    processed_df["occurred_on_date"] = pd.to_datetime(
        processed_df["occurred_on_date"], utc=True, errors="coerce"
    )
    processed_df = processed_df.dropna(subset=["h3_index", "occurred_on_date"])

    # Compute raw danger rate per (h3_index, hour_bucket)
    danger = processed_df.groupby(["h3_index", "hour_bucket"], as_index=False).agg(
        total_severity=("severity_weight", "sum"),
        days_active=("occurred_on_date", lambda x: x.dt.normalize().nunique()),
        incident_count=("incident_number", "count"),
    )

    # Normalise by days_active so cells with short history are not penalised
    # A cell with 2 incidents in 2 days is as dangerous as one with 30 in 30 days
    danger[target_col] = danger["total_severity"] / danger["days_active"].clip(lower=1)

    logger.info(
        f"Computed {target_col} for {len(danger):,} cell-bucket pairs. "
        f"Mean: {danger[target_col].mean():.4f}, Max: {danger[target_col].max():.4f}"
    )

    # Get district info for bias analysis (most common district per cell)
    district_map = (
        processed_df.groupby("h3_index")["district"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "UNKNOWN")
        .to_dict()
    )

    # Join label onto feature DataFrame
    # Left join — cells in features with no incidents get danger_rate = 0.0
    label_cols = ["h3_index", "hour_bucket", target_col, "incident_count"]
    training_df = features_df.merge(
        danger[label_cols],
        on=["h3_index", "hour_bucket"],
        how="left",
    )

    # Fill missing danger_rate with 0.0 (cells with no incidents)
    training_df[target_col] = training_df[target_col].fillna(0.0)
    training_df["incident_count"] = training_df["incident_count"].fillna(0).astype(int)

    # Add district for bias analysis
    training_df["district"] = training_df["h3_index"].map(district_map).fillna("UNKNOWN")

    zero_cells = int((training_df[target_col] == 0.0).sum())
    logger.info(
        f"Training matrix: {len(training_df):,} rows, "
        f"{zero_cells:,} zero-rate cells, "
        f"mean_rate={training_df[target_col].mean():.4f}"
    )

    return training_df, TargetBuildResult(
        rows=len(training_df),
        h3_cells=training_df["h3_index"].nunique(),
        mean_danger_rate=float(training_df[target_col].mean()),
        zero_rate_cells=zero_cells,
        success=True,
    )
