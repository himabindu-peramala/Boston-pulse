"""
Boston Pulse ML - Scorer.

Score all active cells. features_df from feature_loader is reused directly —
no additional GCS read. Per-bucket min-max scaling ensures score 80 means
more dangerous than 80% of cells at THIS hour.

Risk score semantics:
  0   = safest cell in Boston at this hour
  100 = most dangerous cell in Boston at this hour
  Score is relative within each hour_bucket — compare only same-hour routes.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from shared.gcs_loader import GCSLoader
from shared.schemas import ScoringResult

logger = logging.getLogger(__name__)


def score_all_cells(
    model: lgb.Booster,
    features_df: pd.DataFrame,  # reused from feature_loader — no re-read
    execution_date: str,
    cfg: dict[str, Any],
    bucket: str,
    model_version: str,
) -> tuple[pd.DataFrame, ScoringResult]:
    """
    Score all active cells with the trained model.

    Args:
        model: trained LightGBM model
        features_df: DataFrame from feature_loader (one row per cell-bucket)
        execution_date: "YYYY-MM-DD"
        cfg: parsed crime_navigate_train.yaml
        bucket: GCS bucket name
        model_version: version string for the model

    Returns:
        (scores DataFrame, ScoringResult for XCom)

    The scores DataFrame has columns:
        h3_index, hour_bucket, predicted_danger, risk_score, risk_tier,
        model_version, scored_at
    """
    feature_cols = cfg["features"]["input_columns"]
    tiers = cfg["scoring"]["tiers"]

    # Make predictions (danger_rate cannot be negative)
    raw = np.maximum(model.predict(features_df[feature_cols]), 0.0)

    # Build output DataFrame
    out = features_df[["h3_index", "hour_bucket"]].copy()
    out["predicted_danger"] = raw
    out["model_version"] = model_version
    out["scored_at"] = execution_date

    # Scale to 0-100 per bucket
    out["risk_score"] = _scale_per_bucket(out)

    # Assign risk tiers
    out["risk_tier"] = out["risk_score"].apply(
        lambda s: (
            "HIGH" if s >= tiers["high"][0] else ("MEDIUM" if s >= tiers["medium"][0] else "LOW")
        )
    )

    # Write to GCS
    loader = GCSLoader(bucket)
    output_path = loader.write_parquet(
        out[
            [
                "h3_index",
                "hour_bucket",
                "predicted_danger",
                "risk_score",
                "risk_tier",
                "model_version",
                "scored_at",
            ]
        ],
        prefix=cfg["data"]["scores_prefix"],
        date=execution_date,
        filename="scores.parquet",
    )

    dist = out["risk_tier"].value_counts().to_dict()
    logger.info(f"Scored {len(out):,} cells. Distribution: {dist}")

    return out, ScoringResult(
        rows_scored=len(out),
        h3_cells=out["h3_index"].nunique(),
        output_gcs_path=output_path,
        score_distribution=dist,
        model_version=model_version,
        success=True,
    )


def _scale_per_bucket(df: pd.DataFrame) -> pd.Series:
    """
    Per-bucket min-max scaling.

    Score 80 = more dangerous than 80% of cells at this hour.
    Each hour bucket is scaled independently so scores are comparable
    within the same hour, not across hours.
    """
    scaled = df["predicted_danger"].copy().astype(float)

    for b in df["hour_bucket"].unique():
        mask = df["hour_bucket"] == b
        vals = df.loc[mask, "predicted_danger"]
        lo, hi = vals.min(), vals.max()

        if hi > lo:
            scaled[mask] = (vals - lo) / (hi - lo) * 100.0
        else:
            # All values are the same — assign neutral score
            scaled[mask] = 50.0

    return scaled.round(2)


def _scale_global(df: pd.DataFrame) -> pd.Series:
    """
    Global min-max scaling (alternative to per-bucket).

    Use this if you want scores comparable across hours.
    """
    vals = df["predicted_danger"]
    lo, hi = vals.min(), vals.max()

    if hi > lo:
        return ((vals - lo) / (hi - lo) * 100.0).round(2)
    else:
        return pd.Series([50.0] * len(vals), index=vals.index)


def get_score_statistics(scores_df: pd.DataFrame) -> dict[str, Any]:
    """Get statistics about the scored cells."""
    return {
        "total_cells": len(scores_df),
        "unique_h3": scores_df["h3_index"].nunique(),
        "hour_buckets": sorted(scores_df["hour_bucket"].unique().tolist()),
        "risk_distribution": scores_df["risk_tier"].value_counts().to_dict(),
        "mean_risk_score": float(scores_df["risk_score"].mean()),
        "median_risk_score": float(scores_df["risk_score"].median()),
        "mean_predicted_danger": float(scores_df["predicted_danger"].mean()),
    }
