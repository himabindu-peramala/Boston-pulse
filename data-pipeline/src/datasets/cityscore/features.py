"""
Boston Pulse - CityScore Feature Builder

Builds CityScore-related features for urban trends.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class CityScoreFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston CityScore data.

    Builds temporal features based on CityScore metrics.
    Note: CityScore data is city-wide and lacks geographic coordinates.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize CityScore feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "cityscore"

    def get_entity_key(self) -> str:
        """Return entity key for aggregation."""
        return "date"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Return feature definitions."""
        features = [
            FeatureDefinition(
                name="date",
                description="The date of the score",
                dtype="string",
                source_columns=["date"],
            ),
            FeatureDefinition(
                name="avg_day_score",
                description="Average CityScore across all metrics for the day",
                dtype="float",
                source_columns=["day_score"],
                aggregation="mean",
            ),
        ]
        return features

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build CityScore features.
        """
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "metric",
                    "date",
                    "avg_score_7d",
                    "avg_score_30d",
                    "score_vs_target",
                    "execution_date",
                ]
            )

        df = df.copy()
        logger.info(f"Building CityScore features from {len(df)} records")

        if len(df) == 0:
            return pd.DataFrame()

        # Pivot data to have metrics as columns
        # This creates a time-series of scores
        features_df = df.groupby("date").agg({"day_score": "mean"}).reset_index()
        features_df.rename(columns={"day_score": "avg_day_score"}, inplace=True)

        # Add individual metric scores as features
        pivoted = df.pivot_table(
            index="date", columns="metric_name", values="day_score", aggfunc="last"
        ).reset_index()

        # Merge overall average with pivoted metrics
        features_df = features_df.merge(pivoted, on="date", how="left")

        # Standardize column names (lowercase, no spaces)
        features_df.columns = [
            c.lower().replace(" ", "_").replace("-", "_") for c in features_df.columns
        ]

        return features_df


def build_cityscore_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for building CityScore features."""
    builder = CityScoreFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
