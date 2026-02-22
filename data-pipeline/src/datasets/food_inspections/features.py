"""
Boston Pulse - Food Inspections Feature Builder

Builds food inspection-related features for urban analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class FoodInspectionsFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston Food Establishment Inspections data.

    Builds aggregated food inspection features at the grid cell level.
    """

    # Grid cell size in degrees
    GRID_SIZE = 0.001

    # Rolling window sizes in days (longer for inspections)
    WINDOW_SIZES = [90, 180, 365]

    def __init__(self, config: Settings | None = None):
        """Initialize food inspections feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "food_inspections"

    def get_entity_key(self) -> str:
        """Return entity key for aggregation."""
        return "grid_cell"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Return feature definitions."""
        features = [
            # Grid identification
            FeatureDefinition(
                name="grid_cell",
                description="Grid cell identifier (lat_lon)",
                dtype="string",
                source_columns=["lat", "long"],
            ),
            # Inspection counts
            FeatureDefinition(
                name="inspection_count_180d",
                description="Total inspections in past 180 days",
                dtype="int",
                source_columns=["_id"],
                aggregation="count",
                window_days=180,
            ),
            FeatureDefinition(
                name="failure_count_180d",
                description="Total failed inspections in past 180 days",
                dtype="int",
                source_columns=["result"],
                window_days=180,
            ),
            FeatureDefinition(
                name="failure_ratio_180d",
                description="Ratio of failed inspections in past 180 days",
                dtype="float",
                source_columns=["result"],
                window_days=180,
            ),
        ]
        return features

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build food inspection features from processed data."""
        logger.info(f"Building food inspection features from {len(df)} records")

        # Filter to records with valid coordinates
        df = df[df["lat"].notna() & df["long"].notna()].copy()
        if len(df) == 0:
            return pd.DataFrame()

        # Add grid cell
        df = self._add_grid_cell(df)

        # Get reference date
        reference_date = df["resultdttm"].max()
        
        # Build features by grid cell
        features_list = []
        for grid_cell, group in df.groupby("grid_cell"):
            features = self._compute_cell_features(group, reference_date)
            features["grid_cell"] = grid_cell
            features_list.append(features)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        
        # Add grid coordinates from cell ID
        features_df["grid_lat"] = features_df["grid_cell"].apply(lambda x: float(x.split("_")[0]))
        features_df["grid_long"] = features_df["grid_cell"].apply(lambda x: float(x.split("_")[1]))

        return features_df

    def _add_grid_cell(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add grid cell identifier."""
        df["grid_lat"] = (df["lat"] / self.GRID_SIZE).round() * self.GRID_SIZE
        df["grid_long"] = (df["long"] / self.GRID_SIZE).round() * self.GRID_SIZE
        df["grid_cell"] = (
            df["grid_lat"].apply(lambda x: f"{x:.3f}")
            + "_"
            + df["grid_long"].apply(lambda x: f"{x:.3f}")
        )
        return df

    def _compute_cell_features(self, group: pd.DataFrame, reference_date: datetime) -> dict[str, Any]:
        """Compute features for a single grid cell."""
        features: dict[str, Any] = {}
        # Get neighborhood from another source if needed, but here we'll just skip it or pick a dummy if not in df
        features["neighborhood"] = "Unknown" # Neighborhood not joined in this simple version

        for window in self.WINDOW_SIZES:
            window_mask = group["resultdttm"] >= (reference_date - pd.Timedelta(days=window))
            window_data = group[window_mask]
            features[f"inspection_count_{window}d"] = len(window_data)

            if window == 180:
                if len(window_data) > 0:
                    # In Boston data, results like "Fail", "Pass", etc.
                    # We'll consider anything containing "Fail" as a failure for this example.
                    failure_mask = window_data["result"].str.contains("Fail", case=False, na=False)
                    failure_count = failure_mask.sum()
                    features["failure_count_180d"] = int(failure_count)
                    features["failure_ratio_180d"] = failure_count / len(window_data)
                else:
                    features["failure_count_180d"] = 0
                    features["failure_ratio_180d"] = 0.0

        return features


def build_food_inspections_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for building food inspection features."""
    builder = FoodInspectionsFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
