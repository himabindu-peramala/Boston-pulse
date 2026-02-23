"""
Boston Pulse - 311 Feature Builder

Builds 311-related features for urban analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class Service311FeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston 311 data.

    Builds aggregated 311 features at the grid cell level.
    """

    # Grid cell size in degrees
    GRID_SIZE = 0.001

    # Rolling window sizes in days
    WINDOW_SIZES = [7, 30, 90]

    def __init__(self, config: Settings | None = None):
        """Initialize 311 feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "311"

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
            FeatureDefinition(
                name="neighborhood",
                description="Neighborhood name",
                dtype="string",
                source_columns=["neighborhood"],
            ),
            # Request counts
            FeatureDefinition(
                name="request_count_7d",
                description="Total 311 requests in past 7 days",
                dtype="int",
                source_columns=["case_id"],
                aggregation="count",
                window_days=7,
            ),
            FeatureDefinition(
                name="request_count_30d",
                description="Total 311 requests in past 30 days",
                dtype="int",
                source_columns=["case_id"],
                aggregation="count",
                window_days=30,
            ),
            # Topic features (placeholders for specific common topics if needed)
            FeatureDefinition(
                name="overdue_ratio_30d",
                description="Ratio of overdue requests in past 30 days",
                dtype="float",
                source_columns=["on_time"],
                window_days=30,
            ),
        ]
        return features

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 311 features from processed data."""
        logger.info(f"Building 311 features from {len(df)} records")

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "grid_cell",
                    "grid_lat",
                    "grid_long",
                    "execution_date",
                    "request_count_7d",
                    "request_count_30d",
                    "request_count_90d",
                    "overdue_ratio_30d",
                    "overdue_ratio_90d",
                ]
            )

        # Filter to records with valid coordinates
        df = df[df["lat"].notna() & df["long"].notna()].copy()
        if len(df) == 0:
            return pd.DataFrame()

        # Add grid cell
        df = self._add_grid_cell(df)

        # Get reference date
        reference_date = df["open_date"].max()

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

    def _compute_cell_features(
        self, group: pd.DataFrame, reference_date: datetime
    ) -> dict[str, Any]:
        """Compute features for a single grid cell."""
        features: dict[str, Any] = {}
        features["neighborhood"] = (
            group["neighborhood"].mode().iloc[0] if len(group) > 0 else "Unknown"
        )

        for window in self.WINDOW_SIZES:
            window_mask = group["open_date"] >= (reference_date - pd.Timedelta(days=window))
            window_data = group[window_mask]
            features[f"request_count_{window}d"] = len(window_data)

            if window == 30:
                if len(window_data) > 0:
                    overdue_count = (window_data["on_time"] == "OVERDUE").sum()
                    features["overdue_ratio_30d"] = overdue_count / len(window_data)
                else:
                    features["overdue_ratio_30d"] = 0.0

        return features


def build_311_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for building 311 features."""
    builder = Service311FeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
