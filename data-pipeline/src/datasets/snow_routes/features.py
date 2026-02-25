"""
Boston Pulse - Snow Emergency Routes Feature Builder

Generates features for Snow Emergency Routes data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class SnowRoutesFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for Boston Snow Emergency Routes."""

    def get_dataset_name(self) -> str:
        return "snow_routes"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        return [
            FeatureDefinition(
                name="is_city_maintained",
                description="Whether the route is maintained by the City of Boston",
                dtype="bool",
                source_columns=["responsibility"],
            )
        ]

    def get_entity_key(self) -> str:
        return "route_id"

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for snow routes."""
        if df.empty:
            return df

        # Simple feature: flag routes based on responsibility
        df["is_city_maintained"] = df["responsibility"].str.contains("City", case=False, na=False)

        # Add spatial clusters/categories if needed
        # For now, we'll just keep the cleaned data as features

        return df


def build_snow_routes_features(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for building snow routes features."""
    builder = SnowRoutesFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
