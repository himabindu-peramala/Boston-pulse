"""
Boston Pulse - City Owned Property Feature Builder

Generates features for City Owned Property data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class CityPropertyFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for Boston City Owned Property."""

    def get_dataset_name(self) -> str:
        return "city_property"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        return [
            FeatureDefinition(
                name="is_education",
                description="Whether the property is owned by an educational department",
                dtype="bool",
                source_columns=["owner"],
            ),
            FeatureDefinition(
                name="is_housing",
                description="Whether the property is owned by a housing department",
                dtype="bool",
                source_columns=["owner"],
            ),
        ]

    def get_entity_key(self) -> str:
        return "property_id"

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for city property."""
        if df.empty:
            return df

        # Flag major owners (e.g. BHA, School Dept)
        df["is_education"] = df["owner"].str.contains("School", case=False, na=False)
        df["is_housing"] = df["owner"].str.contains("Housing", case=False, na=False)

        return df


def build_city_property_features(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for building city property features."""
    builder = CityPropertyFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
