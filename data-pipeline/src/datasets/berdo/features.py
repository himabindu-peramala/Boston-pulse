"""
Boston Pulse - BERDO Feature Builder

Builds building energy and emissions features for urban analytics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class BerdoFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston BERDO data.

    Generates features useful for:
    - Housing intelligence (building energy efficiency scores)
    - Neighborhood sustainability rankings
    - Emissions trend analysis
    """

    def __init__(self, config: Settings | None = None):
        """Initialize BERDO feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "berdo"

    def get_entity_key(self) -> str:
        """Return entity key for aggregation."""
        return "_id"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Return feature definitions."""
        return [
            FeatureDefinition(
                name="_id",
                description="Building record identifier",
                dtype="int",
                source_columns=["_id"],
            ),
            FeatureDefinition(
                name="emissions_per_sqft",
                description="GHG emissions per square foot of floor area",
                dtype="float",
                source_columns=["total_ghg_emissions", "gross_floor_area"],
            ),
            FeatureDefinition(
                name="energy_per_sqft",
                description="Site energy use per square foot",
                dtype="float",
                source_columns=["site_energy_use_kbtu", "gross_floor_area"],
            ),
            FeatureDefinition(
                name="high_emitter",
                description="Flag for buildings in top 25% of emissions",
                dtype="int",
                source_columns=["total_ghg_emissions"],
            ),
            FeatureDefinition(
                name="energy_star_category",
                description="Energy Star score category (low/medium/high)",
                dtype="string",
                source_columns=["energy_star_score"],
            ),
            FeatureDefinition(
                name="electricity_ratio",
                description="Ratio of electricity to total energy use",
                dtype="float",
                source_columns=["electricity_use_grid_purchase", "site_energy_use_kbtu"],
            ),
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from preprocessed BERDO data."""
        if df.empty:
            return df

        df = self._engineer_emissions_features(df)
        df = self._engineer_energy_features(df)
        df = self._engineer_efficiency_category(df)

        return df

    def _engineer_emissions_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer emissions-related features."""
        if "total_ghg_emissions" in df.columns and "gross_floor_area" in df.columns:
            df["emissions_per_sqft"] = df["total_ghg_emissions"] / df["gross_floor_area"].replace(
                0, float("nan")
            )
            threshold = df["total_ghg_emissions"].quantile(0.75)
            df["high_emitter"] = (df["total_ghg_emissions"] >= threshold).astype(int)
        return df

    def _engineer_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer energy usage features."""
        if "site_energy_use_kbtu" in df.columns and "gross_floor_area" in df.columns:
            df["energy_per_sqft"] = df["site_energy_use_kbtu"] / df["gross_floor_area"].replace(
                0, float("nan")
            )

        if (
            "electricity_use_grid_purchase" in df.columns
            and "site_energy_use_kbtu" in df.columns
        ):
            df["electricity_ratio"] = df[
                "electricity_use_grid_purchase"
            ] / df["site_energy_use_kbtu"].replace(0, float("nan"))

        return df

    def _engineer_efficiency_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize buildings by Energy Star score."""
        if "energy_star_score" in df.columns:

            def categorize(score):
                if pd.isna(score):
                    return "Unknown"
                elif score >= 75:
                    return "High"
                elif score >= 50:
                    return "Medium"
                else:
                    return "Low"

            df["energy_star_category"] = df["energy_star_score"].apply(categorize)
        return df


def build_berdo_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for building BERDO features."""
    builder = BerdoFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()