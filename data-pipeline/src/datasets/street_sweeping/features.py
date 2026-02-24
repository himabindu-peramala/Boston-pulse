"""
Boston Pulse - Street Sweeping Feature Builder

Builds street sweeping-related features for urban analytics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class StreetSweepingFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston Street Sweeping Schedules data.

    Generates features useful for:
    - Route planning (which streets are restricted when)
    - Civic intelligence (answering sweeping schedule queries)
    - Parking compliance notifications
    """

    def __init__(self, config: Settings | None = None):
        """Initialize street sweeping feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "street_sweeping"

    def get_entity_key(self) -> str:
        """Return entity key for aggregation."""
        return "sam_street_id"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Return feature definitions."""
        return [
            FeatureDefinition(
                name="sam_street_id",
                description="Street segment identifier",
                dtype="string",
                source_columns=["sam_street_id"],
            ),
            FeatureDefinition(
                name="is_year_round",
                description="Whether street is swept year round",
                dtype="int",
                source_columns=["year_round"],
            ),
            FeatureDefinition(
                name="is_every_week",
                description="Whether street is swept every week",
                dtype="int",
                source_columns=["week_1", "week_2", "week_3", "week_4"],
            ),
            FeatureDefinition(
                name="sweep_days_count",
                description="Number of days per week street is swept",
                dtype="int",
                source_columns=["monday", "tuesday", "wednesday", "thursday", "friday"],
            ),
            FeatureDefinition(
                name="district_code",
                description="Encoded district identifier",
                dtype="int",
                source_columns=["district"],
            ),
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from preprocessed street sweeping data."""
        if df.empty:
            return df

        df = self._engineer_schedule_features(df)
        df = self._engineer_district_features(df)

        return df

    def _engineer_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract schedule-related features."""
        # Year round flag
        if "year_round" in df.columns:
            df["is_year_round"] = (
                df["year_round"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
            ).astype(int)

        # Every week flag — if all 4 weeks are active
        week_cols = [c for c in ["week_1", "week_2", "week_3", "week_4"] if c in df.columns]
        if week_cols:
            df["is_every_week"] = (
                df[week_cols].apply(
                    lambda row: all(str(v).upper() in ["Y", "YES", "TRUE", "1"] for v in row),
                    axis=1,
                )
            ).astype(int)

        return df

    def _engineer_district_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode district as a categorical feature."""
        if "district" in df.columns:
            df["district_code"] = df["district"].astype("category").cat.codes
        return df


def build_street_sweeping_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for building street sweeping features."""
    builder = StreetSweepingFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
