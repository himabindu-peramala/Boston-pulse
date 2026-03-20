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
            FeatureDefinition(
                name="tow_enforced",
                description="Whether towing is enforced for this street",
                dtype="int",
                source_columns=["side_of_street"],
            ),
            FeatureDefinition(
                name="season_start_month",
                description="Month street sweeping season begins",
                dtype="int",
                source_columns=[],
            ),
            FeatureDefinition(
                name="season_end_month",
                description="Month street sweeping season ends",
                dtype="int",
                source_columns=[],
            ),
            FeatureDefinition(
                name="active_months_count",
                description="Number of months sweeping is active",
                dtype="int",
                source_columns=[],
            ),
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from preprocessed street sweeping data."""
        if df.empty:
            return df

        df = self._engineer_schedule_features(df)
        df = self._engineer_district_features(df)

        # Ensure strictly required schema columns exist
        for req in ["_id", "district", "is_every_week", "tow_enforced"]:
            if req not in df.columns:
                df[req] = pd.NA

        # Select only the features defined in get_feature_definitions + standard IDs
        feature_names = [f.name for f in self.get_feature_definitions()] + ["_id", "district"]
        available = [c for c in feature_names if c in df.columns]
        return df[available].copy()

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

        # Count of sweep days
        day_cols = [
            c for c in ["monday", "tuesday", "wednesday", "thursday", "friday"] if c in df.columns
        ]
        if day_cols:
            df["sweep_days_count"] = (
                df[day_cols].apply(
                    lambda row: sum(
                        str(v).upper() in ["Y", "YES", "TRUE", "1", 1, "1.0"] for v in row
                    ),
                    axis=1,
                )
            ).astype(int)
        else:
            df["sweep_days_count"] = 0

        return df

    def _engineer_district_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode district as a categorical feature."""
        if "district" in df.columns:
            df["district_code"] = df["district"].astype("category").cat.codes

        # Towing is usually enforced on all major routes
        df["tow_enforced"] = 1

        # Boston sweeping season is generally April (4) to November (11)
        df["season_start_month"] = 4
        df["season_end_month"] = 11
        df["active_months_count"] = 8

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
