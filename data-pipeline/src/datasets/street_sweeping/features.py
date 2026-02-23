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
                name="tow_enforced",
                description="Whether tow zone is enforced during sweeping",
                dtype="int",
                source_columns=["tow_zone"],
            ),
            FeatureDefinition(
                name="is_every_week",
                description="Whether street is swept every week",
                dtype="int",
                source_columns=["week_type"],
            ),
            FeatureDefinition(
                name="season_start_month",
                description="Numeric month when sweeping season starts",
                dtype="int",
                source_columns=["season_start"],
            ),
            FeatureDefinition(
                name="season_end_month",
                description="Numeric month when sweeping season ends",
                dtype="int",
                source_columns=["season_end"],
            ),
            FeatureDefinition(
                name="active_months_count",
                description="Total number of months in sweeping season",
                dtype="int",
                source_columns=["season_start", "season_end"],
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
        df = self._engineer_tow_risk_feature(df)

        self.log_transformation("build_features")
        return df

    def _engineer_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract schedule-related features."""
        month_map = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }

        for col, new_col in [
            ("season_start", "season_start_month"),
            ("season_end", "season_end_month"),
        ]:
            if col in df.columns:
                df[new_col] = df[col].str[:3].str.upper().map(month_map)

        if "season_start_month" in df.columns and "season_end_month" in df.columns:
            df["active_months_count"] = (
                df["season_end_month"] - df["season_start_month"] + 1
            ).clip(lower=0)

        if "week_type" in df.columns:
            df["is_every_week"] = (
                df["week_type"].str.upper().str.contains("EVERY", na=False).astype(int)
            )

        return df

    def _engineer_district_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode district as a categorical feature."""
        if "district" in df.columns:
            df["district_code"] = df["district"].astype("category").cat.codes
        return df

    def _engineer_tow_risk_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag streets with tow zone enforcement."""
        if "tow_zone" in df.columns:
            df["tow_enforced"] = df["tow_zone"].str.upper().ne("NO").astype(int)
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
