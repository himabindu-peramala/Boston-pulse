"""
Boston Pulse - Street Sweeping Feature Builder

Engineers features from cleaned Street Sweeping Schedules data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder
from src.shared.config import Settings

logger = logging.getLogger(__name__)

# Day name to number mapping
DAY_ORDER = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2,
    "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


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
        # Parse season months
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }

        for col, new_col in [("season_start", "season_start_month"), ("season_end", "season_end_month")]:
            if col in df.columns:
                df[new_col] = df[col].str[:3].str.upper().map(month_map)

        # Flag active season (April - November typical Boston sweeping)
        if "season_start_month" in df.columns and "season_end_month" in df.columns:
            df["active_months_count"] = (
                df["season_end_month"] - df["season_start_month"] + 1
            ).clip(lower=0)

        # Week type encoding
        if "week_type" in df.columns:
            df["is_every_week"] = df["week_type"].str.upper().str.contains("EVERY", na=False).astype(int)

        return df

    def _engineer_district_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode district as a categorical feature."""
        if "district" in df.columns:
            df["district_code"] = df["district"].astype("category").cat.codes
        return df

    def _engineer_tow_risk_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag streets with tow zone enforcement."""
        if "tow_zone" in df.columns:
            df["tow_enforced"] = (
                df["tow_zone"].str.upper().ne("NO").astype(int)
            )
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