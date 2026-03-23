"""
Boston Pulse - Street Sweeping Schedules Data Preprocessor

Cleans and validates Street Sweeping Schedules data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class StreetSweepingPreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston Street Sweeping Schedules data.
    """

    COLUMN_MAPPINGS = {
        "_id": "_id",
        "sam_street_id": "sam_street_id",
        "full_street_name": "full_street_name",
        "district": "district",
        "side_of_street": "side_of_street",
        "from_street": "from_street",
        "to_street": "to_street",
        "season_start": "season_start",
        "season_end": "season_end",
        "week_type": "week_type",
        "tow_zone": "tow_zone",
    }

    REQUIRED_COLUMNS = [
        "_id",
        "sam_street_id",
        "full_street_name",
        "district",
        "side_of_street",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize street sweeping preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "street_sweeping"

    def get_required_columns(self) -> list[str]:
        """Return required output columns."""
        return self.REQUIRED_COLUMNS

    def get_column_mappings(self) -> dict[str, str]:
        """Return column name mappings."""
        return self.COLUMN_MAPPINGS

    def get_dtype_mappings(self) -> dict[str, str]:
        """Return data type mappings."""
        return {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply street sweeping transformations."""
        if df.empty:
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df

        # Rename columns to standardized names
        df = df.rename(columns=self.COLUMN_MAPPINGS)

        df = self._standardize_strings(df)
        df = self._handle_missing_values(df)
        df = self.drop_duplicates(df, subset=["_id"], keep="last")
        df = self._select_output_columns(df)

        return df

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize string fields."""
        # Force sam_street_id as string
        if "sam_street_id" in df.columns:
            df["sam_street_id"] = df["sam_street_id"].astype(str).str.split(".").str[0]

        for col in ["full_street_name", "district", "district_name", "side_of_street"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace("NAN", np.nan)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        for col in ["district", "side_of_street", "district_name"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # Ensure weekday columns exist for features
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            if day not in df.columns:
                df[day] = 0

        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns based on processed_schema.json."""
        output_columns = [
            "_id",
            "sam_street_id",
            "full_street_name",
            "from_street",
            "to_street",
            "district",
            "side_of_street",
            "season_start",
            "season_end",
            "week_type",
            "tow_zone",
            "lat",
            "long",
            # Intermediate cleaned features needed for feature builder
            "is_year_round",
            "is_every_week",
            "sweep_days_count",
        ]
        available = [c for c in output_columns if c in df.columns]
        return df[available].copy()


def preprocess_street_sweeping_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing street sweeping data."""
    preprocessor = StreetSweepingPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
