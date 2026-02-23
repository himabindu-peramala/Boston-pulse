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
        "from_street": "from_street",
        "to_street": "to_street",
        "district": "district",
        "side_of_street": "side_of_street",
        "season_start": "season_start",
        "season_end": "season_end",
        "week_type": "week_type",
        "tow_zone": "tow_zone",
        "lat": "lat",
        "long": "long",
    }

    DTYPE_MAPPINGS = {
        "_id": "int",
        "sam_street_id": "string",
        "full_street_name": "string",
        "from_street": "string",
        "to_street": "string",
        "district": "string",
        "side_of_street": "string",
        "season_start": "string",
        "season_end": "string",
        "week_type": "string",
        "tow_zone": "string",
        "lat": "float",
        "long": "float",
    }

    REQUIRED_COLUMNS = [
        "_id",
        "sam_street_id",
        "full_street_name",
        "district",
        "side_of_street",
        "season_start",
        "season_end",
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
        return self.DTYPE_MAPPINGS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply street sweeping transformations."""
        if df.empty:
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df

        # Standardize string columns
        df = self._standardize_strings(df)

        # Validate coordinates
        df = self._validate_coordinates(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop duplicates
        df = self.drop_duplicates(df, subset=["_id"], keep="last")

        # Final column selection
        df = self._select_output_columns(df)

        return df

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize string fields."""
        str_cols = [
            "full_street_name",
            "from_street",
            "to_street",
            "district",
            "side_of_street",
            "week_type",
            "tow_zone",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace("NAN", np.nan)
        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate lat/long against Boston bounds."""
        for col in ["lat", "long"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "lat" in df.columns and "long" in df.columns:
            bounds = self.config.validation.geo_bounds
            out_of_bounds = (
                (df["lat"] < bounds.min_lat)
                | (df["lat"] > bounds.max_lat)
                | (df["long"] < bounds.min_lon)
                | (df["long"] > bounds.max_lon)
            )
            df.loc[out_of_bounds, ["lat", "long"]] = np.nan
            self.log_transformation("validate_coordinates")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in categorical columns."""
        cat_cols = ["district", "side_of_street", "week_type", "tow_zone"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
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
