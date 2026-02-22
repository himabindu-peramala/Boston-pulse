"""
Boston Pulse - Food Inspections Data Preprocessor

Cleans and validates Food Establishment Inspections data.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class FoodInspectionsPreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston Food Establishment Inspections data.
    """

    # Column mapping from raw API names to standardized names
    COLUMN_MAPPINGS = {
        "_id": "_id",
        "businessname": "businessname",
        "licenseno": "licenseno",
        "result": "result",
        "resultdttm": "resultdttm",
        "address": "address",
        "zip": "zip",
        "location": "location",
    }

    # Data type mappings
    DTYPE_MAPPINGS = {
        "_id": "integer",
        "businessname": "string",
        "licenseno": "string",
        "result": "string",
        "resultdttm": "datetime",
        "address": "string",
        "zip": "string",
        "location": "string",
    }

    # Required output columns
    REQUIRED_COLUMNS = [
        "_id",
        "businessname",
        "licenseno",
        "result",
        "resultdttm",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize food inspections preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "food_inspections"

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
        """Apply food investigations-specific transformations."""
        # Process and validate datetime
        df = self._process_datetimes(df)

        # Process and validate coordinates
        df = self._process_location(df)

        # Standardize strings
        df = self._standardize_strings(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop duplicates
        df = self.drop_duplicates(df, subset=["_id"], keep="last")

        # Final column selection
        df = self._select_output_columns(df)

        return df

    def _process_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate datetime fields."""
        for col in ["resultdttm", "issdttm", "violdttm"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "resultdttm" in df.columns:
            # Remove records with invalid result dates
            invalid_dates = df["resultdttm"].isna().sum()
            if invalid_dates > 0:
                self.log_dropped_rows("invalid_resultdttm", int(invalid_dates))
            df = df[df["resultdttm"].notna()].copy()

            # Add temporal components for aggregation
            df["year"] = df["resultdttm"].dt.year
            df["month"] = df["resultdttm"].dt.month
            df["day_of_week"] = df["resultdttm"].dt.day_name()

        return df

    def _process_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate lat/long from location string."""
        if "location" in df.columns:
            # Format is usually "(42.345, -71.098)"
            def extract_coords(loc_str):
                if pd.isna(loc_str) or not isinstance(loc_str, str):
                    return None, None
                match = re.search(r"\(([^,]+),\s*([^)]+)\)", loc_str)
                if match:
                    try:
                        return float(match.group(1)), float(match.group(2))
                    except ValueError:
                        return None, None
                return None, None

            coords = df["location"].apply(extract_coords)
            df["lat"] = coords.apply(lambda x: x[0])
            df["long"] = coords.apply(lambda x: x[1])

            # Validate against Boston bounds
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

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize business and address strings."""
        for col in ["businessname", "address", "zip"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle remaining missing values."""
        categorical_cols = ["result", "businessname", "licenseno"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
        output_columns = [
            "_id", "businessname", "licenseno", "result", "resultdttm",
            "address", "zip", "lat", "long", "year", "month", "day_of_week"
        ]
        available_columns = [c for c in output_columns if c in df.columns]
        return df[available_columns].copy()


def preprocess_food_inspections_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing food inspections data."""
    preprocessor = FoodInspectionsPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
