"""
Boston Pulse - 311 Data Preprocessor

Cleans and validates 311 service request data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class Service311Preprocessor(BasePreprocessor):
    """
    Preprocessor for Boston 311 service request data.
    """

    # Column mapping from raw API names to standardized names
    COLUMN_MAPPINGS = {
        "case_id": "case_id",
        "open_date": "open_date",
        "close_date": "close_date",
        "case_topic": "case_topic",
        "service_name": "service_name",
        "assigned_department": "assigned_department",
        "case_status": "case_status",
        "neighborhood": "neighborhood",
        "latitude": "lat",
        "longitude": "long",
        "on_time": "on_time",}

    # Data type mappings
    DTYPE_MAPPINGS = {
        "case_id": "string",
        "open_date": "datetime",
        "close_date": "datetime",
        "case_topic": "string",
        "service_name": "string",
        "assigned_department": "string",
        "case_status": "string",
        "neighborhood": "string",
        "lat": "float",
        "long": "float",
        "on_time": "string",  # Often 'ON TIME' or 'OVERDUE'}

    # Required output columns
    REQUIRED_COLUMNS = [
        "case_id",
        "open_date",
        "case_topic",
        "neighborhood",
        "lat",
        "long",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize 311 preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "311"

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
        """Apply 311-specific transformations."""
        if df.empty:
            # Ensure required columns exist even in empty DF
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    target_dtype = self.DTYPE_MAPPINGS.get(col, "object")
                    # Map 'datetime' to the actual numpy/pandas dtype
                    if target_dtype == "datetime":
                        target_dtype = "datetime64[ns, UTC]"
                    df[col] = pd.Series(dtype=target_dtype)
            return df

        # Process and validate datetime
        df = self._process_datetimes(df)

        # Process and validate coordinates
        df = self._process_coordinates(df)

        # Standardize neighborhoods
        df = self._standardize_neighborhoods(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop duplicates
        df = self.drop_duplicates(df, subset=["case_id"], keep="last")

        # Final column selection
        df = self._select_output_columns(df)

        return df

    def _process_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate datetime fields."""
        for col in ["open_date", "close_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if not df[col].dt.tz:
                    df[col] = df[col].dt.tz_localize(UTC)
                else:
                    df[col] = df[col].dt.tz_convert(UTC)

        if "open_date" in df.columns:
            # Remove records with invalid open dates
            invalid_dates = df["open_date"].isna().sum()
            if invalid_dates > 0:
                self.log_dropped_rows("invalid_open_date", int(invalid_dates))
            df = df[df["open_date"].notna()].copy()

            # Validate temporal bounds
            now = datetime.now(UTC)
            max_future_days = self.config.validation.temporal.max_future_days
            max_past_years = self.config.validation.temporal.max_past_years

            future_mask = df["open_date"] > now + pd.Timedelta(days=max_future_days)
            past_mask = df["open_date"] < now - pd.Timedelta(days=max_past_years * 365)

            df = df[~future_mask & ~past_mask].copy()

            # Add temporal components for aggregation
            df["year"] = df["open_date"].dt.year
            df["month"] = df["open_date"].dt.month
            df["hour"] = df["open_date"].dt.hour
            df["day_of_week"] = df["open_date"].dt.day_name()

        return df

    def _process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate geographic coordinates."""
        if "lat" in df.columns and "long" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["long"] = pd.to_numeric(df["long"], errors="coerce")

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

    def _standardize_neighborhoods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize neighborhood names."""
        if "neighborhood" in df.columns:
            df["neighborhood"] = df["neighborhood"].str.strip().str.title()
            df["neighborhood"] = df["neighborhood"].fillna("Unknown")
            self.log_transformation("standardize_neighborhood")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle remaining missing values."""
        categorical_cols = [
            "case_topic",
            "service_name",
            "assigned_department",
            "case_status",
            "on_time",
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
        output_columns = [
            "case_id",
            "open_date",
            "close_date",
            "case_topic",
            "service_name",
            "assigned_department",
            "case_status",
            "neighborhood",
            "on_time",
            "lat",
            "long",
            "year",
            "month",
            "hour",
            "day_of_week",
        ]
        available_columns = [c for c in output_columns if c in df.columns]
        return df[available_columns].copy()


def preprocess_311_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing 311 data."""
    preprocessor = Service311Preprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
