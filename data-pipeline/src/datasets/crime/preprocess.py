"""
Boston Pulse - Crime Data Preprocessor

Cleans and validates crime incident data from Boston PD.

Transformations:
    - Column renaming to standardized names
    - Date/time parsing and validation
    - Geographic coordinate validation
    - Category standardization
    - Missing value handling

Usage:
    from src.datasets.crime.preprocess import CrimePreprocessor

    preprocessor = CrimePreprocessor()
    result = preprocessor.run(raw_df, execution_date="2024-01-15")
    processed_df = preprocessor.get_data()
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


class CrimePreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston crime incident data.

    Handles data cleaning, type conversion, and validation for
    crime data fetched from Analyze Boston API.
    """

    # Column mapping from raw API names to standardized names
    COLUMN_MAPPINGS = {
        "INCIDENT_NUMBER": "incident_number",
        "OFFENSE_CODE": "offense_code",
        "OFFENSE_CODE_GROUP": "offense_category",
        "OFFENSE_DESCRIPTION": "offense_description",
        "DISTRICT": "district",
        "REPORTING_AREA": "reporting_area",
        "SHOOTING": "shooting",
        "OCCURRED_ON_DATE": "occurred_on_date",
        "YEAR": "year",
        "MONTH": "month",
        "DAY_OF_WEEK": "day_of_week",
        "HOUR": "hour",
        "UCR_PART": "ucr_part",
        "STREET": "street",
        "Lat": "lat",
        "Long": "long",
        "Location": "location",
    }

    # Data type mappings
    DTYPE_MAPPINGS = {
        "incident_number": "string",
        "offense_code": "int",
        "offense_category": "string",
        "offense_description": "string",
        "district": "string",
        "reporting_area": "string",
        "shooting": "bool",
        "occurred_on_date": "datetime",
        "year": "int",
        "month": "int",
        "hour": "int",
        "ucr_part": "string",
        "street": "string",
        "lat": "float",
        "long": "float",
    }

    # Required output columns
    REQUIRED_COLUMNS = [
        "incident_number",
        "offense_code",
        "offense_category",
        "occurred_on_date",
        "district",
        "lat",
        "long",
        "year",
        "month",
        "hour",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize crime preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "crime"

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
        """
        Apply crime-specific transformations.

        Args:
            df: Raw DataFrame with renamed columns

        Returns:
            Cleaned and validated DataFrame
        """
        # Convert shooting field to boolean
        df = self._process_shooting_field(df)

        # Process and validate datetime
        df = self._process_datetime(df)

        # Process and validate coordinates
        df = self._process_coordinates(df)

        # Standardize categories
        df = self._standardize_categories(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop duplicates
        df = self.drop_duplicates(df, subset=["incident_number"], keep="last")

        # Final column selection
        df = self._select_output_columns(df)

        return df

    def _process_shooting_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the shooting field to boolean."""
        if "shooting" in df.columns:
            # Handle various representations: Y/N, 1/0, True/False, etc.
            df["shooting"] = df["shooting"].apply(
                lambda x: str(x).upper() in ["Y", "1", "TRUE", "YES"]
            )
            self.log_transformation("convert_shooting_to_boolean")
        return df

    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate datetime fields."""
        if "occurred_on_date" in df.columns:
            # Convert to datetime and localize to UTC
            df["occurred_on_date"] = pd.to_datetime(df["occurred_on_date"], errors="coerce")
            if not df["occurred_on_date"].dt.tz:
                df["occurred_on_date"] = df["occurred_on_date"].dt.tz_localize(UTC)
            else:
                df["occurred_on_date"] = df["occurred_on_date"].dt.tz_convert(UTC)

            # Track invalid dates
            invalid_dates = df["occurred_on_date"].isna().sum()
            if invalid_dates > 0:
                self.log_dropped_rows("invalid_date", int(invalid_dates))

            # Remove records with invalid dates
            df = df[df["occurred_on_date"].notna()].copy()

            # Validate temporal bounds
            now = datetime.now(UTC)
            max_future_days = self.config.validation.temporal.max_future_days
            max_past_years = self.config.validation.temporal.max_past_years

            future_mask = df["occurred_on_date"] > now + pd.Timedelta(days=max_future_days)
            past_mask = df["occurred_on_date"] < now - pd.Timedelta(days=max_past_years * 365)

            future_count = future_mask.sum()
            past_count = past_mask.sum()

            if future_count > 0:
                self.log_dropped_rows("future_date", int(future_count))
            if past_count > 0:
                self.log_dropped_rows("old_date", int(past_count))

            df = df[~future_mask & ~past_mask].copy()

            # Extract temporal components if not present
            if "year" not in df.columns or df["year"].isna().all():
                df["year"] = df["occurred_on_date"].dt.year
            if "month" not in df.columns or df["month"].isna().all():
                df["month"] = df["occurred_on_date"].dt.month
            if "hour" not in df.columns or df["hour"].isna().all():
                df["hour"] = df["occurred_on_date"].dt.hour
            if "day_of_week" not in df.columns or df["day_of_week"].isna().all():
                df["day_of_week"] = df["occurred_on_date"].dt.day_name()

            self.log_transformation("process_datetime")

        return df

    def _process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate geographic coordinates."""
        if "lat" in df.columns and "long" in df.columns:
            # Convert to numeric
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["long"] = pd.to_numeric(df["long"], errors="coerce")

            # Get Boston bounds from config
            bounds = self.config.validation.geo_bounds

            # Track coordinate issues
            missing_coords = (df["lat"].isna() | df["long"].isna()).sum()

            # Check for coordinates outside Boston
            out_of_bounds = (
                (df["lat"] < bounds.min_lat)
                | (df["lat"] > bounds.max_lat)
                | (df["long"] < bounds.min_lon)
                | (df["long"] > bounds.max_lon)
            )
            out_of_bounds_count = (out_of_bounds & df["lat"].notna()).sum()

            if missing_coords > 0:
                logger.warning(f"Found {missing_coords} records with missing coordinates")
                # Don't drop - fill with district centroid later or keep as is

            if out_of_bounds_count > 0:
                logger.warning(
                    f"Found {out_of_bounds_count} records with coordinates outside Boston"
                )
                # Set invalid coordinates to NaN
                df.loc[out_of_bounds, "lat"] = np.nan
                df.loc[out_of_bounds, "long"] = np.nan
                self.log_transformation("nullify_invalid_coordinates")

            self.log_transformation("validate_coordinates")

        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical fields."""
        # Standardize district codes
        if "district" in df.columns:
            df["district"] = df["district"].str.strip().str.upper()

            # Map known district variations
            district_map = {
                "": "UNKNOWN",
                " ": "UNKNOWN",
            }
            df["district"] = df["district"].replace(district_map)
            df["district"] = df["district"].fillna("UNKNOWN")
            self.log_transformation("standardize_district")

        # Standardize offense category
        if "offense_category" in df.columns:
            df["offense_category"] = df["offense_category"].str.strip().str.title()
            df["offense_category"] = df["offense_category"].fillna("Other")
            self.log_transformation("standardize_offense_category")

        # Standardize UCR part
        if "ucr_part" in df.columns:
            df["ucr_part"] = df["ucr_part"].str.strip().str.upper()
            self.log_transformation("standardize_ucr_part")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle remaining missing values."""
        # Fill categorical missing values
        fill_values = {
            "offense_description": "Unknown",
            "street": "Unknown",
            "reporting_area": "Unknown",
        }

        for col, fill_val in fill_values.items():
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(fill_val)
                    self.log_transformation(f"fill_missing_{col}")

        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
        output_columns = [
            "incident_number",
            "offense_code",
            "offense_category",
            "offense_description",
            "district",
            "reporting_area",
            "shooting",
            "occurred_on_date",
            "year",
            "month",
            "day_of_week",
            "hour",
            "ucr_part",
            "street",
            "lat",
            "long",
        ]

        # Only include columns that exist
        available_columns = [c for c in output_columns if c in df.columns]
        df = df[available_columns].copy()

        self.log_transformation("select_output_columns")
        return df


# =============================================================================
# Convenience Functions
# =============================================================================


def preprocess_crime_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Convenience function for preprocessing crime data.

    Returns result dictionary suitable for Airflow XCom.
    """
    preprocessor = CrimePreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
