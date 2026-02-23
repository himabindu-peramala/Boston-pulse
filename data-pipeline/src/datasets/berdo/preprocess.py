"""
Boston Pulse - BERDO Data Preprocessor

Cleans and validates BERDO building energy and emissions data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class BerdoPreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston BERDO data.
    """

    COLUMN_MAPPINGS = {
        "_id": "_id",
        "reporting_year": "reporting_year",
        "property_name": "property_name",
        "address": "address",
        "zip": "zip",
        "property_type": "property_type",
        "gross_floor_area": "gross_floor_area",
        "site_energy_use_kbtu": "site_energy_use_kbtu",
        "total_ghg_emissions": "total_ghg_emissions",
        "energy_star_score": "energy_star_score",
        "electricity_use_grid_purchase": "electricity_use_grid_purchase",
        "natural_gas_use": "natural_gas_use",
        "lat": "lat",
        "long": "long",
    }

    DTYPE_MAPPINGS = {
        "_id": "int",
        "reporting_year": "int",
        "property_name": "string",
        "address": "string",
        "zip": "string",
        "property_type": "string",
        "gross_floor_area": "float",
        "site_energy_use_kbtu": "float",
        "total_ghg_emissions": "float",
        "energy_star_score": "float",
        "electricity_use_grid_purchase": "float",
        "natural_gas_use": "float",
        "lat": "float",
        "long": "float",
    }

    REQUIRED_COLUMNS = [
        "_id",
        "reporting_year",
        "property_name",
        "address",
        "property_type",
        "total_ghg_emissions",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize BERDO preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "berdo"

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
        """Apply BERDO transformations."""
        if df.empty:
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df

        df = self._standardize_strings(df)
        df = self._process_numeric_fields(df)
        df = self._validate_coordinates(df)
        df = self._handle_missing_values(df)
        df = self.drop_duplicates(df, subset=["_id"], keep="last")
        df = self._select_output_columns(df)

        return df

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize string fields."""
        for col in ["property_name", "address", "zip", "property_type"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace("NAN", np.nan)
        return df

    def _process_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric fields and handle invalid values."""
        numeric_cols = [
            "gross_floor_area",
            "site_energy_use_kbtu",
            "total_ghg_emissions",
            "energy_star_score",
            "electricity_use_grid_purchase",
            "natural_gas_use",
            "reporting_year",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if col not in ["reporting_year"]:
                    df.loc[df[col] < 0, col] = np.nan
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
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        for col in ["property_type", "property_name"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
        output_columns = [
            "_id",
            "reporting_year",
            "property_name",
            "address",
            "zip",
            "property_type",
            "gross_floor_area",
            "site_energy_use_kbtu",
            "total_ghg_emissions",
            "energy_star_score",
            "electricity_use_grid_purchase",
            "natural_gas_use",
            "lat",
            "long",
        ]
        available = [c for c in output_columns if c in df.columns]
        return df[available].copy()


def preprocess_berdo_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing BERDO data."""
    preprocessor = BerdoPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
