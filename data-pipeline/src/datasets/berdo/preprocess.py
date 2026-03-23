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
        "berdo_id": "berdo_id",
        "reporting_year": "reporting_year",
        "property_name": "property_name",
        "property_owner_name": "property_owner_name",
        "address": "address",
        "building_address_city": "building_address_city",
        "zip": "zip",
        "property_type": "property_type",
        "gross_floor_area": "gross_floor_area",
        "site_energy_use_kbtu": "site_energy_use_kbtu",
        "total_ghg_emissions": "total_ghg_emissions",
        "energy_star_score": "energy_star_score",
        "electricity_use_grid_purchase": "electricity_use_grid_purchase",
        "natural_gas_use": "natural_gas_use",
        "compliance_status": "compliance_status",
        "site_eui": "site_eui",
        "lat": "lat",
        "long": "long",
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
        return {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply BERDO transformations."""
        if df.empty:
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df

        # Rename columns to standardized names
        df = df.rename(columns=self.COLUMN_MAPPINGS)

        df = self._standardize_strings(df)
        df = self._process_numeric_fields(df)
        df = self._handle_missing_values(df)
        df = self.drop_duplicates(df, subset=["_id"], keep="last")
        df = self._select_output_columns(df)

        return df

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize string fields."""
        for col in [
            "property_owner_name",
            "building_address",
            "zip",
            "property_type",
            "compliance_status",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(["NAN", "NONE", "<NA>", "NULL"], np.nan)
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
            "site_eui",
            "lat",
            "long",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] < 0, col] = np.nan

        # Unit conversion: kWh to kBtu (1 kWh = 3.412 kBtu)
        if "electricity_use_grid_purchase" in df.columns:
            df["electricity_use_grid_purchase"] = df["electricity_use_grid_purchase"] * 3.41214
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        for col in ["property_type", "property_owner_name", "compliance_status"]:
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
        df = df[available].copy()

        # Ensure string columns are strictly strings (prevents float64 issues due to NaNs)
        string_cols = ["property_name", "address", "zip", "property_type"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype("string")

        return df


def preprocess_berdo_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing BERDO data."""
    preprocessor = BerdoPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
