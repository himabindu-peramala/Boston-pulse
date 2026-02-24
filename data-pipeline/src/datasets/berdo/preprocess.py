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
        "property_owner_name": "property_owner_name",
        "building_address": "building_address",
        "building_address_city": "building_address_city",
        "building_address_zip__code": "zip",
        "largest_property_type": "property_type",
        "reported_gross_floor_area_(sq_ft)": "gross_floor_area",
        "total_site_energy_usage_(kbtu)": "site_energy_use_kbtu",
        "estimated_total_ghg_emissions_(kgco2e)": "total_ghg_emissions",
        "energy_star_score": "energy_star_score",
        "electricity_usage_(kwh)": "electricity_usage_kwh",
        "natural_gas_usage_(kbtu)": "natural_gas_use",
        "compliance_status": "compliance_status",
        "site_eui_(energy_use_intensity_kbtu/ft2)": "site_eui",
    }

    REQUIRED_COLUMNS = [
        "_id",
        "berdo_id",
        "property_owner_name",
        "building_address",
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
                df[col] = df[col].replace("NAN", np.nan)
        return df

    def _process_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric fields and handle invalid values."""
        numeric_cols = [
            "gross_floor_area",
            "site_energy_use_kbtu",
            "total_ghg_emissions",
            "energy_star_score",
            "electricity_usage_kwh",
            "natural_gas_use",
            "site_eui",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] < 0, col] = np.nan
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
            "berdo_id",
            "property_owner_name",
            "building_address",
            "building_address_city",
            "zip",
            "property_type",
            "gross_floor_area",
            "site_energy_use_kbtu",
            "site_eui",
            "total_ghg_emissions",
            "energy_star_score",
            "electricity_usage_kwh",
            "natural_gas_use",
            "compliance_status",
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
