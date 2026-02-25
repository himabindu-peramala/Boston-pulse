"""
Boston Pulse - Hospital Locations Preprocessor

Cleans and validates Hospital Locations data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class HospitalPreprocessor(BasePreprocessor):
    """Preprocessor for Boston Hospital Locations data."""

    COLUMN_MAPPINGS = {
        "NAME": "name",
        "AD": "address",
        "NEIGH": "neighborhood",
        "XCOORD": "longitude",
        "YCOORD": "latitude",
    }

    DTYPE_MAPPINGS = {
        "name": "string",
        "address": "string",
        "neighborhood": "string",
        "latitude": "float",
        "longitude": "float",
    }

    REQUIRED_COLUMNS = ["name", "address", "latitude", "longitude"]

    def __init__(self, config: Settings | None = None):
        super().__init__(config)

    def get_dataset_name(self) -> str:
        return "hospitals"

    def get_column_mappings(self) -> dict[str, str]:
        return self.COLUMN_MAPPINGS

    def get_required_columns(self) -> list[str]:
        return self.REQUIRED_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hospital-specific transformations."""
        if df.empty:
            return df

        # Clean string fields
        for col in ["name", "address", "neighborhood"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()

        # Convert coordinates to numeric
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with invalid coordinates
        valid_mask = df["latitude"].notna() & df["longitude"].notna()
        if (~valid_mask).any():
            self.log_dropped_rows("invalid_coordinates", int((~valid_mask).sum()))
        df = df[valid_mask].copy()

        df["hospital_id"] = range(len(df))
        df["processed_at"] = datetime.now(UTC).isoformat()

        # Select and order columns
        output_cols = ["hospital_id"] + self.REQUIRED_COLUMNS + ["neighborhood", "processed_at"]
        return df[output_cols].copy()


def preprocess_hospitals(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for preprocessing hospitals."""
    preprocessor = HospitalPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
