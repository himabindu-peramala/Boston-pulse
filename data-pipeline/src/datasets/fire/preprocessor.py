"""
Boston Pulse - Fire Data Preprocessor

Cleans and validates fire incident data.
"""

from __future__ import annotations

import logging
import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class FirePreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston fire incident data.

    Mirrors CrimePreprocessor structure.
    """

    # Example column mappings (adjust to match your raw fire dataset)
    COLUMN_MAPPINGS = {
        "INCIDENT_ID": "incident_id",
        "DISTRICT": "district",
        "INCIDENT_DATE": "incident_date",
        "SEVERITY": "severity",
        "TOTAL_LOSS": "total_loss",
        "LAT": "lat",
        "LONG": "long",
    }

    DTYPE_MAPPINGS = {
        "incident_id": "string",
        "district": "string",
        "incident_date": "datetime",
        "severity": "string",
        "total_loss": "float",
        "lat": "float",
        "long": "float",
    }

    REQUIRED_COLUMNS = [
        "incident_id",
        "incident_date",
        "district",
    ]

    def __init__(self, config: Settings | None = None):
        super().__init__(config)

    def get_dataset_name(self) -> str:
        return "fire"

    def get_required_columns(self) -> list[str]:
        return self.REQUIRED_COLUMNS

    def get_column_mappings(self) -> dict[str, str]:
        return self.COLUMN_MAPPINGS

    def get_dtype_mappings(self) -> dict[str, str]:
        return self.DTYPE_MAPPINGS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fire-specific transformations.
        """

        # Convert datetime
        if "incident_date" in df.columns:
            df["incident_date"] = pd.to_datetime(
                df["incident_date"],
                errors="coerce",
            )

            # Drop invalid dates
            df = df[df["incident_date"].notna()].copy()

        # Standardize district
        if "district" in df.columns:
            df["district"] = df["district"].str.strip().str.upper()
            df["district"] = df["district"].fillna("UNKNOWN")

        # Drop duplicates by primary key
        if "incident_id" in df.columns:
            df = self.drop_duplicates(df, subset=["incident_id"], keep="last")

        return df
