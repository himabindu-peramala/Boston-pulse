"""
Boston Pulse - City Owned Property Preprocessor

Cleans and validates City Owned Property data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class CityPropertyPreprocessor(BasePreprocessor):
    """Preprocessor for Boston City Owned Property data."""

    COLUMN_MAPPINGS = {
        "OWNER": "owner",
        "FULL_ADDRE": "address",
    }

    DTYPE_MAPPINGS = {
        "owner": "string",
        "address": "string",
        "latitude": "float",
        "longitude": "float",
    }

    REQUIRED_COLUMNS = ["owner", "address", "latitude", "longitude"]

    def get_dataset_name(self) -> str:
        return "city_property"

    def get_column_mappings(self) -> dict[str, str]:
        return self.COLUMN_MAPPINGS

    def get_required_columns(self) -> list[str]:
        return self.REQUIRED_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply city-property transformations."""
        if df.empty:
            return df

        # Extract coordinates from the_geom (WKT)
        df = self._extract_wkt_coords(df)

        # Standardize strings
        for col in ["owner", "address"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()

        df["property_id"] = range(len(df))  # Generate a simple ID for the session
        df["processed_at"] = datetime.now(UTC).isoformat()

        # Select and validate
        df = df[["property_id"] + self.REQUIRED_COLUMNS + ["processed_at"]]
        return df

    def _extract_wkt_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract latitude/longitude from the_geom (WKT) geometry string."""
        from shapely import wkt

        df["latitude"] = None
        df["longitude"] = None

        if "the_geom" not in df.columns:
            return df

        def parse_wkt(wkt_str: str) -> tuple[float | None, float | None]:
            if not isinstance(wkt_str, str):
                return None, None
            try:
                geom = wkt.loads(wkt_str)
                centroid = geom.centroid
                return float(centroid.y), float(centroid.x)
            except Exception:
                return None, None

        coords = df["the_geom"].apply(parse_wkt)
        df[["latitude", "longitude"]] = pd.DataFrame(coords.tolist(), index=df.index)

        # Filter out rows where we couldn't extract coordinates
        valid_mask = df["latitude"].notna() & df["longitude"].notna()
        df = df[valid_mask].copy()

        return df


def preprocess_city_property(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for preprocessing city property."""
    preprocessor = CityPropertyPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
