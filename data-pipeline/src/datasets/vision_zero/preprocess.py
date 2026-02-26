"""
Boston Pulse - Vision Zero Preprocessor

Cleans and validates Vision Zero safety concerns.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class VisionZeroPreprocessor(BasePreprocessor):
    """Preprocessor for Boston Vision Zero Safety Concerns."""

    COLUMN_MAPPINGS = {
        "globalid": "concern_id",
        "CreationDate": "creation_date",
        "request": "request_type",
        "your_mode_of_transportation": "mode",
        "additional_comments": "comments",
    }

    DTYPE_MAPPINGS = {
        "concern_id": "string",
        "creation_date": "datetime",
        "request_type": "string",
        "mode": "string",
        "comments": "string",
        "latitude": "float",
        "longitude": "float",
    }

    REQUIRED_COLUMNS = ["concern_id", "creation_date", "request_type", "latitude", "longitude"]

    def get_dataset_name(self) -> str:
        return "vision_zero"

    def get_column_mappings(self) -> dict[str, str]:
        return self.COLUMN_MAPPINGS

    def get_required_columns(self) -> list[str]:
        return self.REQUIRED_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply vision-zero transformations."""
        if df.empty:
            return df

        # Process dates (handle M/D/Y format)
        if "creation_date" in df.columns:
            df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
            if df["creation_date"].dt.tz is None:
                df["creation_date"] = df["creation_date"].dt.tz_localize(UTC)
            else:
                df["creation_date"] = df["creation_date"].dt.tz_convert(UTC)

        # Extract coordinates from POINT_X/Y if available, else shape_wkt
        if "POINT_X" in df.columns and "POINT_Y" in df.columns:
            df["longitude"] = pd.to_numeric(df["POINT_X"], errors="coerce")
            df["latitude"] = pd.to_numeric(df["POINT_Y"], errors="coerce")
        elif "shape_wkt" in df.columns:
            df = self._extract_wkt_coords(df)

        # Standardize strings
        for col in ["request_type", "mode"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()

        df["processed_at"] = datetime.now(UTC).isoformat()

        # Select and validate
        df = df[self.REQUIRED_COLUMNS + ["mode", "comments", "processed_at"]]
        return df

    def _extract_wkt_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract coords from WKT if POINT_X/Y failed."""
        from shapely import wkt

        def parse_wkt(wkt_str: str):
            if not isinstance(wkt_str, str):
                return None, None
            try:
                geom = wkt.loads(wkt_str)
                centroid = geom.centroid
                return float(centroid.y), float(centroid.x)
            except Exception:
                return None, None

        coords = df["shape_wkt"].apply(parse_wkt)
        df[["latitude", "longitude"]] = pd.DataFrame(coords.tolist(), index=df.index)
        return df


def preprocess_vision_zero(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for preprocessing vision zero data."""
    preprocessor = VisionZeroPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
