"""
Boston Pulse - Snow Emergency Routes Preprocessor

Cleans and validates Snow Emergency Routes data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class SnowRoutesPreprocessor(BasePreprocessor):
    """Preprocessor for Boston Snow Emergency Routes data."""

    COLUMN_MAPPINGS = {
        "FULL_NAME": "street_name",
        "Responsibility": "responsibility",
    }

    DTYPE_MAPPINGS = {
        "street_name": "string",
        "responsibility": "string",
        "latitude": "float",
        "longitude": "float",
    }

    REQUIRED_COLUMNS = ["street_name", "latitude", "longitude"]

    def get_dataset_name(self) -> str:
        return "snow_routes"

    def get_column_mappings(self) -> dict[str, str]:
        return self.COLUMN_MAPPINGS

    def get_required_columns(self) -> list[str]:
        return self.REQUIRED_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply snow-routes transformations."""
        if df.empty:
            return df

        # Extract coordinates from shape_wkt
        df = self._extract_wkt_coords(df)

        # Standardize strings
        if "street_name" in df.columns:
            df["street_name"] = df["street_name"].str.strip().str.title()

        df["processed_at"] = datetime.now(UTC).isoformat()

        # Select and validate
        df = df[self.REQUIRED_COLUMNS + ["responsibility", "processed_at"]]
        return df

    def _extract_wkt_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract latitude/longitude from WKT geometry string using shapely."""
        from shapely import wkt

        df["latitude"] = None
        df["longitude"] = None

        if "shape_wkt" not in df.columns:
            return df

        def parse_wkt(wkt_str: str) -> tuple[float | None, float | None]:
            if not isinstance(wkt_str, str):
                return None, None
            try:
                geom = wkt.loads(wkt_str)
                # For points, it's direct. For lines/polygons, use centroid.
                centroid = geom.centroid
                return float(centroid.y), float(centroid.x)
            except Exception as e:
                logger.debug(f"WKT parse error: {e}")
                return None, None

        coords = df["shape_wkt"].apply(parse_wkt)
        df[["latitude", "longitude"]] = pd.DataFrame(coords.tolist(), index=df.index)

        # Filter out rows where we couldn't extract coordinates
        valid_mask = df["latitude"].notna() & df["longitude"].notna()
        if (~valid_mask).any():
            self.log_dropped_rows("invalid_geometry", int((~valid_mask).sum()))

        df = df[valid_mask].copy()

        return df


def preprocess_snow_routes(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for preprocessing snow routes."""
    preprocessor = SnowRoutesPreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
