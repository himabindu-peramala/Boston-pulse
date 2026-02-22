"""
Boston Pulse - CityScore Data Preprocessor

Cleans and validates CityScore metric data.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class CityScorePreprocessor(BasePreprocessor):
    """
    Preprocessor for Boston CityScore data.
    """

    # Column mapping from raw API names to standardized names
    COLUMN_MAPPINGS = {
        "metric_name": "metric_name",
        "score_calculated_ts": "timestamp",
        "target": "target",
        "metric_logic": "metric_logic",
        "day_score": "day_score",
        "day_numerator": "day_numerator",
        "day_denominator": "day_denominator",
        "week_score": "week_score",
        "month_score": "month_score",
    }

    # Data type mappings
    DTYPE_MAPPINGS = {
        "metric_name": "string",
        "timestamp": "datetime",
        "target": "float",
        "day_score": "float",
        "week_score": "float",
        "month_score": "float",
    }

    # Required output columns
    REQUIRED_COLUMNS = [
        "metric_name",
        "timestamp",
        "day_score",
    ]

    def __init__(self, config: Settings | None = None):
        """Initialize CityScore preprocessor."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "cityscore"

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
        """Apply CityScore-specific transformations."""
        if df.empty:
            for col in self.REQUIRED_COLUMNS:
                if col not in df.columns:
                    target_dtype = self.DTYPE_MAPPINGS.get(col, "object")
                    if target_dtype == "datetime":
                        target_dtype = "datetime64[ns, UTC]"
                    df[col] = pd.Series(dtype=target_dtype)
            return df

        # Process and validate timestamp
        df = self._process_timestamp(df)

        # Standardize metric names
        df = self._standardize_metrics(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop duplicates
        df = self.drop_duplicates(df, subset=["metric_name", "timestamp"], keep="last")

        # Final column selection
        df = self._select_output_columns(df)

        return df

    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamp field."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if not df["timestamp"].dt.tz:
                df["timestamp"] = df["timestamp"].dt.tz_localize(UTC)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert(UTC)

            # Remove records with invalid timestamps
            invalid_dates = df["timestamp"].isna().sum()
            if invalid_dates > 0:
                self.log_dropped_rows("invalid_timestamp", int(invalid_dates))
            df = df[df["timestamp"].notna()].copy()

            # Add temporal components
            df["year"] = df["timestamp"].dt.year
            df["month_num"] = df["timestamp"].dt.month
            df["date"] = df["timestamp"].dt.date

        return df

    def _standardize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize metric names."""
        if "metric_name" in df.columns:
            df["metric_name"] = df["metric_name"].str.strip().str.title()
            self.log_transformation("standardize_metric_names")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in scores."""
        numeric_cols = ["target", "day_score", "week_score", "month_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order output columns."""
        output_columns = [
            "metric_name",
            "timestamp",
            "target",
            "day_score",
            "week_score",
            "month_score",
            "year",
            "month_num",
            "date",
        ]
        available_columns = [c for c in output_columns if c in df.columns]
        return df[available_columns].copy()


def preprocess_cityscore_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for preprocessing CityScore data."""
    preprocessor = CityScorePreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
