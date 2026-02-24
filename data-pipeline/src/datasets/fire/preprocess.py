"""
Boston Pulse - Fire Data Preprocessor
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.datasets.base import BasePreprocessor
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class FirePreprocessor(BasePreprocessor):

    COLUMN_MAPPINGS = {
        "incident_number": "incident_number",
        "alarm_date": "alarm_date",
        "alarm_time": "alarm_time",
        "incident_type": "incident_type",
        "incident_description": "incident_description",
        "district": "district",
        "neighborhood": "neighborhood",
        "zip": "zip",
        "property_use": "property_use",
        "property_description": "property_description",
        "estimated_property_loss": "estimated_property_loss",
        "estimated_content_loss": "estimated_content_loss",
        "street_number": "street_number",
        "street_name": "street_name",
        "street_type": "street_type",
    }

    DTYPE_MAPPINGS = {
        "incident_number": "string",
        "alarm_date": "datetime",
        "alarm_time": "string",
        "incident_type": "string",
        "incident_description": "string",
        "district": "string",
        "neighborhood": "string",
        "zip": "string",
        "property_use": "string",
        "property_description": "string",
        "estimated_property_loss": "float",
        "estimated_content_loss": "float",
        "street_number": "string",
        "street_name": "string",
        "street_type": "string",
    }

    REQUIRED_COLUMNS = [
        "incident_number",
        "alarm_date",
        "incident_type",
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
        df = self._process_datetime(df)
        df = self._process_financials(df)
        df = self._standardize_categories(df)
        df = self._build_address(df)
        df = self._handle_missing_values(df)
        df = self.drop_duplicates(df, subset=["incident_number"], keep="last")
        df = self._select_output_columns(df)
        return df

    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if "alarm_date" in df.columns:
            df["alarm_date"] = pd.to_datetime(df["alarm_date"], errors="coerce")

            if df["alarm_date"].dt.tz is None:
                df["alarm_date"] = df["alarm_date"].dt.tz_localize(UTC)
            else:
                df["alarm_date"] = df["alarm_date"].dt.tz_convert(UTC)

            invalid = df["alarm_date"].isna().sum()
            if invalid > 0:
                self.log_dropped_rows("invalid_date", int(invalid))

            df = df[df["alarm_date"].notna()].copy()

            now = datetime.now(UTC)
            max_future_days = self.config.validation.temporal.max_future_days
            max_past_years = self.config.validation.temporal.max_past_years

            future_mask = df["alarm_date"] > now + pd.Timedelta(days=max_future_days)
            past_mask = df["alarm_date"] < now - pd.Timedelta(days=max_past_years * 365)
            df = df[~future_mask & ~past_mask].copy()

            # Extract time components
            df["year"] = df["alarm_date"].dt.year
            df["month"] = df["alarm_date"].dt.month
            df["day_of_week"] = df["alarm_date"].dt.day_name()

            # Parse hour from alarm_time if available
            if "alarm_time" in df.columns:
                df["hour"] = pd.to_datetime(
                    df["alarm_time"], format="%H:%M:%S", errors="coerce"
                ).dt.hour

            self.log_transformation("process_datetime")
        return df

    def _process_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["estimated_property_loss", "estimated_content_loss"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        self.log_transformation("process_financials")
        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if "incident_type" in df.columns:
            df["incident_type"] = df["incident_type"].str.strip()
            df["incident_type"] = df["incident_type"].fillna("Unknown")

        if "incident_description" in df.columns:
            df["incident_description"] = df["incident_description"].str.strip().str.title()
            df["incident_description"] = df["incident_description"].fillna("Unknown")

        if "district" in df.columns:
            df["district"] = df["district"].str.strip().str.upper()
            df["district"] = df["district"].fillna("UNKNOWN")

        if "neighborhood" in df.columns:
            df["neighborhood"] = df["neighborhood"].str.strip().str.title()
            df["neighborhood"] = df["neighborhood"].fillna("Unknown")

        self.log_transformation("standardize_categories")
        return df

    def _build_address(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col in ["street_number", "street_name", "street_type"]:
            if col in df.columns:
                parts.append(df[col].fillna("").str.strip())

        if parts:
            df["address"] = parts[0]
            for p in parts[1:]:
                df["address"] = df["address"] + " " + p
            df["address"] = df["address"].str.strip()
            df["address"] = df["address"].replace("", "Unknown")

        self.log_transformation("build_address")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        fill_values = {
            "neighborhood": "Unknown",
            "zip": "Unknown",
            "property_use": "Unknown",
            "property_description": "Unknown",
            "address": "Unknown",
        }
        for col, val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        output_columns = [
            "incident_number",
            "alarm_date",
            "alarm_time",
            "incident_type",
            "incident_description",
            "district",
            "neighborhood",
            "zip",
            "property_use",
            "property_description",
            "estimated_property_loss",
            "estimated_content_loss",
            "address",
            "year",
            "month",
            "day_of_week",
            "hour",
        ]
        available = [c for c in output_columns if c in df.columns]
        df = df[available].copy()
        self.log_transformation("select_output_columns")
        return df


def preprocess_fire_data(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    preprocessor = FirePreprocessor(config)
    result = preprocessor.run(df, execution_date)
    return result.to_dict()
