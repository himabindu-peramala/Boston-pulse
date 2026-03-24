"""
Boston Pulse - Street Sweeping Schedules Data Ingester

Fetches Street Sweeping Schedules data from the Analyze Boston open data portal
via direct CSV file download.

Data Source:
    Street Sweeping Schedules
    https://data.boston.gov/dataset/street-sweeping-schedules

Configuration:
    All settings loaded from configs/datasets/street_sweeping.yaml
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.datasets.base import BaseIngester
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration (loaded from street_sweeping.yaml)
# =============================================================================
DATASET_CONFIG = get_dataset_config("street_sweeping")

API_CONFIG = DATASET_CONFIG.get("api", {})
DOWNLOAD_URL = API_CONFIG.get(
    "download_url",
    "https://data.boston.gov/dataset/00c015a1-2b62-4072-a71e-79b292ce9670/resource/9fdbdcad-67c8-4b23-b6ec-861e77d56227/download/tmpoij9sywp.csv",
)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)

INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "sam_street_id")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "_id")


class StreetSweepingIngester(BaseIngester):
    """
    Ingester for Boston Street Sweeping Schedules data.

    Downloads street sweeping schedule data from the Analyze Boston
    open data portal as a CSV file and loads it into a DataFrame.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize street sweeping ingester with config."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "street_sweeping"

    def get_watermark_field(self) -> str:
        """Return the field used for incremental ingestion."""
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        """Return the primary key field."""
        return PRIMARY_KEY

    def get_api_endpoint(self) -> str:
        """Get the download URL."""
        return DOWNLOAD_URL

    def fetch_data(
        self, since: datetime | None = None, until: datetime | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        """Fetch street sweeping data from Analyze Boston via direct CSV download."""
        logger.info(f"Downloading street sweeping data from {DOWNLOAD_URL}")

        try:
            response = requests.get(DOWNLOAD_URL, timeout=TIMEOUT)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

        logger.info("Parsing street sweeping CSV file")
        df = pd.read_csv(StringIO(response.text))

        # Standardize column names for RAW stage (must match raw_schema.json)
        # 1. Normalize headers first
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # 2. Map normalized names to schema names
        raw_mapping = {
            "main_id": "sam_street_id",
            "st_name": "full_street_name",
            "from": "from_street",
            "to": "to_street",
            "dist": "district",
            "side": "side_of_street",
        }
        df = df.rename(columns=raw_mapping)

        # Add missing fields required by raw_schema.json
        if "season_start" not in df.columns:
            df["season_start"] = "April"
        if "season_end" not in df.columns:
            df["season_end"] = "November"
        if "week_type" not in df.columns:
            df["week_type"] = "Every Week"
        if "tow_zone" not in df.columns:
            df["tow_zone"] = "YES"

        # Add lat/long if missing (Use NaN to satisfy schema while indicating no geo data)
        if "lat" not in df.columns:
            df["lat"] = np.nan
        if "long" not in df.columns:
            df["long"] = np.nan

        # Critical: Add a synthetic _id column (Required for primary key and preprocessor)
        if "_id" not in df.columns:
            df.insert(0, "_id", range(1, len(df) + 1))

        # Only keep columns expected by the raw schema
        expected_cols = {
            "_id",
            "sam_street_id",
            "full_street_name",
            "from_street",
            "to_street",
            "district",
            "side_of_street",
            "season_start",
            "season_end",
            "week_type",
            "tow_zone",
            "lat",
            "long",
        }
        available_cols = [c for c in df.columns if c in expected_cols]
        df = df[available_cols].copy()

        # Convert all object columns to string to avoid pyarrow type errors
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", np.nan)

        logger.info(f"Fetched {len(df)} street sweeping records after filtering")
        return df


def ingest_street_sweeping_data(
    execution_date: str,
    watermark_start: datetime | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for ingesting street sweeping data."""
    ingester = StreetSweepingIngester(config)

    until = datetime.strptime(execution_date, "%Y-%m-%d")
    since = watermark_start

    df = ingester.fetch_data(since=since, until=until)
    ingester._data = df

    result = ingester.run(execution_date, watermark_start)
    return result.to_dict()
