"""
Boston Pulse - BERDO Data Ingester

Fetches Building Energy Reporting and Disclosure Ordinance (BERDO) data
from the Analyze Boston open data portal via direct file download.

Data Source:
    BERDO Reported Energy and Water Metrics
    https://data.boston.gov/dataset/building-emissions-reduction-and-disclosure-ordinance

Configuration:
    All settings loaded from configs/datasets/berdo.yaml
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
import requests

from src.datasets.base import BaseIngester
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration (loaded from berdo.yaml)
# =============================================================================
DATASET_CONFIG = get_dataset_config("berdo")

API_CONFIG = DATASET_CONFIG.get("api", {})
DOWNLOAD_URL = API_CONFIG.get(
    "download_url",
    "https://data.boston.gov/dataset/b09a8b71-274b-4365-9ce6-49b8b44602ef/resource/87521565-7f15-4b8d-a225-ac4df9e3f309/download/2024-reported-energy-and-water-metrics-1.xlsx",
)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)

INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "reporting_year")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "_id")


class BerdoIngester(BaseIngester):
    """
    Ingester for Boston BERDO data.

    Downloads BERDO energy and emissions data from the Analyze Boston
    open data portal as an Excel file and loads it into a DataFrame.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize BERDO ingester with config."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "berdo"

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
        """Fetch BERDO data from Analyze Boston via direct file download."""
        logger.info(f"Downloading BERDO data from {DOWNLOAD_URL}")

        try:
            response = requests.get(DOWNLOAD_URL, timeout=TIMEOUT)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

        logger.info("Parsing BERDO Excel file")
        df = pd.read_excel(BytesIO(response.content))

        # Standardize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Add a synthetic _id column if not present
        if "_id" not in df.columns:
            df.insert(0, "_id", range(1, len(df) + 1))

        # Drop footer/notes rows — filter out rows where berdo_id is a long string
        if "berdo_id" in df.columns:
            df = df[df["berdo_id"].astype(str).str.len() < 50].copy()
            df = df.reset_index(drop=True)

        # Convert all object columns to string to avoid pyarrow type errors
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)

        logger.info(f"Fetched {len(df)} BERDO records")
        return df


def ingest_berdo_data(
    execution_date: str,
    watermark_start: datetime | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for ingesting BERDO data."""
    ingester = BerdoIngester(config)

    until = datetime.strptime(execution_date, "%Y-%m-%d")
    since = watermark_start

    df = ingester.fetch_data(since=since, until=until)
    ingester._data = df

    result = ingester.run(execution_date, watermark_start)
    return result.to_dict()
