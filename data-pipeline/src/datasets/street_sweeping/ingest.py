"""
Boston Pulse - Street Sweeping Schedules Data Ingester

Fetches Street Sweeping Schedules data from the Analyze Boston API.

Data Source:
    Street Sweeping Schedules
    https://data.boston.gov/dataset/street-sweeping

Configuration:
    All settings loaded from configs/datasets/street_sweeping.yaml
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

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
RESOURCE_ID = API_CONFIG.get("resource_id", "5b4b5c1b-2c77-46e4-b9d6-e1888f36dd7e")
BASE_URL = API_CONFIG.get("base_url", "https://data.boston.gov/api/3/action")
ENDPOINT = API_CONFIG.get("endpoint", "datastore_search_sql")
BATCH_SIZE = API_CONFIG.get("batch_size", 1000)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)

INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "sam_street_id")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "_id")
LOOKBACK_DAYS = INGESTION_CONFIG.get("lookback_days", 30)


class StreetSweepingIngester(BaseIngester):
    """
    Ingester for Boston Street Sweeping Schedules data.

    Fetches data from the Analyze Boston API using the CKAN datastore_search_sql
    endpoint. Supports full and incremental ingestion.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize street sweeping ingester with config."""
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

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
        """Get the API endpoint."""
        return self.api_url

    def _fetch_page(self, offset: int) -> list[dict]:
        """Fetch one page of records."""
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f'ORDER BY "{PRIMARY_KEY}" ASC '
            f"LIMIT {BATCH_SIZE} OFFSET {offset}"
        )

        response = requests.get(
            self.api_url,
            params={"sql": sql},
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        if not data.get("success"):
            error = data.get("error", {})
            raise ValueError(f"API error: {error}")

        return data.get("result", {}).get("records", [])

    def fetch_data(
        self, since: datetime | None = None, until: datetime | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        """Fetch street sweeping data from Analyze Boston API."""
        logger.info("Fetching street sweeping schedule data")

        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            try:
                records = self._fetch_page(offset)
            except Exception as e:
                logger.error(f"API request failed: {e}")
                raise

            if not records:
                break

            all_records.extend(records)
            if len(records) < BATCH_SIZE:
                break

            offset += len(records)
            time.sleep(0.5)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.drop(columns=["_full_text"], errors="ignore")
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
