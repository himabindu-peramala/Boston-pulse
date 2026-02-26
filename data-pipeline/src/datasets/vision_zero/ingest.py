"""
Boston Pulse - Vision Zero Safety Concerns Ingester

Fetches Vision Zero safety concerns from the Analyze Boston API using incremental logic.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.datasets.base import BaseIngester
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

# Load config
DATASET_CONFIG = get_dataset_config("vision_zero")
API_CONFIG = DATASET_CONFIG.get("api", {})
RESOURCE_ID = API_CONFIG.get("resource_id")
BASE_URL = API_CONFIG.get("base_url")
ENDPOINT = API_CONFIG.get("endpoint")
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)
BATCH_SIZE = API_CONFIG.get("batch_size", 1000)

INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "CreationDate")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "globalid")
LOOKBACK_DAYS = INGESTION_CONFIG.get("lookback_days", 30)


class VisionZeroIngester(BaseIngester):
    """Ingester for Vision Zero Safety Concerns."""

    def __init__(self, config: Settings | None = None):
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

    def get_dataset_name(self) -> str:
        return "vision_zero"

    def get_watermark_field(self) -> str:
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        return PRIMARY_KEY

    def _fetch_page(self, since: str, offset: int) -> list[dict]:
        """Fetch a page of Vision Zero records."""
        # Note: Since the date format in the datastore is M/D/Y,
        # standard string comparison might be tricky depending on the underlying DB.
        # However, Analyze Boston's SQL engine often handles common formats.
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"WHERE \"{WATERMARK_FIELD}\" >= '{since}' "
            f'ORDER BY "{WATERMARK_FIELD}" ASC '
            f"LIMIT {BATCH_SIZE} OFFSET {offset}"
        )

        response = requests.get(self.api_url, params={"sql": sql}, timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            raise ValueError(f"API Error: {data.get('error')}")

        return data.get("result", {}).get("records", [])

    def fetch_data(
        self, since: datetime | None = None, _until: datetime | None = None
    ) -> pd.DataFrame:
        """Fetch Vision Zero data incrementally."""
        if since is None:
            since = datetime.now(UTC) - timedelta(days=LOOKBACK_DAYS)

        # We'll use a simple date string for the SQL filter
        since_str = since.strftime("%Y-%m-%d")
        logger.info(f"Fetching Vision Zero data since {since_str}...")

        all_records = []
        offset = 0

        while True:
            records = self._fetch_page(since_str, offset)
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
        df = df.drop(columns=["_id", "_full_text"], errors="ignore")
        return df


def ingest_vision_zero(
    execution_date: str, watermark_start: datetime | None = None, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for ingesting vision zero data."""
    ingester = VisionZeroIngester(config)
    df = ingester.fetch_data(since=watermark_start)
    ingester._data = df
    result = ingester.run(execution_date, watermark_start)
    return result.to_dict()
