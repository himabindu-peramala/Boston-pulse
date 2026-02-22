"""
Boston Pulse - CityScore Data Ingester

Fetches CityScore metric data from the Analyze Boston API.

Data Source:
    CityScore Full Metric List
    https://data.boston.gov/dataset/cityscore

Configuration:
    All settings loaded from configs/datasets/cityscore.yaml
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

# =============================================================================
# API Configuration (loaded from cityscore.yaml)
# =============================================================================
DATASET_CONFIG = get_dataset_config("cityscore")

# API settings
API_CONFIG = DATASET_CONFIG.get("api", {})
RESOURCE_ID = API_CONFIG.get("resource_id", "dd657c02-3443-4c00-8b29-56a40cfe7ee4")
BASE_URL = API_CONFIG.get("base_url", "https://data.boston.gov/api/3/action")
ENDPOINT = API_CONFIG.get("endpoint", "datastore_search_sql")
BATCH_SIZE = API_CONFIG.get("batch_size", 1000)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)

# Ingestion settings
INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "score_calculated_ts")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "_id")
LOOKBACK_DAYS = INGESTION_CONFIG.get("lookback_days", 30)


class CityScoreIngester(BaseIngester):
    """
    Ingester for Boston CityScore data.

    Fetches data from the Analyze Boston API using the CKAN datastore_search_sql
    endpoint. Supports incremental ingestion via watermark filtering.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize CityScore ingester with config from cityscore.yaml."""
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "cityscore"

    def get_watermark_field(self) -> str:
        """Return the field used for incremental ingestion."""
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        """Return the primary key field."""
        return PRIMARY_KEY

    def get_api_endpoint(self) -> str:
        """Get the API endpoint for CityScore data."""
        return self.api_url

    def _fetch_page(self, since: str, until: str, offset: int) -> list[dict]:
        """Fetch one page of CityScore records."""
        # Note: score_calculated_ts is a timestamp type in the API
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"WHERE \"{WATERMARK_FIELD}\" >= '{since}' "
            f"AND \"{WATERMARK_FIELD}\" <= '{until}' "
            f'ORDER BY "{WATERMARK_FIELD}" ASC '
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
        self, since: datetime | None = None, until: datetime | None = None
    ) -> pd.DataFrame:
        """Fetch CityScore data from Analyze Boston API."""
        if until is None:
            until = datetime.now(UTC)

        if since is None:
            since = until - timedelta(days=LOOKBACK_DAYS)

        # Use ISO format for timestamp filtering
        since_str = since.isoformat()
        until_str = until.isoformat()

        logger.info(f"Fetching CityScore data from {since_str} to {until_str}")

        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            try:
                records = self._fetch_page(since_str, until_str, offset)
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
        # Note: we keep _id here as it's our primary key for this dataset
        return df

    def fetch_sample_data(self, n: int = 1000) -> pd.DataFrame:
        """Fetch a sample of CityScore data."""
        sql = f'SELECT * FROM "{RESOURCE_ID}" ORDER BY "{WATERMARK_FIELD}" DESC LIMIT {n}'
        response = requests.get(self.api_url, params={"sql": sql}, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        records = data.get("result", {}).get("records", [])
        return pd.DataFrame(records)


def ingest_cityscore_data(
    execution_date: str,
    watermark_start: datetime | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for ingesting CityScore data."""
    ingester = CityScoreIngester(config)

    until = datetime.strptime(execution_date, "%Y-%m-%d")
    since = watermark_start

    df = ingester.fetch_data(since=since, until=until)
    ingester._data = df

    result = ingester.run(execution_date, watermark_start)
    return result.to_dict()
