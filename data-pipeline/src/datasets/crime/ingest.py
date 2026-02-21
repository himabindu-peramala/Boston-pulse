"""
Boston Pulse - Crime Data Ingester

Fetches crime incident data from the Boston Police Department via Analyze Boston API.

Data Source:
    Boston Police Department Crime Incident Reports
    https://data.boston.gov/dataset/crime-incident-reports

Configuration:
    All settings loaded from configs/datasets/crime.yaml

Usage:
    from src.datasets.crime.ingest import CrimeIngester

    ingester = CrimeIngester()
    result = ingester.run(execution_date="2024-01-15")
    df = ingester.get_data()
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
# API Configuration (loaded from crime.yaml)
# =============================================================================
DATASET_CONFIG = get_dataset_config("crime")

# API settings
API_CONFIG = DATASET_CONFIG.get("api", {})
RESOURCE_ID = API_CONFIG.get("resource_id", "b973d8cb-eeb2-4e7e-99da-c92938efc9c0")
BASE_URL = API_CONFIG.get("base_url", "https://data.boston.gov/api/3/action")
ENDPOINT = API_CONFIG.get("endpoint", "datastore_search_sql")
BATCH_SIZE = API_CONFIG.get("batch_size", 1000)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)

# Ingestion settings
INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "OCCURRED_ON_DATE")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "INCIDENT_NUMBER")
LOOKBACK_DAYS = INGESTION_CONFIG.get("lookback_days", 7)


class CrimeIngester(BaseIngester):
    """
    Ingester for Boston crime incident data.

    Fetches data from the Analyze Boston API using the CKAN datastore_search_sql
    endpoint. Supports incremental ingestion via watermark filtering.

    All configuration is loaded from configs/datasets/crime.yaml.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize crime ingester with config from crime.yaml."""
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "crime"

    def get_watermark_field(self) -> str:
        """Return the field used for incremental ingestion (from config)."""
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        """Return the primary key field (from config)."""
        return PRIMARY_KEY

    def get_api_endpoint(self) -> str:
        """Get the API endpoint for crime data."""
        return self.api_url

    def _fetch_page(self, since: str, until: str, offset: int) -> list[dict]:
        """
        Fetch one page of crime records between two dates using SQL query.

        Args:
            since: Start date in "YYYY-MM-DD" format
            until: End date in "YYYY-MM-DD" format
            offset: Pagination offset

        Returns:
            List of record dictionaries
        """
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"WHERE \"{WATERMARK_FIELD}\" >= '{since} 00:00' "
            f"AND \"{WATERMARK_FIELD}\" <= '{until} 23:59' "
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
        """
        Fetch crime data from Analyze Boston API.

        Args:
            since: Start datetime for filtering (uses lookback if None)
            until: End datetime for filtering (uses today if None)

        Returns:
            DataFrame with crime incident data
        """
        # Determine date range
        if until is None:
            until = datetime.now(UTC)

        if since is None:
            since = until - timedelta(days=LOOKBACK_DAYS)

        since_str = since.strftime("%Y-%m-%d")
        until_str = until.strftime("%Y-%m-%d")

        logger.info(
            f"Fetching crime data from {since_str} to {until_str}",
            extra={"since": since_str, "until": until_str},
        )

        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            logger.info(f"Requesting offset {offset}...")

            try:
                records = self._fetch_page(since_str, until_str, offset)
            except Exception as e:
                logger.error(f"API request failed: {e}")
                raise

            if not records:
                logger.info("No more records.")
                break

            all_records.extend(records)
            logger.info(f"Got {len(records)} records. Total so far: {len(all_records)}")

            if len(records) < BATCH_SIZE:
                logger.info("Last page reached.")
                break

            offset += len(records)
            time.sleep(0.5)

        # Convert to DataFrame
        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Drop internal CKAN columns
        df = df.drop(columns=["_id", "_full_text"], errors="ignore")

        logger.info(
            f"Fetched {len(df)} total crime records",
            extra={"rows": len(df), "columns": list(df.columns) if len(df) > 0 else []},
        )

        return df

    def fetch_sample_data(self, n: int = 1000) -> pd.DataFrame:
        """
        Fetch a sample of crime data for testing/development.

        Args:
            n: Number of records to fetch

        Returns:
            DataFrame with sample crime data
        """
        sql = f'SELECT * FROM "{RESOURCE_ID}" ' f'ORDER BY "{WATERMARK_FIELD}" DESC ' f"LIMIT {n}"

        response = requests.get(
            self.api_url,
            params={"sql": sql},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError("API request failed")

        records = data.get("result", {}).get("records", [])
        df = pd.DataFrame(records)
        return df.drop(columns=["_id", "_full_text"], errors="ignore")

    def get_schema_info(self) -> list[dict[str, str]]:
        """
        Get schema information from the API.

        Returns:
            List of field definitions with id, type, and info
        """
        sql = f'SELECT * FROM "{RESOURCE_ID}" LIMIT 0'

        response = requests.get(
            self.api_url,
            params={"sql": sql},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        return data.get("result", {}).get("fields", [])


# =============================================================================
# Convenience Functions
# =============================================================================


def ingest_crime_data(
    execution_date: str,
    watermark_start: datetime | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Convenience function for ingesting crime data.

    Returns result dictionary suitable for Airflow XCom.
    """
    ingester = CrimeIngester(config)

    # Parse execution date as the "until" date
    until = datetime.strptime(execution_date, "%Y-%m-%d")

    # Use watermark as "since" if provided, otherwise use lookback
    since = watermark_start

    # Override fetch_data call with proper date range
    df = ingester.fetch_data(since=since, until=until)
    ingester._data = df  # Store for get_data()

    result = ingester.run(execution_date, watermark_start)
    return result.to_dict()


def get_crime_sample(n: int = 1000, config: Settings | None = None) -> pd.DataFrame:
    """Convenience function to get a sample of crime data."""
    ingester = CrimeIngester(config)
    return ingester.fetch_sample_data(n)
