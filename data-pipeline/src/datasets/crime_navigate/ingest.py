"""
Navigate Crime Ingester — Analyze Boston API, watermark-based.

First run: pull from config.watermark.first_run_start (e.g. 2015-01-01).
Subsequent runs: pull from watermark date forward. Drops _id and _full_text.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from src.datasets.base import BaseIngester, IngestionResult
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

DATASET_CONFIG = get_dataset_config("crime_navigate")
API_CONFIG = DATASET_CONFIG.get("api", {})
WATERMARK_CONFIG = DATASET_CONFIG.get("watermark", {})

RESOURCE_ID = API_CONFIG.get("resource_id", "b973d8cb-eeb2-4e7e-99da-c92938efc9c0")
BASE_URL = API_CONFIG.get("base_url", "https://data.boston.gov/api/3/action")
ENDPOINT = API_CONFIG.get("endpoint", "datastore_search_sql")
WATERMARK_FIELD = API_CONFIG.get("watermark_field", "OCCURRED_ON_DATE")
PRIMARY_KEY = API_CONFIG.get("primary_key", "INCIDENT_NUMBER")
BATCH_SIZE = API_CONFIG.get("batch_size", 10000)
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)
RATE_LIMIT_SLEEP = API_CONFIG.get("rate_limit_sleep_seconds", 0.5)
FIRST_RUN_START = WATERMARK_CONFIG.get("first_run_start", "2023-01-01")


class CrimeNavigateIngester(BaseIngester):
    """
    Ingester for Navigate crime data from Analyze Boston API.
    Uses crime_navigate.yaml; watermark from navigate/watermarks/crime_navigate.json.
    """

    def __init__(self, config: Settings | None = None):
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

    def get_dataset_name(self) -> str:
        return "crime_navigate"

    def get_watermark_field(self) -> str:
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        return PRIMARY_KEY

    def _fetch_month(
        self,
        year: int,
        month: int,
        overall_since: datetime,
        overall_until: datetime,
    ) -> list[dict[str, Any]]:
        """
        Fetch all records for a calendar month, clipped to overall_since/overall_until.

        This uses a date filter so OFFSET resets to 0 for every month, avoiding
        CKAN's silent cap on large unfiltered OFFSET scans.
        """
        # Month boundaries in UTC
        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)

        # Clip to overall [since, until] window (date-level)
        since_clip = pd.Timestamp(overall_since.date(), tz="UTC")
        until_clip = pd.Timestamp(overall_until.date(), tz="UTC") + pd.Timedelta(days=1)

        start = max(month_start, since_clip)
        end = min(month_end, until_clip)

        # If this month does not intersect the desired range, skip
        if start >= end:
            return []

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        month_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            sql = (
                f'SELECT * FROM "{RESOURCE_ID}" '
                f"WHERE \"{WATERMARK_FIELD}\" >= '{start_str} 00:00:00' "
                f"AND \"{WATERMARK_FIELD}\" < '{end_str} 00:00:00' "
                f'ORDER BY "{WATERMARK_FIELD}" ASC '
                f"LIMIT {BATCH_SIZE} OFFSET {offset}"
            )

            try:
                response = requests.get(
                    self.api_url,
                    params={"sql": sql},
                    timeout=TIMEOUT,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                data = response.json()
                if not data.get("success"):
                    logger.warning("API error for %s-%02d: %s", year, month, data.get("error"))
                    break

                records = data.get("result", {}).get("records", [])
                if not records:
                    break

                month_records.extend(records)

                # Last page for this month
                if len(records) < BATCH_SIZE:
                    break

                offset += len(records)
                time.sleep(RATE_LIMIT_SLEEP)
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning(
                    "Request failed for %s-%02d at offset %d: %s",
                    year,
                    month,
                    offset,
                    exc,
                )
                time.sleep(2)
                break

        logger.info(f"Fetched {len(month_records)} records for {year}-{month}")

        return month_records

    def fetch_data(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> pd.DataFrame:
        """
        Fetch crime data using year-by-year, month-by-month paging.

        - If since is None (no watermark): start from FIRST_RUN_START (2023-01-01).
        - If since is set (watermark exists): start from that date.
        - Always stop at 'until' (execution date, inclusive).
        """
        if until is None:
            until = datetime.now(UTC)

        if since is None:
            # First run — full backfill from configured start year (2023+)
            since = datetime.strptime(FIRST_RUN_START, "%Y-%m-%d").replace(tzinfo=UTC)

        # Ensure ordering
        if since > until:
            logger.info(
                "crime_navigate fetch: since %s > until %s, returning empty frame",
                since,
                until,
            )
            return pd.DataFrame()

        logger.info(
            "Fetching crime_navigate from %s to %s (month-by-month)",
            since.date(),
            until.date(),
        )

        all_records: list[dict[str, Any]] = []
        start_year = since.year
        end_year = until.year

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                records = self._fetch_month(year, month, since, until)
                if records:
                    all_records.extend(records)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.drop(columns=["_id", "_full_text"], errors="ignore")
        logger.info("Fetched %d crime_navigate records", len(df))
        return df

    def run(
    self,
    execution_date: str,
    watermark_start: datetime | None = None,
    ) -> IngestionResult:
        """
        Override BaseIngester.run to use dataset-specific first_run_start
        instead of the global lookback_days fallback.

        - watermark_start=None (first run): fetch from FIRST_RUN_START
        - watermark_start set (subsequent runs): fetch from that date forward
        """
        import time
        start_time = time.time()

        logger.info(
            "Starting ingestion for crime_navigate",
            extra={
                "execution_date": execution_date,
                "watermark_start": watermark_start.isoformat() if watermark_start else None,
            },
        )

        try:
            until = datetime.strptime(execution_date, "%Y-%m-%d").replace(tzinfo=UTC)

            # Pass watermark_start directly — fetch_data handles None correctly
            # (uses FIRST_RUN_START). Do NOT let BaseIngester substitute now-7d.
            df = self.fetch_data(since=watermark_start, until=until)

            watermark_end = None
            if WATERMARK_FIELD in df.columns and len(df) > 0:
                try:
                    watermark_end = pd.to_datetime(df[WATERMARK_FIELD]).max()
                except Exception:
                    pass

            duration = time.time() - start_time
            self._data = df

            result = IngestionResult(
                dataset=self.get_dataset_name(),
                execution_date=execution_date,
                rows_fetched=len(df),
                rows_new=len(df),
                rows_updated=0,
                watermark_start=watermark_start,
                watermark_end=watermark_end,
                duration_seconds=duration,
                success=True,
                metadata={
                    "primary_key": PRIMARY_KEY,
                    "watermark_field": WATERMARK_FIELD,
                    "columns": list(df.columns),
                },
            )

            logger.info(
                "Ingestion complete for crime_navigate: %d rows in %.1fs",
                len(df),
                duration,
            )
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Ingestion failed for crime_navigate: %s", e, exc_info=True)
            return IngestionResult(
                dataset=self.get_dataset_name(),
                execution_date=execution_date,
                rows_fetched=0,
                rows_new=0,
                rows_updated=0,
                watermark_start=watermark_start,
                watermark_end=None,
                duration_seconds=duration,
                success=False,
                error_message=str(e),
            )



def ingest_crime_navigate(
    execution_date: str,
    watermark_start: datetime | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Convenience: run ingester and return result dict for XCom."""
    ingester = CrimeNavigateIngester(config)
    until = datetime.strptime(execution_date, "%Y-%m-%d").replace(tzinfo=UTC)
    df = ingester.fetch_data(since=watermark_start, until=until)
    ingester._data = df
    result = ingester.run(execution_date=execution_date, watermark_start=watermark_start)
    return result.to_dict()
