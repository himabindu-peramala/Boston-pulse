"""
Boston Pulse - Fire Data Ingester
Uses CKAN datastore_search (NOT SQL)
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from src.datasets.base import BaseIngester
from src.shared.config import Settings

logger = logging.getLogger(__name__)

class FireIngester(BaseIngester):
    BASE_URL = "https://data.boston.gov/api/3/action/datastore_search"
    RESOURCE_ID = "91a38b1f-8439-46df-ba47-a30c48845e06"

    def __init__(self, config: Settings | None = None):
        super().__init__(config)
        self.batch_size = 5000

    def get_dataset_name(self) -> str:
        return "fire"

    def get_primary_key(self) -> str:
        return "incident_number"

    def get_watermark_field(self) -> str:
        return "alarm_date"

    def fetch_data(self, since: datetime | None = None) -> pd.DataFrame:
        logger.info("Starting ingestion for fire")
        if since is None:
            raise ValueError("FireIngester requires watermark value")

        # ✅ Use date-only strings for comparison since alarm_date is "YYYY-MM-DD"
        since_str = since.strftime("%Y-%m-%d")
        until_str = datetime.utcnow().strftime("%Y-%m-%d")

        logger.info(f"Fetching fire data from {since_str} to {until_str}")

        all_records = []
        offset = 0
        while True:
            records = self._fetch_page(offset)
            if not records:
                break

            for r in records:
                alarm_date = r.get("alarm_date", "")
                if alarm_date and since_str <= alarm_date[:10] <= until_str:
                    all_records.append(r)

            if len(records) < self.batch_size:
                break
            offset += self.batch_size

        if not all_records:
            logger.info("No new fire records found")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} fire records")
        return df

    def _fetch_page(self, offset: int):
        response = requests.get(
            self.BASE_URL,
            params={
                "resource_id": self.RESOURCE_ID,
                "limit": self.batch_size,
                "offset": offset,
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        if not data.get("success", False):
            raise RuntimeError(f"CKAN error: {data}")
        return data["result"]["records"]
