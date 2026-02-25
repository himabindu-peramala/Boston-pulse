"""
Boston Pulse - Snow Emergency Routes Ingester

Fetches snow emergency route data from the Analyze Boston API.
This is a static dataset containing spatial information about routes.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from src.datasets.base import BaseIngester
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

# Load config
DATASET_CONFIG = get_dataset_config("snow_routes")
API_CONFIG = DATASET_CONFIG.get("api", {})
RESOURCE_ID = API_CONFIG.get("resource_id")
BASE_URL = API_CONFIG.get("base_url")
ENDPOINT = API_CONFIG.get("endpoint")
TIMEOUT = API_CONFIG.get("timeout_seconds", 60)


class SnowRoutesIngester(BaseIngester):
    """Ingester for Boston Snow Emergency Routes."""

    def __init__(self, config: Settings | None = None):
        super().__init__(config)
        self.api_url = f"{BASE_URL}/{ENDPOINT}"

    def get_dataset_name(self) -> str:
        return "snow_routes"

    def get_watermark_field(self) -> str | None:
        return None

    def get_primary_key(self) -> str:
        return "_id"

    def fetch_data(self, **_kwargs) -> pd.DataFrame:
        """Fetch all snow routes from Analyze Boston."""
        logger.info("Fetching all Snow Emergency Routes...")

        sql = f'SELECT * FROM "{RESOURCE_ID}"'
        response = requests.get(self.api_url, params={"sql": sql}, timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            raise ValueError(f"API Error: {data.get('error')}")

        records = data.get("result", {}).get("records", [])
        df = pd.DataFrame(records)

        if not df.empty:
            df = df.drop(columns=["_id", "_full_text"], errors="ignore")

        return df


def ingest_snow_routes(execution_date: str, config: Settings | None = None) -> dict[str, Any]:
    """Convenience function for ingesting snow routes."""
    ingester = SnowRoutesIngester(config)
    df = ingester.fetch_data()
    ingester._data = df
    result = ingester.run(execution_date)
    return result.to_dict()
