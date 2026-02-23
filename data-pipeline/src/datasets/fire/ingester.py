"""
Boston Pulse - Fire Incidents Ingester

Fetches fire incident data from configured source.

Configuration:
    All settings loaded from configs/datasets/fire.yaml
"""

from __future__ import annotations

import logging
from datetime import datetime


import pandas as pd

from src.datasets.base import BaseIngester
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DATASET_CONFIG = get_dataset_config("fire")

INGESTION_CONFIG = DATASET_CONFIG.get("ingestion", {})
WATERMARK_FIELD = INGESTION_CONFIG.get("watermark_field", "incident_date")
PRIMARY_KEY = INGESTION_CONFIG.get("primary_key", "incident_id")


class FireIngester(BaseIngester):
    """
    Ingester for Boston Fire Incident data.

    Mirrors CrimeIngester structure.
    """

    def __init__(self, config: Settings | None = None):
        super().__init__(config)

    def get_dataset_name(self) -> str:
        return "fire"

    def get_watermark_field(self) -> str:
        return WATERMARK_FIELD

    def get_primary_key(self) -> str:
        return PRIMARY_KEY

    def fetch_data(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fire incident data.

        Replace this logic with real ingestion source
        (API, database, file, etc.)
        """

        logger.info("Fetching fire incident data")

        # Example placeholder — replace with real source
        df = pd.read_csv("data/raw/fire_incidents.csv")

        logger.info(f"Fetched {len(df)} fire records")

        return df
