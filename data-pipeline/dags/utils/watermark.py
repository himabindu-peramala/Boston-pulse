"""
Boston Pulse - Watermark Management

Utilities for managing watermarks for incremental data ingestion.
Watermarks track the last successfully processed timestamp for each dataset,
enabling efficient incremental updates.

Usage:
    from dags.utils.watermark import WatermarkManager

    wm = WatermarkManager()
    last_watermark = wm.get_watermark("crime")
    # ... process data ...
    wm.set_watermark("crime", new_watermark, execution_date)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from google.cloud import storage

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class WatermarkManager:
    """
    Manages watermarks for incremental data ingestion.

    Watermarks are stored in GCS as JSON files, one per dataset.
    Each watermark contains:
    - value: The datetime of the last processed record
    - updated_at: When the watermark was last updated
    - execution_date: The Airflow execution date when updated
    - history: Recent watermark history for debugging
    """

    WATERMARKS_PATH = "watermarks"
    MAX_HISTORY = 10

    def __init__(self, config: Settings | None = None):
        """
        Initialize watermark manager.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.bucket_name = self.config.storage.buckets.main

        # Initialize GCS client
        if self.config.storage.emulator.enabled:
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": self.config.storage.emulator.host},
            )
        else:
            self.client = storage.Client(project=self.config.gcp_project_id)

        self.bucket = self.client.bucket(self.bucket_name)

    def _get_watermark_path(self, dataset: str) -> str:
        """Get the GCS path for a dataset's watermark file."""
        return f"{self.WATERMARKS_PATH}/{dataset}.json"

    def get_watermark(self, dataset: str) -> datetime | None:
        """
        Get the current watermark for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Watermark datetime, or None if no watermark exists
        """
        path = self._get_watermark_path(dataset)

        try:
            blob = self.bucket.blob(path)
            if not blob.exists():
                logger.info(f"No watermark found for {dataset}")
                return None

            content = json.loads(blob.download_as_string())
            watermark_str = content.get("value")

            if watermark_str:
                watermark = datetime.fromisoformat(watermark_str)
                logger.info(
                    f"Retrieved watermark for {dataset}: {watermark}",
                    extra={"dataset": dataset, "watermark": watermark_str},
                )
                return watermark

            return None

        except Exception as e:
            logger.warning(f"Failed to get watermark for {dataset}: {e}")
            return None

    def set_watermark(
        self,
        dataset: str,
        watermark: datetime,
        execution_date: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Set the watermark for a dataset.

        Args:
            dataset: Dataset name
            watermark: New watermark datetime
            execution_date: Airflow execution date
            metadata: Optional additional metadata
        """
        path = self._get_watermark_path(dataset)

        # Load existing data for history
        existing = self._load_watermark_data(dataset)
        history = existing.get("history", []) if existing else []

        # Add current value to history
        if existing and existing.get("value"):
            history.insert(
                0,
                {
                    "value": existing["value"],
                    "updated_at": existing.get("updated_at"),
                    "execution_date": existing.get("execution_date"),
                },
            )
            # Keep only recent history
            history = history[: self.MAX_HISTORY]

        # Create new watermark data
        watermark_data = {
            "value": watermark.isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "execution_date": execution_date,
            "dataset": dataset,
            "history": history,
        }

        if metadata:
            watermark_data["metadata"] = metadata

        # Save to GCS
        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(watermark_data, indent=2),
            content_type="application/json",
        )

        logger.info(
            f"Set watermark for {dataset}: {watermark}",
            extra={
                "dataset": dataset,
                "watermark": watermark.isoformat(),
                "execution_date": execution_date,
            },
        )

    def _load_watermark_data(self, dataset: str) -> dict[str, Any] | None:
        """Load the full watermark data for a dataset."""
        path = self._get_watermark_path(dataset)

        try:
            blob = self.bucket.blob(path)
            if not blob.exists():
                return None
            return json.loads(blob.download_as_string())
        except Exception:
            return None

    def get_watermark_info(self, dataset: str) -> dict[str, Any] | None:
        """
        Get full watermark information for a dataset.

        Returns:
            Dictionary with watermark value, history, and metadata
        """
        return self._load_watermark_data(dataset)

    def delete_watermark(self, dataset: str) -> bool:
        """
        Delete the watermark for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            True if deleted, False if not found
        """
        path = self._get_watermark_path(dataset)

        try:
            blob = self.bucket.blob(path)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted watermark for {dataset}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete watermark for {dataset}: {e}")
            return False

    def list_watermarks(self) -> dict[str, datetime | None]:
        """
        List all watermarks.

        Returns:
            Dictionary mapping dataset names to watermark datetimes
        """
        prefix = f"{self.WATERMARKS_PATH}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        watermarks = {}
        for blob in blobs:
            if blob.name.endswith(".json"):
                dataset = blob.name.replace(prefix, "").replace(".json", "")
                watermarks[dataset] = self.get_watermark(dataset)

        return watermarks

    def get_effective_watermark(
        self,
        dataset: str,
        lookback_days: int | None = None,
    ) -> datetime:
        """
        Get the effective watermark for ingestion.

        If no watermark exists, uses the configured lookback period.

        Args:
            dataset: Dataset name
            lookback_days: Override lookback days (uses config default if not provided)

        Returns:
            Effective watermark datetime to use for ingestion
        """
        from datetime import timedelta

        watermark = self.get_watermark(dataset)

        if watermark is not None:
            return watermark

        # Use lookback period for initial ingestion
        if lookback_days is None:
            lookback_days = self.config.datasets.watermark.lookback_days

        effective = datetime.now(UTC) - timedelta(days=lookback_days)
        logger.info(f"No watermark for {dataset}, using {lookback_days} day lookback: {effective}")

        return effective


# =============================================================================
# Convenience Functions
# =============================================================================


def get_watermark(dataset: str, config: Settings | None = None) -> datetime | None:
    """Convenience function to get a watermark."""
    wm = WatermarkManager(config)
    return wm.get_watermark(dataset)


def set_watermark(
    dataset: str,
    watermark: datetime,
    execution_date: str,
    config: Settings | None = None,
) -> None:
    """Convenience function to set a watermark."""
    wm = WatermarkManager(config)
    wm.set_watermark(dataset, watermark, execution_date)


def get_effective_watermark(
    dataset: str,
    lookback_days: int | None = None,
    config: Settings | None = None,
) -> datetime:
    """Convenience function to get effective watermark."""
    wm = WatermarkManager(config)
    return wm.get_effective_watermark(dataset, lookback_days)
