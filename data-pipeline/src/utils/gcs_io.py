"""
Boston Pulse - GCS I/O Utilities

Utilities for reading and writing data to Google Cloud Storage.
Provides consistent interface for:
- Parquet file I/O
- JSON file I/O
- Path generation
- Latest file pointers

Usage:
    from dags.utils.gcs_io import GCSDataIO

    gcs = GCSDataIO()
    df = gcs.read_parquet("crime", "raw", "2024-01-15")
    gcs.write_parquet(df, "crime", "processed", "2024-01-15")
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pandas as pd
from google.cloud import storage

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


class GCSDataIO:
    """
    GCS Data I/O handler for the Boston Pulse pipeline.

    Handles reading and writing data files to GCS with consistent
    path conventions and format handling.
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize GCS I/O handler.

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

    def get_path(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str = "data.parquet",
    ) -> str:
        """
        Generate a GCS path for data.

        Args:
            dataset: Dataset name (crime, service_311, etc.)
            layer: Data layer (raw, processed, features)
            execution_date: Execution date in YYYY-MM-DD format
            filename: Filename (default: data.parquet)

        Returns:
            GCS path like "raw/crime/dt=2024-01-15/data.parquet"
        """
        layer_path = getattr(self.config.storage.paths, layer, layer)
        return f"{layer_path}/{dataset}/dt={execution_date}/{filename}"

    def get_full_path(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str = "data.parquet",
    ) -> str:
        """
        Generate a full GCS URI for data.

        Returns:
            Full GCS URI like "gs://bucket/raw/crime/dt=2024-01-15/data.parquet"
        """
        path = self.get_path(dataset, layer, execution_date, filename)
        return f"gs://{self.bucket_name}/{path}"

    def read_parquet(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str = "data.parquet",
    ) -> pd.DataFrame:
        """
        Read a Parquet file from GCS.

        Args:
            dataset: Dataset name
            layer: Data layer
            execution_date: Execution date
            filename: Filename

        Returns:
            DataFrame with the data

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = self.get_path(dataset, layer, execution_date, filename)

        logger.info(
            "Reading parquet from GCS",
            extra={"dataset": dataset, "layer": layer, "path": path},
        )

        try:
            blob = self.bucket.blob(path)
            content = blob.download_as_bytes()
            df = pd.read_parquet(BytesIO(content))

            logger.info(
                f"Read {len(df)} rows from {path}",
                extra={"dataset": dataset, "rows": len(df)},
            )

            return df

        except Exception as e:
            raise FileNotFoundError(f"Failed to read {path}: {e}") from e

    def write_parquet(
        self,
        df: pd.DataFrame,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str = "data.parquet",
    ) -> str:
        """
        Write a DataFrame to GCS as Parquet.

        Args:
            df: DataFrame to write
            dataset: Dataset name
            layer: Data layer
            execution_date: Execution date
            filename: Filename

        Returns:
            Full GCS path where data was written
        """
        path = self.get_path(dataset, layer, execution_date, filename)

        logger.info(
            "Writing parquet to GCS",
            extra={"dataset": dataset, "layer": layer, "path": path, "rows": len(df)},
        )

        # Convert to parquet bytes
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        # Upload to GCS
        blob = self.bucket.blob(path)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        # Update latest pointer
        self._update_latest_pointer(dataset, layer, execution_date, filename)

        full_path = f"gs://{self.bucket_name}/{path}"
        logger.info(f"Wrote {len(df)} rows to {full_path}")

        return full_path

    def read_latest_parquet(
        self,
        dataset: str,
        layer: str,
    ) -> pd.DataFrame:
        """
        Read the latest Parquet file for a dataset/layer.

        Args:
            dataset: Dataset name
            layer: Data layer

        Returns:
            DataFrame with the latest data
        """
        # Read the latest pointer
        pointer_path = f"{layer}/{dataset}/latest.json"

        try:
            blob = self.bucket.blob(pointer_path)
            content = json.loads(blob.download_as_string())
            execution_date = content["execution_date"]
            filename = content.get("filename", "data.parquet")

            return self.read_parquet(dataset, layer, execution_date, filename)

        except Exception as e:
            raise FileNotFoundError(f"No latest data found for {dataset}/{layer}: {e}") from e

    def _update_latest_pointer(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str,
    ) -> None:
        """Update the latest pointer file."""
        layer_path = getattr(self.config.storage.paths, layer, layer)
        pointer_path = f"{layer_path}/{dataset}/latest.json"

        pointer_data = {
            "execution_date": execution_date,
            "filename": filename,
            "updated_at": datetime.now(UTC).isoformat(),
            "path": self.get_path(dataset, layer, execution_date, filename),
        }

        blob = self.bucket.blob(pointer_path)
        blob.upload_from_string(
            json.dumps(pointer_data, indent=2),
            content_type="application/json",
        )

    def read_json(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str,
    ) -> dict[str, Any]:
        """Read a JSON file from GCS."""
        path = self.get_path(dataset, layer, execution_date, filename)

        try:
            blob = self.bucket.blob(path)
            content = blob.download_as_string()
            return json.loads(content)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read {path}: {e}") from e

    def write_json(
        self,
        data: dict[str, Any],
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str,
    ) -> str:
        """Write a JSON file to GCS."""
        path = self.get_path(dataset, layer, execution_date, filename)

        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )

        return f"gs://{self.bucket_name}/{path}"

    def file_exists(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        filename: str = "data.parquet",
    ) -> bool:
        """Check if a file exists in GCS."""
        path = self.get_path(dataset, layer, execution_date, filename)
        blob = self.bucket.blob(path)
        return blob.exists()

    def list_execution_dates(
        self,
        dataset: str,
        layer: str,
    ) -> list[str]:
        """
        List all available execution dates for a dataset/layer.

        Returns:
            List of execution dates sorted newest first
        """
        layer_path = getattr(self.config.storage.paths, layer, layer)
        prefix = f"{layer_path}/{dataset}/dt="

        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix, delimiter="/")

        dates = set()
        for blob in blobs:
            # Extract date from path like "raw/crime/dt=2024-01-15/data.parquet"
            parts = blob.name.split("/")
            for part in parts:
                if part.startswith("dt="):
                    dates.add(part.replace("dt=", ""))
                    break

        # Also check prefixes (for folder-like structures)
        for prefix in blobs.prefixes:
            if "dt=" in prefix:
                date_part = prefix.split("dt=")[1].rstrip("/")
                dates.add(date_part)

        return sorted(dates, reverse=True)


# =============================================================================
# Convenience Functions
# =============================================================================


def read_data(
    dataset: str,
    layer: str,
    execution_date: str,
    config: Settings | None = None,
) -> pd.DataFrame:
    """Convenience function to read data."""
    gcs = GCSDataIO(config)
    return gcs.read_parquet(dataset, layer, execution_date)


def write_data(
    df: pd.DataFrame,
    dataset: str,
    layer: str,
    execution_date: str,
    config: Settings | None = None,
) -> str:
    """Convenience function to write data."""
    gcs = GCSDataIO(config)
    return gcs.write_parquet(df, dataset, layer, execution_date)


def get_latest_data(
    dataset: str,
    layer: str,
    config: Settings | None = None,
) -> pd.DataFrame:
    """Convenience function to get latest data."""
    gcs = GCSDataIO(config)
    return gcs.read_latest_parquet(dataset, layer)
