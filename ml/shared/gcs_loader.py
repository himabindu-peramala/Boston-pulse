"""
Boston Pulse ML - GCS Loader.

Generic GCS parquet reader for the ML pipeline.
Reads partitioned parquet files written by data-pipeline/.
Does not import from data-pipeline/ — path patterns are passed as config.
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from typing import Any

import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSLoader:
    """
    Generic GCS parquet reader for the ML pipeline.

    Reads partitioned parquet files written by data-pipeline/.
    Does not import from data-pipeline/ — path patterns are passed as config.
    """

    def __init__(self, bucket: str, project: str | None = None):
        """
        Initialize GCS loader.

        Args:
            bucket: GCS bucket name
            project: GCP project ID (uses default if not provided)
        """
        self.bucket_name = bucket
        self.project = project or os.getenv("GCP_PROJECT_ID")

        # Check for emulator
        emulator_host = os.getenv("STORAGE_EMULATOR_HOST")
        if emulator_host:
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": emulator_host},
            )
        else:
            self.client = storage.Client(project=self.project)

        self.bucket = self.client.bucket(bucket)

    def list_partitions(
        self,
        prefix: str,
        after: str | None = None,
        before: str | None = None,
    ) -> list[str]:
        """
        List all dt=YYYY-MM-DD partition dates under a GCS prefix.

        Args:
            prefix: e.g. "features/crime_navigate"
            after:  only return partitions on or after this date (inclusive)
            before: only return partitions on or before this date (inclusive)

        Returns:
            Sorted list of date strings: ["2023-01-01", "2023-01-04", ...]
        """
        full_prefix = f"{prefix}/dt="
        blobs = self.client.list_blobs(self.bucket_name, prefix=full_prefix)

        dates = set()
        for blob in blobs:
            parts = blob.name.split("/")
            for part in parts:
                if part.startswith("dt="):
                    date = part[3:]
                    dates.add(date)

        dates_list = sorted(dates)

        if after:
            dates_list = [d for d in dates_list if d >= after]
        if before:
            dates_list = [d for d in dates_list if d <= before]

        return dates_list

    def read_partition(
        self,
        prefix: str,
        date: str,
        filename: str = "data.parquet",
    ) -> pd.DataFrame:
        """
        Read a single date partition.

        Args:
            prefix: GCS prefix (e.g. "features/crime_navigate")
            date: Partition date (YYYY-MM-DD)
            filename: File name within partition

        Returns:
            DataFrame with partition data
        """
        path = f"gs://{self.bucket_name}/{prefix}/dt={date}/{filename}"
        logger.debug(f"Reading partition: {path}")
        return pd.read_parquet(path)

    def read_all_partitions(
        self,
        prefix: str,
        filename: str = "data.parquet",
        after: str | None = None,
        before: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Read and concatenate all partitions in date range.

        This is the main entry point for feature_loader.py and target_builder.py.
        Adds a 'date' column from the partition key if not already present.

        Args:
            prefix: GCS prefix (e.g. "features/crime_navigate")
            filename: File name within each partition
            after: Only include partitions on or after this date
            before: Only include partitions on or before this date
            columns: Optional list of columns to select

        Returns:
            Concatenated DataFrame with all partitions
        """
        dates = self.list_partitions(prefix, after=after, before=before)

        if not dates:
            raise FileNotFoundError(
                f"No partitions found at gs://{self.bucket_name}/{prefix}/ "
                f"between {after} and {before}"
            )

        logger.info(f"Loading {len(dates)} partitions from {prefix} ({dates[0]} → {dates[-1]})")

        dfs = []
        for date in dates:
            try:
                df = self.read_partition(prefix, date, filename)
                if columns:
                    available = [c for c in columns if c in df.columns]
                    df = df[available]
                if "date" not in df.columns:
                    df["date"] = date
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping partition {date}: {e}")
                continue

        if not dfs:
            raise RuntimeError(f"All partitions failed to load from {prefix}")

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(result):,} rows from {len(dfs)} partitions")
        return result

    def write_parquet(
        self,
        df: pd.DataFrame,
        prefix: str,
        date: str,
        filename: str = "data.parquet",
    ) -> str:
        """
        Write a DataFrame to a dated GCS partition.

        Args:
            df: DataFrame to write
            prefix: GCS prefix (e.g. "ml/scores/crime_navigate")
            date: Partition date (YYYY-MM-DD)
            filename: Output filename

        Returns:
            Full GCS path where data was written
        """
        path = f"{prefix}/dt={date}/{filename}"

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        blob = self.bucket.blob(path)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        full_path = f"gs://{self.bucket_name}/{path}"
        logger.info(f"Wrote {len(df):,} rows to {full_path}")
        return full_path

    def write_json(
        self,
        data: dict[str, Any],
        prefix: str,
        date: str,
        filename: str,
    ) -> str:
        """
        Write a JSON object to GCS.

        Args:
            data: Dictionary to serialize
            prefix: GCS prefix
            date: Partition date
            filename: Output filename

        Returns:
            Full GCS path
        """
        path = f"{prefix}/dt={date}/{filename}"
        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )
        full_path = f"gs://{self.bucket_name}/{path}"
        logger.info(f"Wrote JSON to {full_path}")
        return full_path

    def read_json(self, prefix: str, date: str, filename: str) -> dict[str, Any]:
        """Read a JSON file from GCS."""
        path = f"{prefix}/dt={date}/{filename}"
        blob = self.bucket.blob(path)
        content = blob.download_as_text()
        return json.loads(content)

    def file_exists(self, prefix: str, date: str, filename: str) -> bool:
        """Check if a file exists in GCS."""
        path = f"{prefix}/dt={date}/{filename}"
        blob = self.bucket.blob(path)
        return blob.exists()

    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a local file to GCS."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return f"gs://{self.bucket_name}/{gcs_path}"

    def download_file(self, gcs_path: str, local_path: str) -> str:
        """Download a GCS file to local path."""
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        return local_path
