"""
Boston Pulse - Base Ingester

Abstract base class for all dataset ingesters. Provides a consistent interface
for fetching data from various sources with:
- Watermark-based incremental ingestion
- Error handling and retries
- Structured result reporting

Usage:
    class CrimeIngester(BaseIngester):
        def fetch_data(self, since: Optional[datetime]) -> pd.DataFrame:
            ...
        def get_watermark_field(self) -> str:
            return "occurred_on_date"
        def get_primary_key(self) -> str:
            return "incident_number"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""

    dataset: str
    execution_date: str
    rows_fetched: int
    rows_new: int
    rows_updated: int
    watermark_start: datetime | None
    watermark_end: datetime | None
    output_path: str | None = None
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for XCom/logging."""
        return {
            "dataset": self.dataset,
            "execution_date": self.execution_date,
            "rows_fetched": self.rows_fetched,
            "rows_new": self.rows_new,
            "rows_updated": self.rows_updated,
            "watermark_start": self.watermark_start.isoformat() if self.watermark_start else None,
            "watermark_end": self.watermark_end.isoformat() if self.watermark_end else None,
            "output_path": self.output_path,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,}


class BaseIngester(ABC):
    """
    Abstract base class for dataset ingestion.

    Subclasses must implement:
    - fetch_data(): Fetch data from the source
    - get_watermark_field(): Return the field used for incremental ingestion
    - get_primary_key(): Return the primary key field
    - get_dataset_name(): Return the dataset name
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize the ingester.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()

    @abstractmethod
    def fetch_data(self, since: datetime | None = None) -> pd.DataFrame:
        """
        Fetch data from the source.

        Args:
            since: If provided, only fetch data modified since this datetime.
                   If None, fetch all available data.

        Returns:
            DataFrame containing the fetched data
        """
        pass

    @abstractmethod
    def get_watermark_field(self) -> str:
        """
        Get the field name used for watermark-based incremental ingestion.

        Returns:
            Column name that contains the timestamp for incremental filtering
        """
        pass

    @abstractmethod
    def get_primary_key(self) -> str:
        """
        Get the primary key field name.

        Returns:
            Column name that uniquely identifies each record
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Get the dataset name.

        Returns:
            Dataset name (e.g., "crime", "service_311")
        """
        pass

    def get_api_endpoint(self) -> str | None:
        """
        Get the API endpoint for this dataset (optional).

        Returns:
            API endpoint URL or None if not applicable
        """
        return None

    def run(
        self,
        execution_date: str,
        watermark_start: datetime | None = None,
    ) -> IngestionResult:
        """
        Run the ingestion process.

        Args:
            execution_date: Execution date in YYYY-MM-DD format
            watermark_start: Starting watermark for incremental ingestion

        Returns:
            IngestionResult with details about the ingestion
        """
        import time

        start_time = time.time()
        dataset_name = self.get_dataset_name()

        logger.info(
            f"Starting ingestion for {dataset_name}",
            extra={
                "dataset": dataset_name,
                "execution_date": execution_date,
                "watermark_start": watermark_start.isoformat() if watermark_start else None,},
        )

        try:
            # Determine watermark
            if watermark_start is None and self.config.datasets.watermark.enabled:
                lookback_days = self.config.datasets.watermark.lookback_days
                watermark_start = datetime.now(UTC) - timedelta(days=lookback_days)

            # Fetch data
            df = self.fetch_data(since=watermark_start)

            # Calculate watermark end from data
            watermark_field = self.get_watermark_field()
            watermark_end = None
            # Calculate watermark end from data
            if watermark_field in df.columns and len(df) > 0:
                if pd.api.types.is_datetime64_any_dtype(df[watermark_field]):
                    watermark_end = df[watermark_field].max()
                else:
                    with suppress(Exception):
                        watermark_end = pd.to_datetime(df[watermark_field]).max()

            # Calculate row statistics
            primary_key = self.get_primary_key()
            rows_fetched = len(df)
            rows_new = rows_fetched
            rows_updated = 0

            duration = time.time() - start_time

            result = IngestionResult(
                dataset=dataset_name,
                execution_date=execution_date,
                rows_fetched=rows_fetched,
                rows_new=rows_new,
                rows_updated=rows_updated,
                watermark_start=watermark_start,
                watermark_end=watermark_end,
                duration_seconds=duration,
                success=True,
                metadata={
                    "primary_key": primary_key,
                    "watermark_field": watermark_field,
                    "columns": list(df.columns),},
            )

            logger.info(
                f"Ingestion complete for {dataset_name}: {rows_fetched} rows",
                extra=result.to_dict(),
            )

            # Store the dataframe for downstream access
            self._data = df

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Ingestion failed for {dataset_name}: {e}",
                extra={"dataset": dataset_name, "error": str(e)},
                exc_info=True,
            )

            return IngestionResult(
                dataset=dataset_name,
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

    def get_data(self) -> pd.DataFrame | None:
        """Get the most recently fetched data."""
        return getattr(self, "_data", None)

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Perform basic schema validation on fetched data.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check primary key exists
        pk = self.get_primary_key()
        if pk not in df.columns:
            errors.append(f"Primary key column '{pk}' not found")

        # Check watermark field exists
        wf = self.get_watermark_field()
        if wf not in df.columns:
            errors.append(f"Watermark field '{wf}' not found")

        # Check for empty dataframe
        if len(df) == 0:
            errors.append("DataFrame is empty")

        return len(errors) == 0, errors
