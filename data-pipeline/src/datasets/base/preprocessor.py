"""
Boston Pulse - Base Preprocessor

Abstract base class for all dataset preprocessors. Provides a consistent interface
for data cleaning and transformation with:
- Column standardization
- Data type conversion
- Missing value handling
- Geographic and temporal validation

Usage:
    class CrimePreprocessor(BasePreprocessor):
        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            ...
        def get_column_mappings(self) -> dict[str, str]:
            return {"offense_code_group": "offense_category"}
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of a preprocessing operation."""

    dataset: str
    execution_date: str
    rows_input: int
    rows_output: int
    rows_dropped: int
    columns_input: int
    columns_output: int
    output_path: str | None = None
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    transformations_applied: list[str] = field(default_factory=list)
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for XCom/logging."""
        return {
            "dataset": self.dataset,
            "execution_date": self.execution_date,
            "rows_input": self.rows_input,
            "rows_output": self.rows_output,
            "rows_dropped": self.rows_dropped,
            "columns_input": self.columns_input,
            "columns_output": self.columns_output,
            "output_path": self.output_path,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "transformations_applied": self.transformations_applied,
            "drop_reasons": self.drop_reasons,
        }


class BasePreprocessor(ABC):
    """
    Abstract base class for dataset preprocessing.

    Subclasses must implement:
    - transform(): Apply dataset-specific transformations
    - get_dataset_name(): Return the dataset name
    - get_required_columns(): Return list of required output columns
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self._transformations: list[str] = []
        self._drop_reasons: dict[str, int] = {}

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dataset-specific transformations.

        Args:
            df: Raw DataFrame to transform

        Returns:
            Transformed DataFrame
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

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """
        Get list of required columns in the output.

        Returns:
            List of column names that must be present after preprocessing
        """
        pass

    def get_column_mappings(self) -> dict[str, str]:
        """
        Get column name mappings (old -> new).

        Override this method to rename columns during preprocessing.

        Returns:
            Dictionary mapping old column names to new names
        """
        return {}

    def get_dtype_mappings(self) -> dict[str, str]:
        """
        Get data type mappings for columns.

        Override this method to specify target data types.

        Returns:
            Dictionary mapping column names to target dtypes
        """
        return {}

    def run(
        self,
        df: pd.DataFrame,
        execution_date: str,
    ) -> PreprocessingResult:
        """
        Run the preprocessing pipeline.

        Args:
            df: Raw DataFrame to preprocess
            execution_date: Execution date in YYYY-MM-DD format

        Returns:
            PreprocessingResult with details about the preprocessing
        """
        import time

        start_time = time.time()
        dataset_name = self.get_dataset_name()
        rows_input = len(df)
        columns_input = len(df.columns)

        logger.info(
            f"Starting preprocessing for {dataset_name}",
            extra={
                "dataset": dataset_name,
                "execution_date": execution_date,
                "rows_input": rows_input,
            },
        )

        try:
            # Reset tracking
            self._transformations = []
            self._drop_reasons = {}

            # Apply column mappings
            df = self._apply_column_mappings(df)

            # Apply data type conversions
            df = self._apply_dtype_conversions(df)

            # Apply dataset-specific transformations
            df = self.transform(df)

            # Validate required columns
            self._validate_required_columns(df)

            duration = time.time() - start_time
            rows_output = len(df)

            result = PreprocessingResult(
                dataset=dataset_name,
                execution_date=execution_date,
                rows_input=rows_input,
                rows_output=rows_output,
                rows_dropped=rows_input - rows_output,
                columns_input=columns_input,
                columns_output=len(df.columns),
                duration_seconds=duration,
                success=True,
                transformations_applied=self._transformations,
                drop_reasons=self._drop_reasons,
            )

            logger.info(
                f"Preprocessing complete for {dataset_name}: {rows_input} -> {rows_output} rows",
                extra=result.to_dict(),
            )

            # Store processed data
            self._data = df

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Preprocessing failed for {dataset_name}: {e}",
                extra={"dataset": dataset_name, "error": str(e)},
                exc_info=True,
            )

            return PreprocessingResult(
                dataset=dataset_name,
                execution_date=execution_date,
                rows_input=rows_input,
                rows_output=0,
                rows_dropped=rows_input,
                columns_input=columns_input,
                columns_output=0,
                duration_seconds=duration,
                success=False,
                error_message=str(e),
            )

    def get_data(self) -> pd.DataFrame | None:
        """Get the most recently processed data."""
        return getattr(self, "_data", None)

    def _apply_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column name mappings."""
        mappings = self.get_column_mappings()
        if mappings:
            df = df.rename(columns=mappings)
            self._transformations.append(f"renamed_columns: {list(mappings.keys())}")
        return df

    def _apply_dtype_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data type conversions."""
        dtype_mappings = self.get_dtype_mappings()
        for col, dtype in dtype_mappings.items():
            if col in df.columns:
                try:
                    if dtype == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif dtype == "int":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif dtype == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif dtype == "bool":
                        df[col] = df[col].astype(bool)
                    elif dtype == "string":
                        df[col] = df[col].astype(str)
                    else:
                        df[col] = df[col].astype(dtype)
                    self._transformations.append(f"converted_{col}_to_{dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present."""
        required = set(self.get_required_columns())
        present = set(df.columns)
        missing = required - present

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def log_transformation(self, name: str) -> None:
        """Log a transformation that was applied."""
        self._transformations.append(name)

    def log_dropped_rows(self, reason: str, count: int) -> None:
        """Log rows that were dropped."""
        self._drop_reasons[reason] = self._drop_reasons.get(reason, 0) + count

    # ==========================================================================
    # Common Preprocessing Utilities
    # ==========================================================================

    def standardize_coordinates(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "long",
    ) -> pd.DataFrame:
        """
        Standardize geographic coordinates.

        - Converts to numeric
        - Filters invalid coordinates
        - Optionally filters to Boston bounds
        """
        bounds = self.config.validation.geo_bounds

        # Convert to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        # Count invalid before filtering
        invalid_mask = (
            df[lat_col].isna()
            | df[lon_col].isna()
            | (df[lat_col] < bounds.min_lat)
            | (df[lat_col] > bounds.max_lat)
            | (df[lon_col] < bounds.min_lon)
            | (df[lon_col] > bounds.max_lon)
        )

        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            self.log_dropped_rows("invalid_coordinates", int(invalid_count))

        # Filter to valid coordinates
        df = df[~invalid_mask].copy()
        self.log_transformation("standardize_coordinates")

        return df

    def standardize_datetime(
        self,
        df: pd.DataFrame,
        col: str,
        output_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Standardize a datetime column.

        - Converts to datetime
        - Handles various formats
        - Extracts common temporal features
        """
        output_col = output_col or col

        # Convert to datetime
        df[output_col] = pd.to_datetime(df[col], errors="coerce")

        # Extract temporal features
        df[f"{output_col}_year"] = df[output_col].dt.year
        df[f"{output_col}_month"] = df[output_col].dt.month
        df[f"{output_col}_day"] = df[output_col].dt.day
        df[f"{output_col}_hour"] = df[output_col].dt.hour
        df[f"{output_col}_dayofweek"] = df[output_col].dt.dayofweek

        self.log_transformation(f"standardize_datetime_{col}")

        return df

    def drop_duplicates(
        self,
        df: pd.DataFrame,
        subset: list[str] | None = None,
        keep: str = "last",
    ) -> pd.DataFrame:
        """Drop duplicate rows."""
        before_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        dropped = before_count - len(df)

        if dropped > 0:
            self.log_dropped_rows("duplicates", dropped)
            self.log_transformation("drop_duplicates")

        return df

    def fill_missing(
        self,
        df: pd.DataFrame,
        col: str,
        value: Any,
    ) -> pd.DataFrame:
        """Fill missing values in a column."""
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(value)
            self.log_transformation(f"fill_missing_{col}")
        return df
