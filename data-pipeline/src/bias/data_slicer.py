"""
Boston Pulse - Data Slicer

Slice datasets along multiple dimensions for fairness evaluation:
- Categorical slicing (neighborhood, day_of_week, etc.)
- Quantile-based slicing (continuous features)
- Range-based slicing (custom ranges)
- Cross-dimensional slicing (combinations)

Slicing enables fairness analysis by comparing outcomes across
different subgroups to identify potential bias.

Usage:
    slicer = DataSlicer(config)

    # Slice by neighborhood
    slices = slicer.slice_by_category(df, "neighborhood")

    # Slice by quantiles
    slices = slicer.slice_by_quantiles(df, "income", num_quantiles=4)

    # Get all slices for fairness evaluation
    all_slices = slicer.get_default_slices(df, dataset="crime")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


@dataclass
class DataSlice:
    """
    A slice of data along a specific dimension.

    Represents a subset of data filtered by a specific criterion.
    """

    dimension: str  # e.g., "neighborhood", "hour_of_day"
    value: Any  # e.g., "Dorchester", 14
    data: pd.DataFrame
    size: int
    percentage: float  # Percentage of total dataset

    def __repr__(self) -> str:
        return f"DataSlice({self.dimension}={self.value}, size={self.size})"


@dataclass
class SliceConfig:
    """Configuration for a slicing dimension."""

    dimension: str
    slice_type: str  # "categorical", "quantile", "range"
    config: dict[str, Any]  # Type-specific configuration


class DataSlicer:
    """
    Slice datasets along multiple dimensions for fairness evaluation.

    Supports:
    - Categorical slicing: Group by discrete values
    - Quantile slicing: Group by percentiles
    - Range slicing: Group by value ranges
    - Cross-dimensional: Combinations of dimensions
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize data slicer.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()

    def slice_by_category(
        self,
        df: pd.DataFrame,
        dimension: str,
        min_slice_size: int = 10,
    ) -> list[DataSlice]:
        """
        Slice data by categorical values.

        Args:
            df: Source DataFrame
            dimension: Column name to slice by
            min_slice_size: Minimum samples per slice (smaller slices excluded)

        Returns:
            List of DataSlice objects, one per category

        Example:
            slices = slicer.slice_by_category(df, "neighborhood")
            for slice in slices:
                print(f"{slice.dimension}={slice.value}: {slice.size} samples")
        """
        if dimension not in df.columns:
            logger.warning(f"Column '{dimension}' not found in DataFrame")
            return []

        slices = []
        total_size = len(df)

        # Group by the dimension
        for value, group in df.groupby(dimension):
            if len(group) < min_slice_size:
                continue

            slices.append(
                DataSlice(
                    dimension=dimension,
                    value=value,
                    data=group,
                    size=len(group),
                    percentage=len(group) / total_size * 100,
                )
            )

        # Sort by size (largest first)
        slices.sort(key=lambda s: s.size, reverse=True)

        logger.info(
            f"Created {len(slices)} slices by '{dimension}'",
            extra={"dimension": dimension, "num_slices": len(slices)},
        )

        return slices

    def slice_by_quantiles(
        self,
        df: pd.DataFrame,
        dimension: str,
        num_quantiles: int = 4,
        labels: list[str] | None = None,
    ) -> list[DataSlice]:
        """
        Slice data by quantiles (e.g., quartiles, quintiles).

        Args:
            df: Source DataFrame
            dimension: Column name to slice by (must be numeric)
            num_quantiles: Number of quantiles (4 = quartiles, 5 = quintiles)
            labels: Optional labels for quantiles

        Returns:
            List of DataSlice objects, one per quantile

        Example:
            # Create quartiles for income
            slices = slicer.slice_by_quantiles(
                df, "income", num_quantiles=4,
                labels=["Q1_low", "Q2_medium-low", "Q3_medium-high", "Q4_high"]
            )
        """
        if dimension not in df.columns:
            logger.warning(f"Column '{dimension}' not found in DataFrame")
            return []

        if not pd.api.types.is_numeric_dtype(df[dimension]):
            logger.warning(f"Column '{dimension}' is not numeric")
            return []

        # Generate default labels if not provided
        if labels is None:
            labels = [f"Q{i + 1}" for i in range(num_quantiles)]

        # Create quantile bins
        df_copy = df.copy()
        df_copy[f"{dimension}_quantile"] = pd.qcut(
            df_copy[dimension],
            q=num_quantiles,
            labels=labels,
            duplicates="drop",
        )

        slices = []
        total_size = len(df)

        # Create slices for each quantile
        for quantile_label in labels:
            mask = df_copy[f"{dimension}_quantile"] == quantile_label
            group = df[mask]

            if len(group) == 0:
                continue

            slices.append(
                DataSlice(
                    dimension=f"{dimension}_quantile",
                    value=quantile_label,
                    data=group,
                    size=len(group),
                    percentage=len(group) / total_size * 100,
                )
            )

        logger.info(
            f"Created {len(slices)} quantile slices for '{dimension}'",
            extra={"dimension": dimension, "num_quantiles": num_quantiles},
        )

        return slices

    def slice_by_ranges(
        self,
        df: pd.DataFrame,
        dimension: str,
        ranges: list[tuple[float, float, str]],
    ) -> list[DataSlice]:
        """
        Slice data by custom ranges.

        Args:
            df: Source DataFrame
            dimension: Column name to slice by (must be numeric)
            ranges: List of (min, max, label) tuples

        Returns:
            List of DataSlice objects, one per range

        Example:
            # Create time-of-day slices
            slices = slicer.slice_by_ranges(df, "hour_of_day", [
                (0, 6, "night"),
                (6, 12, "morning"),
                (12, 18, "afternoon"),
                (18, 24, "evening"),
            ])
        """
        if dimension not in df.columns:
            logger.warning(f"Column '{dimension}' not found in DataFrame")
            return []

        slices = []
        total_size = len(df)

        for min_val, max_val, label in ranges:
            mask = (df[dimension] >= min_val) & (df[dimension] < max_val)
            group = df[mask]

            if len(group) == 0:
                continue

            slices.append(
                DataSlice(
                    dimension=f"{dimension}_range",
                    value=label,
                    data=group,
                    size=len(group),
                    percentage=len(group) / total_size * 100,
                )
            )

        logger.info(
            f"Created {len(slices)} range slices for '{dimension}'",
            extra={"dimension": dimension, "num_ranges": len(ranges)},
        )

        return slices

    def slice_by_time_of_day(
        self, df: pd.DataFrame, hour_column: str = "hour_of_day"
    ) -> list[DataSlice]:
        """
        Convenience method to slice by time periods.

        Args:
            df: Source DataFrame
            hour_column: Column containing hour (0-23)

        Returns:
            List of DataSlice objects for time periods
        """
        return self.slice_by_ranges(
            df,
            hour_column,
            [
                (0, 6, "night"),
                (6, 12, "morning"),
                (12, 18, "afternoon"),
                (18, 24, "evening"),
            ],
        )

    def get_default_slices(
        self,
        df: pd.DataFrame,
        dataset: str,
    ) -> dict[str, list[DataSlice]]:
        """
        Get default slices for a dataset based on configuration.

        Args:
            df: Source DataFrame
            dataset: Dataset name

        Returns:
            Dictionary mapping dimension names to slice lists

        Example:
            all_slices = slicer.get_default_slices(df, "crime")
            for dimension, slices in all_slices.items():
                print(f"{dimension}: {len(slices)} slices")
        """
        default_dimensions = self.config.fairness.default_slices
        result = {}

        for dimension in default_dimensions:
            if dimension not in df.columns:
                logger.debug(f"Skipping dimension '{dimension}' (not in DataFrame)")
                continue

            # Determine slicing strategy based on column type
            if dimension == "hour_of_day":
                slices = self.slice_by_time_of_day(df, dimension)
            elif pd.api.types.is_numeric_dtype(df[dimension]):
                # Numeric: use quantiles
                slices = self.slice_by_quantiles(df, dimension, num_quantiles=4)
            else:
                # Categorical: group by values
                slices = self.slice_by_category(df, dimension)

            if slices:
                result[dimension] = slices

        logger.info(
            f"Created default slices for {dataset}",
            extra={"dataset": dataset, "dimensions": len(result)},
        )

        return result

    def cross_slice(
        self,
        df: pd.DataFrame,
        dimensions: list[str],
        min_slice_size: int = 10,
    ) -> list[DataSlice]:
        """
        Create slices across multiple dimensions (intersection).

        Args:
            df: Source DataFrame
            dimensions: List of dimensions to combine
            min_slice_size: Minimum samples per slice

        Returns:
            List of DataSlice objects for cross-dimensional slices

        Example:
            # Slice by neighborhood AND time of day
            slices = slicer.cross_slice(df, ["neighborhood", "hour_of_day"])
        """
        # Check all dimensions exist
        missing = [dim for dim in dimensions if dim not in df.columns]
        if missing:
            logger.warning(f"Missing dimensions: {missing}")
            return []

        slices = []
        total_size = len(df)

        # Group by all dimensions
        for values, group in df.groupby(dimensions):
            if len(group) < min_slice_size:
                continue

            # Create combined dimension name and value
            if isinstance(values, tuple):
                combined_dimension = "_x_".join(dimensions)
                combined_value = "_x_".join(str(v) for v in values)
            else:
                combined_dimension = dimensions[0]
                combined_value = str(values)

            slices.append(
                DataSlice(
                    dimension=combined_dimension,
                    value=combined_value,
                    data=group,
                    size=len(group),
                    percentage=len(group) / total_size * 100,
                )
            )

        # Sort by size
        slices.sort(key=lambda s: s.size, reverse=True)

        logger.info(
            f"Created {len(slices)} cross-dimensional slices",
            extra={"dimensions": dimensions, "num_slices": len(slices)},
        )

        return slices

    def get_slice_summary(self, slices: list[DataSlice]) -> pd.DataFrame:
        """
        Get summary statistics for a set of slices.

        Args:
            slices: List of DataSlice objects

        Returns:
            DataFrame with slice statistics
        """
        summary = []
        for slice_obj in slices:
            summary.append(
                {
                    "dimension": slice_obj.dimension,
                    "value": slice_obj.value,
                    "size": slice_obj.size,
                    "percentage": slice_obj.percentage,
                }
            )

        return pd.DataFrame(summary)


# =============================================================================
# Convenience Functions
# =============================================================================


def slice_data(
    df: pd.DataFrame,
    dimension: str,
    slice_type: str = "auto",
    config: Settings | None = None,
    **kwargs: Any,
) -> list[DataSlice]:
    """
    Convenience function to slice data.

    Args:
        df: Source DataFrame
        dimension: Dimension to slice by
        slice_type: "categorical", "quantile", "range", or "auto"
        config: Configuration object
        **kwargs: Additional arguments for specific slice types

    Returns:
        List of DataSlice objects
    """
    slicer = DataSlicer(config)

    if slice_type == "auto":
        # Auto-detect based on column type
        slice_type = "quantile" if pd.api.types.is_numeric_dtype(df[dimension]) else "categorical"

    if slice_type == "categorical":
        return slicer.slice_by_category(df, dimension, **kwargs)
    elif slice_type == "quantile":
        return slicer.slice_by_quantiles(df, dimension, **kwargs)
    elif slice_type == "range":
        return slicer.slice_by_ranges(df, dimension, **kwargs)
    else:
        raise ValueError(f"Invalid slice_type: {slice_type}")
