"""
Boston Pulse - Base Feature Builder

Abstract base class for all dataset feature builders. Provides a consistent interface
for feature engineering with:
- Aggregation over time windows
- Spatial feature computation
- Feature normalization
- Feature registry integration

Usage:
    class CrimeFeatureBuilder(BaseFeatureBuilder):
        def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
            ...
        def get_feature_definitions(self) -> list[FeatureDefinition]:
            ...
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
class FeatureDefinition:
    """Definition of a computed feature."""

    name: str
    description: str
    dtype: str
    source_columns: list[str]
    aggregation: str | None = None  # sum, mean, count, max, min, etc.
    window_days: int | None = None
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class FeatureBuildResult:
    """Result of a feature building operation."""

    dataset: str
    execution_date: str
    rows_input: int
    rows_output: int
    features_computed: int
    output_path: str | None = None
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for XCom/logging."""
        return {
            "dataset": self.dataset,
            "execution_date": self.execution_date,
            "rows_input": self.rows_input,
            "rows_output": self.rows_output,
            "features_computed": self.features_computed,
            "output_path": self.output_path,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "feature_stats": self.feature_stats,}


class BaseFeatureBuilder(ABC):
    """
    Abstract base class for feature building.

    Subclasses must implement:
    - build_features(): Compute features from processed data
    - get_dataset_name(): Return the dataset name
    - get_feature_definitions(): Return list of feature definitions
    - get_entity_key(): Return the entity key column(s)
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize the feature builder.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self._feature_stats: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features from processed data.

        Args:
            df: Processed DataFrame

        Returns:
            DataFrame with computed features
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
    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """
        Get list of feature definitions.

        Returns:
            List of FeatureDefinition objects describing each feature
        """
        pass

    @abstractmethod
    def get_entity_key(self) -> str | list[str]:
        """
        Get the entity key column(s) for aggregation.

        This is typically a spatial key like grid_cell or neighborhood.

        Returns:
            Column name or list of column names
        """
        pass

    def run(
        self,
        df: pd.DataFrame,
        execution_date: str,
    ) -> FeatureBuildResult:
        """
        Run the feature building pipeline.

        Args:
            df: Processed DataFrame
            execution_date: Execution date in YYYY-MM-DD format

        Returns:
            FeatureBuildResult with details about the feature building
        """
        import time

        start_time = time.time()
        dataset_name = self.get_dataset_name()
        rows_input = len(df)

        logger.info(
            f"Starting feature building for {dataset_name}",
            extra={
                "dataset": dataset_name,
                "execution_date": execution_date,
                "rows_input": rows_input,},
        )

        try:
            # Reset feature stats
            self._feature_stats = {}

            # Build features
            features_df = self.build_features(df)

            # Compute feature statistics
            self._compute_feature_stats(features_df)

            # Validate features
            self._validate_features(features_df)

            duration = time.time() - start_time

            result = FeatureBuildResult(
                dataset=dataset_name,
                execution_date=execution_date,
                rows_input=rows_input,
                rows_output=len(features_df),
                features_computed=len(features_df.columns),
                duration_seconds=duration,
                success=True,
                feature_stats=self._feature_stats,
            )

            logger.info(
                f"Feature building complete for {dataset_name}: "
                f"{len(features_df)} rows, {len(features_df.columns)} features",
                extra=result.to_dict(),
            )

            # Store features
            self._data = features_df

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Feature building failed for {dataset_name}: {e}",
                extra={"dataset": dataset_name, "error": str(e)},
                exc_info=True,
            )

            return FeatureBuildResult(
                dataset=dataset_name,
                execution_date=execution_date,
                rows_input=rows_input,
                rows_output=0,
                features_computed=0,
                duration_seconds=duration,
                success=False,
                error_message=str(e),
            )

    def get_data(self) -> pd.DataFrame | None:
        """Get the most recently built features."""
        return getattr(self, "_data", None)

    def _compute_feature_stats(self, df: pd.DataFrame) -> None:
        """Compute statistics for each feature."""
        for col in df.columns:
            stats: dict[str, Any] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "null_ratio": float(df[col].isna().mean()),}

            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    stats.update(
                        {
                            "mean": float(non_null.mean()),
                            "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "median": float(non_null.median()),}
                    )
            else:
                stats["unique_count"] = int(df[col].nunique())

            self._feature_stats[col] = stats

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate features against definitions."""
        definitions = {f.name: f for f in self.get_feature_definitions()}

        for col in df.columns:
            if col not in definitions:
                continue

            defn = definitions[col]

            # Check for nulls if not nullable
            if not defn.nullable and df[col].isna().any():
                logger.warning(f"Feature '{col}' has null values but is marked as non-nullable")

            # Check value ranges for numeric features
            if pd.api.types.is_numeric_dtype(df[col]):
                if defn.min_value is not None and (df[col] < defn.min_value).any():
                    logger.warning(f"Feature '{col}' has values below minimum {defn.min_value}")
                if defn.max_value is not None and (df[col] > defn.max_value).any():
                    logger.warning(f"Feature '{col}' has values above maximum {defn.max_value}")

    # ==========================================================================
    # Common Feature Building Utilities
    # ==========================================================================

    def compute_rolling_count(
        self,
        df: pd.DataFrame,
        group_col: str,
        date_col: str,
        window_days: int,
        output_col: str,
    ) -> pd.DataFrame:
        """
        Compute rolling count for each group over a time window.

        Args:
            df: Input DataFrame
            group_col: Column to group by (e.g., grid_cell)
            date_col: Datetime column for windowing
            window_days: Number of days in the rolling window
            output_col: Name of the output column

        Returns:
            DataFrame with the rolling count feature
        """
        df = df.sort_values(date_col)

        # Group and count
        counts = (
            df.groupby([group_col, pd.Grouper(key=date_col, freq="D")])
            .size()
            .reset_index(name="daily_count")
        )

        # Rolling sum
        counts[output_col] = (
            counts.groupby(group_col)["daily_count"]
            .rolling(window=window_days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        return counts[[group_col, date_col, output_col]]

    def compute_rolling_mean(
        self,
        df: pd.DataFrame,
        group_col: str,
        date_col: str,
        value_col: str,
        window_days: int,
        output_col: str,
    ) -> pd.DataFrame:
        """
        Compute rolling mean for each group over a time window.

        Args:
            df: Input DataFrame
            group_col: Column to group by
            date_col: Datetime column for windowing
            value_col: Column to compute mean of
            window_days: Number of days in the rolling window
            output_col: Name of the output column

        Returns:
            DataFrame with the rolling mean feature
        """
        df = df.sort_values(date_col)

        # Group and mean
        means = (
            df.groupby([group_col, pd.Grouper(key=date_col, freq="D")])[value_col]
            .mean()
            .reset_index(name="daily_mean")
        )

        # Rolling mean
        means[output_col] = (
            means.groupby(group_col)["daily_mean"]
            .rolling(window=window_days, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return means[[group_col, date_col, output_col]]

    def compute_category_ratio(
        self,
        df: pd.DataFrame,
        group_col: str,
        category_col: str,
        category_value: Any,
        output_col: str,
    ) -> pd.DataFrame:
        """
        Compute ratio of a specific category within each group.

        Args:
            df: Input DataFrame
            group_col: Column to group by
            category_col: Categorical column
            category_value: Value to compute ratio for
            output_col: Name of the output column

        Returns:
            DataFrame with the category ratio feature
        """
        total = df.groupby(group_col).size()
        category_count = df[df[category_col] == category_value].groupby(group_col).size()

        ratio = (category_count / total).fillna(0)
        result = pd.DataFrame({group_col: ratio.index, output_col: ratio.values})

        return result

    def add_time_features(
        self,
        df: pd.DataFrame,
        date_col: str,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Add common time-based features.

        Args:
            df: Input DataFrame
            date_col: Datetime column
            prefix: Prefix for output column names

        Returns:
            DataFrame with additional time features
        """
        p = f"{prefix}_" if prefix else ""

        df[f"{p}hour"] = df[date_col].dt.hour
        df[f"{p}dayofweek"] = df[date_col].dt.dayofweek
        df[f"{p}month"] = df[date_col].dt.month
        df[f"{p}is_weekend"] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        df[f"{p}is_night"] = df[date_col].dt.hour.isin(range(20, 24)).astype(int) | df[
            date_col
        ].dt.hour.isin(range(0, 6)).astype(int)

        return df

    def normalize_feature(
        self,
        df: pd.DataFrame,
        col: str,
        method: str = "minmax",
        output_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Normalize a feature.

        Args:
            df: Input DataFrame
            col: Column to normalize
            method: Normalization method ('minmax', 'zscore')
            output_col: Output column name (defaults to col_normalized)

        Returns:
            DataFrame with normalized feature
        """
        output_col = output_col or f"{col}_normalized"

        if method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[output_col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[output_col] = 0.0
        elif method == "zscore":
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[output_col] = (df[col] - mean_val) / std_val
            else:
                df[output_col] = 0.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return df
