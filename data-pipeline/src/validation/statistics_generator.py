"""
Boston Pulse - Statistics Generator

Generate and store data statistics using pandas and numpy.
Statistics are used for:
- Data quality monitoring
- Drift detection
- Schema inference
- Anomaly detection

Usage:
    generator = StatisticsGenerator(config)

    # Generate statistics for a dataset
    stats = generator.generate_statistics(df, dataset="crime", layer="raw")

    # Save statistics to GCS
    generator.save_statistics(stats, dataset="crime", layer="raw")

    # Load historical statistics
    stats = generator.load_statistics(dataset="crime", layer="raw", date="2024-01-15")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


@dataclass
class FeatureStatistics:
    """Statistics for a single feature."""

    name: str
    dtype: str
    count: int
    num_missing: int
    missing_ratio: float

    # Numerical stats (None for non-numeric)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    num_zeros: int | None = None

    # Categorical stats (None for numeric)
    num_unique: int | None = None
    top_values: dict[str, int] | None = None
    value_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "num_missing": self.num_missing,
            "missing_ratio": self.missing_ratio,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "num_zeros": self.num_zeros,
            "num_unique": self.num_unique,
            "top_values": self.top_values,
            "value_counts": self.value_counts,}


@dataclass
class DataStatistics:
    """Container for dataset statistics."""

    dataset: str
    layer: str  # raw, processed, features
    date: datetime
    num_examples: int
    num_features: int
    feature_statistics: list[FeatureStatistics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset,
            "layer": self.layer,
            "date": self.date.isoformat(),
            "num_examples": self.num_examples,
            "num_features": self.num_features,
            "feature_statistics": [f.to_dict() for f in self.feature_statistics],}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataStatistics:
        """Create DataStatistics from dictionary."""
        feature_stats = []
        for f in data.get("feature_statistics", []):
            feature_stats.append(
                FeatureStatistics(
                    name=f["name"],
                    dtype=f["dtype"],
                    count=f["count"],
                    num_missing=f["num_missing"],
                    missing_ratio=f["missing_ratio"],
                    mean=f.get("mean"),
                    std=f.get("std"),
                    min=f.get("min"),
                    max=f.get("max"),
                    median=f.get("median"),
                    q1=f.get("q1"),
                    q3=f.get("q3"),
                    num_zeros=f.get("num_zeros"),
                    num_unique=f.get("num_unique"),
                    top_values=f.get("top_values"),
                    value_counts=f.get("value_counts"),
                )
            )
        return cls(
            dataset=data["dataset"],
            layer=data["layer"],
            date=datetime.fromisoformat(data["date"]),
            num_examples=data["num_examples"],
            num_features=data["num_features"],
            feature_statistics=feature_stats,
        )

    def get_feature_stats(self, feature_name: str) -> FeatureStatistics | None:
        """Get statistics for a specific feature."""
        for f in self.feature_statistics:
            if f.name == feature_name:
                return f
        return None


class StatisticsGenerator:
    """
    Generate and manage dataset statistics using pandas and numpy.

    Statistics are stored in GCS at:
        gs://{bucket}/statistics/{dataset}/{layer}/{date}.json
        gs://{bucket}/statistics/{dataset}/{layer}/latest.json
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize statistics generator.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.bucket_name = self.config.storage.buckets.main
        self.stats_base_path = "statistics"

        # Initialize GCS client
        if self.config.storage.emulator.enabled:
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": self.config.storage.emulator.host},
            )
        else:
            self.client = storage.Client(project=self.config.gcp_project_id)

        self.bucket = self.client.bucket(self.bucket_name)

    def generate_statistics(
        self,
        df: pd.DataFrame,
        dataset: str,
        layer: str,
    ) -> DataStatistics:
        """
        Generate statistics for a DataFrame.

        Args:
            df: Source DataFrame
            dataset: Dataset name
            layer: Data layer (raw, processed, features)

        Returns:
            DataStatistics object with computed statistics

        Example:
            stats = generator.generate_statistics(crime_df, "crime", "raw")
            print(f"Generated stats for {stats.num_examples} examples")
        """
        logger.info(
            f"Generating statistics for {dataset}/{layer}",
            extra={"dataset": dataset, "layer": layer, "rows": len(df)},
        )

        feature_statistics = []

        for col in df.columns:
            feature_stats = self._compute_feature_statistics(df[col], col)
            feature_statistics.append(feature_stats)

        data_stats = DataStatistics(
            dataset=dataset,
            layer=layer,
            date=datetime.now(UTC),
            num_examples=len(df),
            num_features=len(df.columns),
            feature_statistics=feature_statistics,
        )

        logger.info(
            f"Generated statistics for {dataset}/{layer}: "
            f"{data_stats.num_examples} examples, {data_stats.num_features} features"
        )

        return data_stats

    def _compute_feature_statistics(self, series: pd.Series, name: str) -> FeatureStatistics:
        """Compute statistics for a single feature/column."""
        count = len(series)
        num_missing = int(series.isna().sum())
        missing_ratio = num_missing / count if count > 0 else 0.0
        dtype = str(series.dtype)

        # Initialize base statistics
        stats = FeatureStatistics(
            name=name,
            dtype=dtype,
            count=count,
            num_missing=num_missing,
            missing_ratio=missing_ratio,
        )

        # Get non-null values for calculations
        non_null = series.dropna()

        if len(non_null) == 0:
            return stats

        # Numerical statistics
        if pd.api.types.is_numeric_dtype(series):
            stats.mean = float(non_null.mean())
            stats.std = float(non_null.std()) if len(non_null) > 1 else 0.0
            stats.min = float(non_null.min())
            stats.max = float(non_null.max())
            stats.median = float(non_null.median())
            stats.q1 = float(non_null.quantile(0.25))
            stats.q3 = float(non_null.quantile(0.75))
            stats.num_zeros = int((non_null == 0).sum())
            stats.num_unique = int(non_null.nunique())
        else:
            # Categorical/String statistics
            stats.num_unique = int(non_null.nunique())

            # Get value counts
            value_counts = non_null.value_counts()
            stats.value_counts = {str(k): int(v) for k, v in value_counts.items()}

            # Top 10 most frequent values
            top_values = value_counts.head(10)
            stats.top_values = {str(k): int(v) for k, v in top_values.items()}

        return stats

    def save_statistics(
        self,
        stats: DataStatistics,
        date: datetime | None = None,
    ) -> str:
        """
        Save statistics to GCS.

        Args:
            stats: DataStatistics object to save
            date: Date to use in path (uses stats.date if not provided)

        Returns:
            GCS path where statistics were saved
        """
        if date is None:
            date = stats.date

        date_str = date.strftime("%Y%m%d")

        # Save versioned statistics (JSON format)
        versioned_path = f"{self.stats_base_path}/{stats.dataset}/{stats.layer}/{date_str}.json"
        self._upload_json(versioned_path, stats.to_dict())

        # Update latest pointer
        latest_path = f"{self.stats_base_path}/{stats.dataset}/{stats.layer}/latest.json"
        self._upload_json(latest_path, stats.to_dict())

        logger.info(
            f"Saved statistics for {stats.dataset}/{stats.layer} to {versioned_path}",
            extra={"dataset": stats.dataset, "layer": stats.layer, "path": versioned_path},
        )

        return f"gs://{self.bucket_name}/{versioned_path}"

    def load_statistics(
        self,
        dataset: str,
        layer: str,
        date: str | datetime | None = None,
    ) -> DataStatistics:
        """
        Load statistics from GCS.

        Args:
            dataset: Dataset name
            layer: Data layer
            date: Date string (YYYYMMDD) or datetime (uses latest if not provided)

        Returns:
            DataStatistics object

        Raises:
            FileNotFoundError: If statistics don't exist
        """
        if date is None:
            path = f"{self.stats_base_path}/{dataset}/{layer}/latest.json"
        else:
            date_str = date.strftime("%Y%m%d") if isinstance(date, datetime) else date
            path = f"{self.stats_base_path}/{dataset}/{layer}/{date_str}.json"

        try:
            data = self._download_json(path)
            return DataStatistics.from_dict(data)
        except Exception as e:
            raise FileNotFoundError(f"Statistics not found: {dataset}/{layer} date={date}") from e

    def list_statistics_dates(self, dataset: str, layer: str) -> list[str]:
        """
        List all available statistics dates for a dataset/layer.

        Returns:
            List of date strings (YYYYMMDD), sorted newest first
        """
        prefix = f"{self.stats_base_path}/{dataset}/{layer}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        dates = []
        for blob in blobs:
            filename = Path(blob.name).name
            if filename.endswith(".json") and filename != "latest.json":
                date_str = filename.replace(".json", "")
                # Only include valid date formats (8 digits)
                if date_str.isdigit() and len(date_str) == 8:
                    dates.append(date_str)

        dates.sort(reverse=True)
        return dates

    def compare_statistics(
        self,
        current: DataStatistics,
        reference: DataStatistics,
    ) -> dict[str, Any]:
        """
        Compare two sets of statistics.

        Args:
            current: Current dataset statistics
            reference: Reference (baseline) statistics

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "datasets": {"current": f"{current.dataset}/{current.layer}",
                "reference": f"{reference.dataset}/{reference.layer}",
            },
            "dates": {
                "current": current.date.isoformat(),
                "reference": reference.date.isoformat(),},
            "row_count_change": {
                "current": current.num_examples,
                "reference": reference.num_examples,
                "change": current.num_examples - reference.num_examples,
                "change_pct": (
                    (current.num_examples - reference.num_examples) / reference.num_examples * 100
                    if reference.num_examples > 0
                    else 0
                ),},
            "column_count_change": {
                "current": current.num_features,
                "reference": reference.num_features,
                "change": current.num_features - reference.num_features,},
            "feature_comparisons": [],
        }

        # Compare individual features
        current_features = {f.name: f for f in current.feature_statistics}
        reference_features = {f.name: f for f in reference.feature_statistics}

        all_features = set(current_features.keys()) | set(reference_features.keys())

        for feature_name in all_features:
            curr_feat = current_features.get(feature_name)
            ref_feat = reference_features.get(feature_name)

            feat_comparison = {
                "feature": feature_name,
                "in_current": curr_feat is not None,
                "in_reference": ref_feat is not None,}

            if curr_feat and ref_feat:
                feat_comparison["missing_ratio_change"] = (
                    curr_feat.missing_ratio - ref_feat.missing_ratio
                )

                # Compare numerical statistics
                if curr_feat.mean is not None and ref_feat.mean is not None:
                    feat_comparison["mean_change"] = curr_feat.mean - ref_feat.mean
                    feat_comparison["std_change"] = (curr_feat.std or 0) - (ref_feat.std or 0)

                # Compare unique counts
                if curr_feat.num_unique is not None and ref_feat.num_unique is not None:
                    feat_comparison["unique_count_change"] = (
                        curr_feat.num_unique - ref_feat.num_unique
                    )

            comparison["feature_comparisons"].append(feat_comparison)

        return comparison

    def visualize_statistics(
        self,
        stats: DataStatistics,
        output_path: str | None = None,
    ) -> str:
        """
        Generate HTML visualization of statistics.

        Args:
            stats: DataStatistics to visualize
            output_path: Optional path to save HTML (generates temp file if not provided)

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = (
                f"/tmp/stats_{stats.dataset}_{stats.layer}_{stats.date.strftime('%Y%m%d')}.html"
            )

        # Generate HTML report
        html_content = self._generate_html_report(stats)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated statistics visualization at {output_path}")
        return output_path

    def _generate_html_report(self, stats: DataStatistics) -> str:
        """Generate HTML report for statistics."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Statistics Report: {stats.dataset}/{stats.layer}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Statistics Report: {stats.dataset}/{stats.layer}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Date:</strong> {stats.date.isoformat()}</p>
        <p><strong>Number of Examples:</strong> {stats.num_examples:,}</p>
        <p><strong>Number of Features:</strong> {stats.num_features}</p>
    </div>

    <h2>Feature Statistics</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Type</th>
            <th>Count</th>
            <th>Missing</th>
            <th>Missing %</th>
            <th>Unique</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Min</th>
            <th>Max</th>
        </tr>
"""
        for f in stats.feature_statistics:
            html += f"""        <tr>
            <td>{f.name}</td>
            <td>{f.dtype}</td>
            <td>{f.count:,}</td>
            <td>{f.num_missing:,}</td>
            <td>{f.missing_ratio:.2%}</td>
            <td>{f.num_unique if f.num_unique else "-"}</td>
            <td>{f"{f.mean:.2f}" if f.mean is not None else "-"}</td>
            <td>{f"{f.std:.2f}" if f.std is not None else "-"}</td>
            <td>{f"{f.min:.2f}" if f.min is not None else "-"}</td>
            <td>{f"{f.max:.2f}" if f.max is not None else "-"}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""

        return html

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _upload_json(self, path: str, data: dict[str, Any]) -> None:
        """Upload JSON data to GCS."""
        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )

    def _download_json(self, path: str) -> dict[str, Any]:
        """Download JSON data from GCS."""
        blob = self.bucket.blob(path)
        content = blob.download_as_string()
        return json.loads(content)


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_and_save_statistics(
    df: pd.DataFrame,
    dataset: str,
    layer: str,
    config: Settings | None = None,
) -> DataStatistics:
    """
    Convenience function to generate and save statistics in one call.

    Args:
        df: Source DataFrame
        dataset: Dataset name
        layer: Data layer
        config: Configuration object

    Returns:
        DataStatistics object
    """
    generator = StatisticsGenerator(config)
    stats = generator.generate_statistics(df, dataset, layer)
    generator.save_statistics(stats)
    return stats


def get_latest_statistics(
    dataset: str,
    layer: str,
    config: Settings | None = None,
) -> DataStatistics:
    """
    Convenience function to get the latest statistics.

    Args:
        dataset: Dataset name
        layer: Data layer
        config: Configuration object

    Returns:
        Latest DataStatistics object
    """
    generator = StatisticsGenerator(config)
    return generator.load_statistics(dataset, layer)
