"""
Boston Pulse - Crime Feature Builder

Builds crime-related features for urban analytics.

Features:
    - Crime counts by grid cell and time window
    - Crime category distributions
    - Shooting incident features
    - Temporal patterns (hour, day, month)
    - District-level aggregations

Usage:
    from src.datasets.crime.features import CrimeFeatureBuilder

    builder = CrimeFeatureBuilder()
    result = builder.run(processed_df, execution_date="2024-01-15")
    features_df = builder.get_data()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class CrimeFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for Boston crime data.

    Builds aggregated crime features at the grid cell level,
    suitable for use in urban analytics models.
    """

    # Grid cell size in degrees (approximately 100m at Boston's latitude)
    GRID_SIZE = 0.001

    # Rolling window sizes in days
    WINDOW_SIZES = [7, 30, 90]

    def __init__(self, config: Settings | None = None):
        """Initialize crime feature builder."""
        super().__init__(config)

    def get_dataset_name(self) -> str:
        """Return dataset name."""
        return "crime"

    def get_entity_key(self) -> str:
        """Return entity key for aggregation."""
        return "grid_cell"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Return feature definitions."""
        features = [
            # Grid identification
            FeatureDefinition(
                name="grid_cell",
                description="Grid cell identifier (lat_lon)",
                dtype="string",
                source_columns=["lat", "long"],
            ),
            FeatureDefinition(
                name="grid_lat",
                description="Grid cell center latitude",
                dtype="float",
                source_columns=["lat"],
                min_value=42.2,
                max_value=42.4,
            ),
            FeatureDefinition(
                name="grid_long",
                description="Grid cell center longitude",
                dtype="float",
                source_columns=["long"],
                min_value=-71.2,
                max_value=-70.9,
            ),
            # District
            FeatureDefinition(
                name="district",
                description="Police district",
                dtype="string",
                source_columns=["district"],
            ),
            # Crime counts
            FeatureDefinition(
                name="crime_count_7d",
                description="Total crimes in past 7 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=7,
                min_value=0,
            ),
            FeatureDefinition(
                name="crime_count_30d",
                description="Total crimes in past 30 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=30,
                min_value=0,
            ),
            FeatureDefinition(
                name="crime_count_90d",
                description="Total crimes in past 90 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=90,
                min_value=0,
            ),
            # Shooting features
            FeatureDefinition(
                name="shooting_count_30d",
                description="Shooting incidents in past 30 days",
                dtype="int",
                source_columns=["shooting"],
                aggregation="sum",
                window_days=30,
                min_value=0,
            ),
            FeatureDefinition(
                name="shooting_ratio_30d",
                description="Ratio of shootings to total crimes (30d)",
                dtype="float",
                source_columns=["shooting", "incident_number"],
                window_days=30,
                min_value=0.0,
                max_value=1.0,
            ),
            # Category features
            FeatureDefinition(
                name="violent_crime_ratio",
                description="Ratio of violent crimes to total",
                dtype="float",
                source_columns=["offense_category"],
                min_value=0.0,
                max_value=1.0,
            ),
            FeatureDefinition(
                name="property_crime_ratio",
                description="Ratio of property crimes to total",
                dtype="float",
                source_columns=["offense_category"],
                min_value=0.0,
                max_value=1.0,
            ),
            # Temporal features
            FeatureDefinition(
                name="night_crime_ratio",
                description="Ratio of crimes occurring at night (8pm-6am)",
                dtype="float",
                source_columns=["hour"],
                min_value=0.0,
                max_value=1.0,
            ),
            FeatureDefinition(
                name="weekend_crime_ratio",
                description="Ratio of crimes occurring on weekends",
                dtype="float",
                source_columns=["day_of_week"],
                min_value=0.0,
                max_value=1.0,
            ),
            # Risk score
            FeatureDefinition(
                name="crime_risk_score",
                description="Normalized crime risk score (0-1)",
                dtype="float",
                source_columns=[
                    "crime_count_30d",
                    "shooting_count_30d",
                    "violent_crime_ratio",
                ],
                min_value=0.0,
                max_value=1.0,
            ),
        ]
        return features

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build crime features from processed data.

        Args:
            df: Processed crime DataFrame

        Returns:
            DataFrame with aggregated features per grid cell
        """
        logger.info(f"Building crime features from {len(df)} records")

        # Filter to records with valid coordinates
        df = df[df["lat"].notna() & df["long"].notna()].copy()

        if len(df) == 0:
            logger.warning("No records with valid coordinates for feature building")
            return pd.DataFrame()

        # Add grid cell
        df = self._add_grid_cell(df)

        # Get the reference date (most recent date in data)
        reference_date = df["occurred_on_date"].max()
        logger.info(f"Using reference date: {reference_date}")

        # Build features by grid cell
        features = self._build_grid_features(df, reference_date)

        logger.info(f"Built features for {len(features)} grid cells")

        return features

    def _add_grid_cell(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add grid cell identifier based on coordinates."""
        # Round to grid size
        df["grid_lat"] = (df["lat"] / self.GRID_SIZE).round() * self.GRID_SIZE
        df["grid_long"] = (df["long"] / self.GRID_SIZE).round() * self.GRID_SIZE

        # Create cell identifier
        df["grid_cell"] = (
            df["grid_lat"].apply(lambda x: f"{x:.3f}")
            + "_"
            + df["grid_long"].apply(lambda x: f"{x:.3f}")
        )

        return df

    def _build_grid_features(self, df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Build aggregated features per grid cell."""
        features_list = []

        for grid_cell, group in df.groupby("grid_cell"):
            features = self._compute_cell_features(group, reference_date)
            features["grid_cell"] = grid_cell
            features_list.append(features)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)

        # Add grid coordinates from cell ID
        features_df["grid_lat"] = features_df["grid_cell"].apply(lambda x: float(x.split("_")[0]))
        features_df["grid_long"] = features_df["grid_cell"].apply(lambda x: float(x.split("_")[1]))

        # Compute risk score
        features_df = self._compute_risk_score(features_df)

        # Reorder columns
        column_order = [f.name for f in self.get_feature_definitions()]
        available_columns = [c for c in column_order if c in features_df.columns]
        features_df = features_df[available_columns]

        return features_df

    def _compute_cell_features(
        self, group: pd.DataFrame, reference_date: datetime
    ) -> dict[str, Any]:
        """Compute features for a single grid cell."""
        features: dict[str, Any] = {}

        # District (mode)
        features["district"] = group["district"].mode().iloc[0] if len(group) > 0 else "UNKNOWN"

        # Crime counts by window
        for window in self.WINDOW_SIZES:
            window_mask = group["occurred_on_date"] >= (reference_date - pd.Timedelta(days=window))
            window_data = group[window_mask]

            features[f"crime_count_{window}d"] = len(window_data)

            # Shooting count (only for 30d)
            if window == 30:
                if "shooting" in group.columns:
                    features["shooting_count_30d"] = int(window_data["shooting"].sum())
                else:
                    features["shooting_count_30d"] = 0

                # Shooting ratio
                if len(window_data) > 0:
                    features["shooting_ratio_30d"] = features["shooting_count_30d"] / len(
                        window_data
                    )
                else:
                    features["shooting_ratio_30d"] = 0.0

        # Category ratios (using all data for this cell)
        features.update(self._compute_category_ratios(group))

        # Temporal ratios
        features.update(self._compute_temporal_ratios(group))

        return features

    def _compute_category_ratios(self, group: pd.DataFrame) -> dict[str, float]:
        """Compute crime category ratios."""
        ratios = {}
        total = len(group)

        if total == 0:
            return {
                "violent_crime_ratio": 0.0,
                "property_crime_ratio": 0.0,
            }

        # Define violent crime categories
        violent_categories = [
            "Assault",
            "Homicide",
            "Robbery",
            "Aggravated Assault",
            "Simple Assault",
            "Offenses Against Child / Family",
        ]

        # Define property crime categories
        property_categories = [
            "Larceny",
            "Motor Vehicle Accident Response",
            "Vandalism",
            "Burglary",
            "Auto Theft",
            "Larceny From Motor Vehicle",
            "Residential Burglary",
            "Commercial Burglary",
        ]

        if "offense_category" in group.columns:
            violent_count = group["offense_category"].str.title().isin(violent_categories).sum()
            property_count = group["offense_category"].str.title().isin(property_categories).sum()

            ratios["violent_crime_ratio"] = violent_count / total
            ratios["property_crime_ratio"] = property_count / total
        else:
            ratios["violent_crime_ratio"] = 0.0
            ratios["property_crime_ratio"] = 0.0

        return ratios

    def _compute_temporal_ratios(self, group: pd.DataFrame) -> dict[str, float]:
        """Compute temporal pattern ratios."""
        ratios = {}
        total = len(group)

        if total == 0:
            return {
                "night_crime_ratio": 0.0,
                "weekend_crime_ratio": 0.0,
            }

        # Night crimes (8pm - 6am)
        if "hour" in group.columns:
            night_mask = (group["hour"] >= 20) | (group["hour"] < 6)
            ratios["night_crime_ratio"] = night_mask.sum() / total
        else:
            ratios["night_crime_ratio"] = 0.0

        # Weekend crimes
        if "day_of_week" in group.columns:
            weekend_days = ["Saturday", "Sunday"]
            weekend_mask = group["day_of_week"].isin(weekend_days)
            ratios["weekend_crime_ratio"] = weekend_mask.sum() / total
        elif "occurred_on_date" in group.columns:
            weekend_mask = group["occurred_on_date"].dt.dayofweek.isin([5, 6])
            ratios["weekend_crime_ratio"] = weekend_mask.sum() / total
        else:
            ratios["weekend_crime_ratio"] = 0.0

        return ratios

    def _compute_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute normalized crime risk score.

        Risk score is a weighted combination of:
        - Crime count (normalized)
        - Shooting presence
        - Violent crime ratio
        """
        # Normalize crime count (0-1)
        max_crime = df["crime_count_30d"].max()
        if max_crime > 0:
            crime_normalized = df["crime_count_30d"] / max_crime
        else:
            crime_normalized = 0.0

        # Shooting component (binary boost if any shootings)
        shooting_component = (df["shooting_count_30d"] > 0).astype(float) * 0.2

        # Weighted score
        weights = {
            "crime": 0.5,
            "violent": 0.3,
            "shooting": 0.2,
        }

        df["crime_risk_score"] = (
            crime_normalized * weights["crime"]
            + df["violent_crime_ratio"] * weights["violent"]
            + shooting_component
        )

        # Ensure 0-1 range
        df["crime_risk_score"] = df["crime_risk_score"].clip(0, 1)

        return df


# =============================================================================
# Convenience Functions
# =============================================================================


def build_crime_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Convenience function for building crime features.

    Returns result dictionary suitable for Airflow XCom.
    """
    builder = CrimeFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
