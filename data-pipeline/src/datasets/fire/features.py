"""
Boston Pulse - Fire Feature Builder

Builds fire-related features for urban risk analytics.
District-based aggregations (no lat/long in source data).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class FireFeatureBuilder(BaseFeatureBuilder):
    WINDOW_SIZES = [7, 30, 90]

    def __init__(self, config: Settings | None = None):
        super().__init__(config)

    def get_dataset_name(self) -> str:
        return "fire"

    def get_entity_key(self) -> str:
        return "district"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        return [
            FeatureDefinition(
                name="district",
                description="Fire district",
                dtype="string",
                source_columns=["district"],
            ),
            FeatureDefinition(
                name="fire_count_7d",
                description="Fire incidents in last 7 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=7,
                min_value=0,
            ),
            FeatureDefinition(
                name="fire_count_30d",
                description="Fire incidents in last 30 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=30,
                min_value=0,
            ),
            FeatureDefinition(
                name="fire_count_90d",
                description="Fire incidents in last 90 days",
                dtype="int",
                source_columns=["incident_number"],
                aggregation="count",
                window_days=90,
                min_value=0,
            ),
            FeatureDefinition(
                name="avg_property_loss_30d",
                description="Average property loss in last 30 days",
                dtype="float",
                source_columns=["estimated_property_loss"],
                min_value=0.0,
            ),
            FeatureDefinition(
                name="night_fire_ratio",
                description="Ratio of incidents occurring at night",
                dtype="float",
                source_columns=["hour"],
                min_value=0.0,
                max_value=1.0,
            ),
            FeatureDefinition(
                name="weekend_fire_ratio",
                description="Ratio of incidents occurring on weekends",
                dtype="float",
                source_columns=["day_of_week"],
                min_value=0.0,
                max_value=1.0,
            ),
            FeatureDefinition(
                name="fire_risk_score",
                description="Normalized fire risk score (0-1)",
                dtype="float",
                source_columns=["fire_count_30d"],
                min_value=0.0,
                max_value=1.0,
            ),
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building fire features from {len(df)} records")

        if df.empty:
            logger.warning("Empty dataframe, skipping feature building")
            return pd.DataFrame()

        if "alarm_date" not in df.columns:
            logger.error("alarm_date column missing")
            return pd.DataFrame()

        df = df.copy()
        df["alarm_date"] = pd.to_datetime(df["alarm_date"], errors="coerce", utc=True)
        df = df[df["alarm_date"].notna()]

        if df.empty:
            return pd.DataFrame()

        reference_date = df["alarm_date"].max()
        logger.info(f"Reference date: {reference_date}")

        features = self._build_district_features(df, reference_date)
        logger.info(f"Built features for {len(features)} districts")
        return features

    def _build_district_features(self, df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        features_list = []

        for district, group in df.groupby("district"):
            features = self._compute_district_features(group, reference_date)
            features["district"] = district
            features_list.append(features)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        features_df = self._compute_risk_score(features_df)

        column_order = [f.name for f in self.get_feature_definitions()]
        available = [c for c in column_order if c in features_df.columns]
        return features_df[available]

    def _compute_district_features(
        self, group: pd.DataFrame, reference_date: datetime
    ) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for window in self.WINDOW_SIZES:
            mask = group["alarm_date"] >= (reference_date - pd.Timedelta(days=window))
            features[f"fire_count_{window}d"] = mask.sum()

        # Avg property loss last 30 days
        if "estimated_property_loss" in group.columns:
            mask_30d = group["alarm_date"] >= (reference_date - pd.Timedelta(days=30))
            features["avg_property_loss_30d"] = (
                group.loc[mask_30d, "estimated_property_loss"].mean() if mask_30d.sum() > 0 else 0.0
            )
        else:
            features["avg_property_loss_30d"] = 0.0

        features.update(self._compute_temporal_ratios(group))

        return features

    def _compute_temporal_ratios(self, group: pd.DataFrame) -> dict[str, float]:
        total = len(group)
        if total == 0:
            return {"night_fire_ratio": 0.0, "weekend_fire_ratio": 0.0}

        if "hour" in group.columns:
            night_mask = (group["hour"] >= 20) | (group["hour"] < 6)
            night_ratio = night_mask.sum() / total
        else:
            night_ratio = 0.0

        if "day_of_week" in group.columns:
            weekend_mask = group["day_of_week"].isin(["Saturday", "Sunday"])
            weekend_ratio = weekend_mask.sum() / total
        else:
            weekend_ratio = 0.0

        return {
            "night_fire_ratio": float(night_ratio),
            "weekend_fire_ratio": float(weekend_ratio),
        }

    def _compute_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        max_fire = df["fire_count_30d"].max()
        if max_fire > 0:
            df["fire_risk_score"] = (df["fire_count_30d"] / max_fire).clip(0, 1)
        else:
            df["fire_risk_score"] = 0.0
        return df


def build_fire_features(
    df: pd.DataFrame,
    execution_date: str,
    config: Settings | None = None,
) -> dict[str, Any]:
    builder = FireFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
