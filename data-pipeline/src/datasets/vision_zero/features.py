"""
Boston Pulse - Vision Zero Feature Builder

Generates features for Vision Zero Safety Concerns data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureDefinition
from src.shared.config import Settings

logger = logging.getLogger(__name__)


class VisionZeroFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for Boston Vision Zero."""

    def get_dataset_name(self) -> str:
        return "vision_zero"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        return [
            FeatureDefinition(
                name="is_high_priority",
                description="Whether the safety concern is high priority",
                dtype="bool",
                source_columns=["request_type"],
            ),
            FeatureDefinition(
                name="vulnerable_mode",
                description="Whether the concern involves a vulnerable road user",
                dtype="bool",
                source_columns=["request_type"],
            ),
        ]

    def get_entity_key(self) -> str:
        return "concern_id"

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for vision zero."""
        if df.empty:
            return df

        # Flag high-risk concerns
        critical_types = ["speeding", "runlightssigns", "notyielding"]
        df["is_critical_type"] = df["request_type"].isin(critical_types)

        # Flag vulnerable modes
        vulnerable_modes = ["walks", "bikes"]
        df["is_vulnerable_mode"] = df["mode"].isin(vulnerable_modes)

        return df


def build_vision_zero_features(
    df: pd.DataFrame, execution_date: str, config: Settings | None = None
) -> dict[str, Any]:
    """Convenience function for building vision zero features."""
    builder = VisionZeroFeatureBuilder(config)
    result = builder.run(df, execution_date)
    return result.to_dict()
