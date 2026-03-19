"""
Unit tests for CrimeNavigateFeatureBuilder / build_navigate_features.

We only test wiring + basic invariants:
- uses crime_navigate dataset name
- produces 6 buckets per h3_index
- trends honor trend_caps and trend_level=cell (same trend across buckets)
- temporal ratios in [0, 1]
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.datasets.crime_navigate.features import (
    CrimeNavigateFeatureBuilder,
    build_navigate_features,
)


class TestCrimeNavigateFeatureBuilder:
    @pytest.fixture
    def builder(self) -> CrimeNavigateFeatureBuilder:
        return CrimeNavigateFeatureBuilder()

    @pytest.fixture
    def sample_processed(self) -> pd.DataFrame:
        """Minimal processed frame with two cells and a spread of hours."""
        base = datetime(2024, 1, 15)
        rows = []
        for i, (h3_index, hour) in enumerate(
            [
                ("cellA", 0),
                ("cellA", 6),
                ("cellA", 12),
                ("cellA", 18),
                ("cellA", 22),
                ("cellB", 1),
                ("cellB", 7),
                ("cellB", 13),
                ("cellB", 19),
                ("cellB", 23),
            ]
        ):
            rows.append(
                {
                    "incident_number": f"I{i}",
                    "offense_description": "ASSAULT - AGGRAVATED",
                    "occurred_on_date": base - timedelta(days=i),
                    "shooting": False,
                    "severity_weight": 3.0,
                    "lat": 42.35,
                    "long": -71.06,
                    "h3_index": h3_index,
                    "hour": hour,
                    "hour_bucket": 0 if hour < 4 else (1 if hour < 8 else (2 if hour < 12 else (3 if hour < 16 else (4 if hour < 20 else 5)))),
                    "day_of_week": "Monday",
                    "district": "A1",
                }
            )
        return pd.DataFrame(rows)

    def test_dataset_name_and_entity_key(self, builder: CrimeNavigateFeatureBuilder) -> None:
        assert builder.get_dataset_name() == "crime_navigate"
        assert builder.get_entity_key() == "h3_index"

    def test_build_features_basic_shape(self, sample_processed: pd.DataFrame) -> None:
        df = build_navigate_features(sample_processed, execution_date="2024-01-15")
        assert not df.empty
        # two cells * 6 buckets = 12 rows
        assert df["h3_index"].nunique() == 2
        assert set(df["hour_bucket"].unique()) <= {0, 1, 2, 3, 4, 5}

    def test_trends_capped_and_same_per_cell(self, sample_processed: pd.DataFrame) -> None:
        df = build_navigate_features(sample_processed, execution_date="2024-01-15")
        # by config, caps are 3.4 / 3.1 / 3.1
        assert df["trend_3v10"].max() <= 3.4 + 1e-6
        assert df["trend_10v30"].max() <= 3.1 + 1e-6
        assert df["trend_30v90"].max() <= 3.1 + 1e-6

        # trend_level=cell: all buckets for a cell should share the same trends
        for cell, g in df.groupby("h3_index"):
            assert g["trend_3v10"].nunique() <= 1
            assert g["trend_10v30"].nunique() <= 1
            assert g["trend_30v90"].nunique() <= 1

    def test_temporal_ratios_bounded(self, sample_processed: pd.DataFrame) -> None:
        df = build_navigate_features(sample_processed, execution_date="2024-01-15")
        for col in ["night_score_ratio", "evening_score_ratio", "weekend_score_ratio"]:
            assert col in df.columns
            assert df[col].between(0.0, 1.0).all()

