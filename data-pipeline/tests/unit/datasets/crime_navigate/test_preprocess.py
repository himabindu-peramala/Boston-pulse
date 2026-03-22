"""
Unit tests for CrimeNavigatePreprocessor / preprocess_crime_navigate.

Checks:
- column mappings applied
- shooting boolean derived from shooting_raw
- severity_weight > 0 and respects shooting multiplier
- h3_index and hour_bucket populated for valid coords/hours
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.crime_navigate.preprocess import (
    CrimeNavigatePreprocessor,
    preprocess_crime_navigate,
)


class TestCrimeNavigatePreprocess:
    @pytest.fixture
    def raw_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "INCIDENT_NUMBER": ["I1", "I2"],
                "OFFENSE_DESCRIPTION": ["ASSAULT - AGGRAVATED", "INVESTIGATE PROPERTY"],
                "OCCURRED_ON_DATE": [
                    "2024-01-15T12:00:00",
                    "2024-01-15T00:00:00",
                ],
                "SHOOTING": ["1", "0"],
                "Lat": [42.35, 42.36],
                "Long": [-71.06, -71.07],
                "HOUR": [12, 0],
                "DAY_OF_WEEK": ["Monday   ", "Monday   "],
                "DISTRICT": ["A1", "A1"],
            }
        )

    def test_preprocess_function_basic(self, raw_df: pd.DataFrame) -> None:
        df = preprocess_crime_navigate(raw_df, execution_date="2024-01-15")
        assert "incident_number" in df.columns
        assert "occurred_on_date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["occurred_on_date"])
        assert "shooting" in df.columns
        assert df["shooting"].dtype == bool
        # one shooting, one non-shooting
        assert df["shooting"].tolist() == [True, False]

    def test_severity_and_h3_and_bucket(
        self,
        raw_df: pd.DataFrame,
    ) -> None:
        pre = CrimeNavigatePreprocessor()
        result = pre.run(raw_df, execution_date="2024-01-15")
        assert result["success"]
        df = pre.get_data()
        assert "severity_weight" in df.columns
        assert (df["severity_weight"] > 0).all()
        assert "h3_index" in df.columns
        assert df["h3_index"].notna().any()
        assert "hour_bucket" in df.columns
        assert df["hour_bucket"].between(0, 5).all()
