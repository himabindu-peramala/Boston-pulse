"""Navigate fairness — protected dimensions from config."""

from __future__ import annotations

import pandas as pd

from src.bias.fairness_checker import FairnessChecker, FairnessResult
from src.shared.config import get_dataset_config


def _cfg() -> dict:
    return get_dataset_config("crime_navigate")


class NavigateFairnessChecker(FairnessChecker):
    """Uses config.fairness.protected_dimensions for crime_navigate."""

    DATASET = "crime_navigate"

    def evaluate_navigate_fairness(
        self,
        features_df: pd.DataFrame,
        outcome_column: str | None = "risk_score",
    ) -> FairnessResult:
        dimensions = _cfg().get("fairness", {}).get("protected_dimensions", ["district"])
        # If features don't have district, use empty dimensions to avoid key error
        available = [d for d in dimensions if d in features_df.columns]
        return self.evaluate_fairness(
            df=features_df,
            dataset=self.DATASET,
            outcome_column=outcome_column,
            dimensions=available or None,
        )
