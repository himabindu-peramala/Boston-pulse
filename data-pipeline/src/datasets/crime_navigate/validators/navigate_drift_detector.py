"""Navigate drift detection — monitor only config columns."""

from __future__ import annotations

import pandas as pd

from src.shared.config import get_dataset_config
from src.validation.drift_detector import DriftDetector, DriftResult


def _cfg() -> dict:
    return get_dataset_config("crime_navigate")


class NavigateDriftDetector(DriftDetector):
    """Restricts drift monitoring to config.drift.columns_to_monitor."""

    DATASET = "crime_navigate"

    def detect_navigate_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> DriftResult:
        columns = _cfg().get("drift", {}).get("columns_to_monitor", [])
        if not columns:
            return self.detect_drift(current_df, reference_df, self.DATASET)
        common = [c for c in columns if c in current_df.columns and c in reference_df.columns]
        if not common:
            return self.detect_drift(current_df, reference_df, self.DATASET)
        cur = current_df[common].copy()
        ref = reference_df[common].copy()
        return self.detect_drift(cur, ref, self.DATASET)
