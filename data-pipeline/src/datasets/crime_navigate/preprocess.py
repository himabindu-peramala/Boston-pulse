"""
Navigate Crime Preprocessor — fixed order of operations from config.

Drops only columns in raw_columns.dropped; renames; parses dates; coerce lat/long;
severity_weight from config; h3_index and hour_bucket; dedupe on incident_number.
Rows with null h3_index are kept in processed (excluded later in feature build).
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

import pandas as pd

from src.shared.config import get_dataset_config

logger = logging.getLogger(__name__)

try:
    import h3
except ImportError:
    h3 = None  # type: ignore[assignment]


def _load_config() -> dict[str, Any]:
    return get_dataset_config("crime_navigate")


def preprocess_crime_navigate(raw_df: pd.DataFrame, execution_date: str) -> pd.DataFrame:
    """Run full preprocessing; returns processed DataFrame."""
    cfg = _load_config()
    df = raw_df.copy()

    # 1. Drop columns in config.raw_columns.dropped (keep DISTRICT for fairness)
    dropped = cfg.get("raw_columns", {}).get("dropped", [])
    existing_drop = [c for c in dropped if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")

    # 2. Rename using column_mappings
    mappings = cfg.get("column_mappings", {})
    df = df.rename(columns=mappings)

    # 3. Parse occurred_on_date to UTC
    if "occurred_on_date" in df.columns:
        df["occurred_on_date"] = pd.to_datetime(df["occurred_on_date"], utc=True, errors="coerce")

    # 4. Coerce lat/long to float; null outside geo_bounds
    for col in ("lat", "long"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bounds = cfg.get("geo_bounds", {})
    min_lat = bounds.get("min_lat", 42.2)
    max_lat = bounds.get("max_lat", 42.4)
    min_lon = bounds.get("min_lon", -71.2)
    max_lon = bounds.get("max_lon", -70.9)
    if "lat" in df.columns:
        df.loc[(df["lat"].notna()) & ((df["lat"] < min_lat) | (df["lat"] > max_lat)), "lat"] = pd.NA
    if "long" in df.columns:
        df.loc[(df["long"].notna()) & ((df["long"] < min_lon) | (df["long"] > max_lon)), "long"] = pd.NA

    # 5. Shooting boolean: str(x).strip() == "1"
    if "shooting_raw" in df.columns:
        df["shooting"] = df["shooting_raw"].apply(lambda x: str(x).strip() == "1")
    else:
        df["shooting"] = False

    # 6. severity_weight from severity_weights + shooting_multiplier
    weights = cfg.get("severity_weights", {})
    default = weights.get("default_weight", 1.0)
    mult = cfg.get("shooting_multiplier", 2.0)
    desc_col = "offense_description" if "offense_description" in df.columns else None
    if desc_col:
        def assign_weight(row: pd.Series) -> float:
            val = default
            raw = row.get(desc_col)
            if pd.isna(raw):
                raw = ""
            upper = str(raw).upper()
            for keyword, w in weights.items():
                if keyword == "default_weight":
                    continue
                if keyword in upper:
                    val = float(w)
                    break
            if row.get("shooting"):
                val *= mult
            return val

        df["severity_weight"] = df.apply(assign_weight, axis=1)
    else:
        df["severity_weight"] = default

    # 7. Hour to int; day_of_week strip
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].astype(str).str.strip()

    # 8. h3_index
    res = cfg.get("h3", {}).get("resolution", 9)
    if h3 is not None and "lat" in df.columns and "long" in df.columns:
        def to_h3(r: pd.Series) -> str | None:
            try:
                lat, lon = r["lat"], r["long"]
                if pd.isna(lat) or pd.isna(lon):
                    return None
                return h3.latlng_to_cell(float(lat), float(lon), res)
            except Exception:
                return None

        df["h3_index"] = df.apply(to_h3, axis=1)
    else:
        df["h3_index"] = None

    # 9. hour_bucket from config.hour_buckets
    buckets = cfg.get("hour_buckets", {})
    bucket_list = [k for k in buckets if isinstance(k, int)]
    bucket_list.sort()

    def get_bucket(h: int) -> int | None:
        for b in bucket_list:
            rng = buckets[b]
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                lo, hi = int(rng[0]), int(rng[1])
                if lo <= h <= hi:
                    return b
        return None

    df["hour_bucket"] = df["hour"].apply(get_bucket)

    # 10. Drop duplicates on incident_number, keep last
    if "incident_number" in df.columns:
        df = df.drop_duplicates(subset=["incident_number"], keep="last")

    logger.info("Preprocessed crime_navigate: %d rows", len(df))
    return df


class CrimeNavigatePreprocessor:
    """Thin wrapper that calls preprocess_crime_navigate and holds result."""

    def __init__(self, config: Any = None):
        self._config = config
        self._data: pd.DataFrame | None = None

    def run(self, raw_df: pd.DataFrame, execution_date: str) -> dict[str, Any]:
        self._data = preprocess_crime_navigate(raw_df, execution_date)
        print(self._data.columns)
        print(self._data.dtypes)
        return {
            "rows_input": len(raw_df),
            "rows_output": len(self._data),
            "success": True,
        }

    def get_data(self) -> pd.DataFrame | None:
        return self._data
