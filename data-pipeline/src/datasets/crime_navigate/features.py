"""
Navigate Crime Feature Builder — 22 features + risk_score/risk_tier placeholders.

Expects processed_df with 90 days of data (DAG loads from GCS). Builds per (h3_index, hour_bucket);
neighbor features via h3.grid_disk; lag features 0.0 if not provided. Placeholder risk_score=0, risk_tier=LOW.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import h3
import pandas as pd

from src.datasets.base import BaseFeatureBuilder, FeatureBuildResult, FeatureDefinition
from src.shared.config import Settings, get_dataset_config

logger = logging.getLogger(__name__)

DATASET = "crime_navigate"


def _cfg() -> dict[str, Any]:
    return get_dataset_config(DATASET)


def build_navigate_features(
    processed_df: pd.DataFrame,
    execution_date: str,
    reference_date: datetime | None = None,
    yesterday_features: pd.DataFrame | None = None,
    features_7d_ago: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build full feature table for crime_navigate. Call from DAG after loading
    last 90 days of processed data. yesterday_features and features_7d_ago optional (0.0 if missing).
    """
    cfg = _cfg()
    df = processed_df[processed_df["h3_index"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    if reference_date is None:
        reference_date = pd.to_datetime(execution_date)
    # Work entirely in tz-naive timestamps for comparisons with df["_dt"]
    ref_ts = pd.Timestamp(reference_date).tz_localize(None)

    windows = cfg.get("feature_windows", {})
    short_d = windows.get("short_days", 3)
    medium_d = windows.get("medium_days", 10)
    baseline_d = windows.get("baseline_days", 30)
    long_d = windows.get("long_days", 90)
    caps = cfg.get("trend_caps", {})
    default_trend = cfg.get("trend_default_when_denominator_zero", 1.0)

    fc = cfg.get("feature_computation", {})
    trend_level = fc.get("trend_level", "cell")

    df["_dt"] = (
        pd.to_datetime(df["occurred_on_date"]).dt.tz_localize(None)
        if hasattr(df["occurred_on_date"].dtype, "tz")
        else pd.to_datetime(df["occurred_on_date"])
    )
    if df["_dt"].dt.tz is not None:
        df["_dt"] = df["_dt"].dt.tz_localize(None)

    rows: list[dict[str, Any]] = []
    cell_trends: dict[str, tuple[float, float, float]] = {}
    buckets_cfg = cfg.get("hour_buckets", {})
    bucket_ids = sorted([k for k in buckets_cfg if isinstance(k, int)])

    for (h3_index, hour_bucket), group in df.groupby(["h3_index", "hour_bucket"]):
        g = group.copy()
        base = ref_ts - pd.Timedelta(days=long_d)
        g = g[g["_dt"] >= base]
        if g.empty:
            continue

        # Bucket-level window sums (severity_weight) and counts
        def sum_in_days_bucket(days: int, g_=g) -> float:
            cut = ref_ts - pd.Timedelta(days=days)
            return g_[g_["_dt"] >= cut]["severity_weight"].sum()

        def count_in_days_bucket(days: int, g_=g) -> int:
            cut = ref_ts - pd.Timedelta(days=days)
            return len(g_[g_["_dt"] >= cut])

        ws_3 = sum_in_days_bucket(short_d)
        ws_10 = sum_in_days_bucket(medium_d)
        ws_30 = sum_in_days_bucket(baseline_d)
        ws_90 = sum_in_days_bucket(long_d)
        inc_30 = count_in_days_bucket(baseline_d)

        # Cell-level sums for trends if configured at cell level
        cell = df[(df["h3_index"] == h3_index) & (df["_dt"] >= base)]

        if trend_level == "cell":

            def sum_in_days_cell(days: int, cell_=cell) -> float:
                cut = ref_ts - pd.Timedelta(days=days)
                return cell_[cell_["_dt"] >= cut]["severity_weight"].sum()

            c_ws_3 = sum_in_days_cell(short_d)
            c_ws_10 = sum_in_days_cell(medium_d)
            c_ws_30 = sum_in_days_cell(baseline_d)
            c_ws_90 = sum_in_days_cell(long_d)
            daily_3 = c_ws_3 / short_d if short_d else 0
            daily_10 = c_ws_10 / medium_d if medium_d else 0
            daily_30 = c_ws_30 / baseline_d if baseline_d else 0
            daily_90 = c_ws_90 / long_d if long_d else 0
        else:
            # Bucket-level daily rates
            daily_3 = ws_3 / short_d if short_d else 0
            daily_10 = ws_10 / medium_d if medium_d else 0
            daily_30 = ws_30 / baseline_d if baseline_d else 0
            daily_90 = ws_90 / long_d if long_d else 0
        trend_3v10 = (daily_3 / daily_10) if daily_10 else default_trend
        trend_10v30 = (daily_10 / daily_30) if daily_30 else default_trend
        trend_30v90 = (daily_30 / daily_90) if daily_90 else default_trend
        trend_3v10 = min(trend_3v10, caps.get("trend_3v10", 3.4))
        trend_10v30 = min(trend_10v30, caps.get("trend_10v30", 3.1))
        trend_30v90 = min(trend_30v90, caps.get("trend_30v90", 3.1))

        # Record cell-level trends once per cell to broadcast later
        if trend_level == "cell" and h3_index not in cell_trends:
            cell_trends[h3_index] = (trend_3v10, trend_10v30, trend_30v90)

        # Violent (severity >= 5) and gun
        violent_30 = g[g["_dt"] >= (ref_ts - pd.Timedelta(days=baseline_d))]
        violent_30 = violent_30[violent_30["severity_weight"] >= 5.0]["severity_weight"].sum()
        gun_30 = (
            int(g[g["_dt"] >= (ref_ts - pd.Timedelta(days=baseline_d))]["shooting"].sum())
            if "shooting" in g.columns
            else 0
        )
        high_severity_ratio = (violent_30 / ws_30) if ws_30 else 0.0
        high_severity_ratio = min(high_severity_ratio, 1.0)

        # Temporal ratios: use full 30d for this h3_index (all hours)
        cell_30 = df[
            (df["h3_index"] == h3_index) & (df["_dt"] >= (ref_ts - pd.Timedelta(days=baseline_d)))
        ]
        ws_30_full = cell_30["severity_weight"].sum()
        night_buckets = {0, 5}
        night_score = (
            cell_30[cell_30["hour_bucket"].isin(night_buckets)]["severity_weight"].sum()
            if "hour_bucket" in cell_30.columns
            else 0
        )
        evening_score = (
            cell_30[cell_30["hour_bucket"] == 4]["severity_weight"].sum()
            if "hour_bucket" in cell_30.columns
            else 0
        )
        weekend = (
            cell_30[cell_30["day_of_week"].str.strip().str.lower().isin(["saturday", "sunday"])][
                "severity_weight"
            ].sum()
            if "day_of_week" in cell_30.columns
            else 0
        )
        night_ratio = (night_score / ws_30_full) if ws_30_full else 0.0
        evening_ratio = (evening_score / ws_30_full) if ws_30_full else 0.0
        weekend_ratio = (weekend / ws_30_full) if ws_30_full else 0.0
        night_ratio = min(night_ratio, 1.0)
        evening_ratio = min(evening_ratio, 1.0)
        weekend_ratio = min(weekend_ratio, 1.0)

        row = {
            "h3_index": h3_index,
            "hour_bucket": hour_bucket,
            "computed_date": execution_date,
            "weighted_score_3d": ws_3,
            "weighted_score_10d": ws_10,
            "weighted_score_30d": ws_30,
            "weighted_score_90d": ws_90,
            "incident_count_30d": inc_30,
            "trend_3v10": trend_3v10,
            "trend_10v30": trend_10v30,
            "trend_30v90": trend_30v90,
            "violent_score_30d": violent_30,
            "gun_incident_count_30d": gun_30,
            "high_severity_ratio_30d": high_severity_ratio,
            "night_score_ratio": night_ratio,
            "evening_score_ratio": evening_ratio,
            "weekend_score_ratio": weekend_ratio,
            "neighbor_weighted_score_30d": 0.0,
            "neighbor_trend_3v10": 0.0,
            "neighbor_gun_count_30d": 0.0,
            "cell_score_yesterday": 0.0,
            "cell_score_7d_ago": 0.0,
            "neighbor_score_yesterday": 0.0,
            "risk_score": 0.0,
            "risk_tier": "LOW",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Neighbor features (h3.grid_disk)
    if h3 is not None:
        rings = cfg.get("h3", {}).get("neighbor_rings", {0: 1.0, 1: 0.5, 2: 0.2})
        ws30_map = out.set_index(["h3_index", "hour_bucket"])["weighted_score_30d"].to_dict()
        tr_map = out.set_index(["h3_index", "hour_bucket"])["trend_3v10"].to_dict()
        gun_map = out.set_index(["h3_index", "hour_bucket"])["gun_incident_count_30d"].to_dict()

        def neighbor_agg(h3_idx: str, bucket: int, key: str) -> float:
            try:
                cells_k0 = {h3_idx}
                cells_k1 = set(h3.grid_disk(h3_idx, 1)) - cells_k0
                cells_k2 = set(h3.grid_disk(h3_idx, 2)) - cells_k0 - cells_k1
                w0 = rings.get(0, 1.0)
                w1 = rings.get(1, 0.5)
                w2 = rings.get(2, 0.2)
                if key == "weighted_score_30d":
                    m = ws30_map
                elif key == "trend_3v10":
                    m = tr_map
                else:
                    m = gun_map
                vals = [m.get((h3_idx, bucket), 0) * w0]
                for c in cells_k1:
                    vals.append(m.get((c, bucket), 0) * w1)
                for c in cells_k2:
                    vals.append(m.get((c, bucket), 0) * w2)
                total_w = w0 + len(cells_k1) * w1 + len(cells_k2) * w2
                if total_w <= 0:
                    return 0.0
                return sum(vals) / total_w
            except Exception:
                return 0.0

        out["neighbor_weighted_score_30d"] = out.apply(
            lambda r: neighbor_agg(r["h3_index"], r["hour_bucket"], "weighted_score_30d"), axis=1
        )
        out["neighbor_trend_3v10"] = out.apply(
            lambda r: neighbor_agg(r["h3_index"], r["hour_bucket"], "trend_3v10"), axis=1
        )
        out["neighbor_gun_count_30d"] = out.apply(
            lambda r: neighbor_agg(r["h3_index"], r["hour_bucket"], "gun_incident_count_30d"),
            axis=1,
        )

    # Lag placeholders from optional inputs
    if (
        yesterday_features is not None
        and not yesterday_features.empty
        and "weighted_score_30d" in yesterday_features.columns
    ):
        yest = yesterday_features.set_index(["h3_index", "hour_bucket"])["weighted_score_30d"]
        out["cell_score_yesterday"] = (
            out.set_index(["h3_index", "hour_bucket"])
            .index.map(lambda ix: yest.get(ix, 0.0))
            .values
        )
    if (
        features_7d_ago is not None
        and not features_7d_ago.empty
        and "weighted_score_30d" in features_7d_ago.columns
    ):
        past = features_7d_ago.set_index(["h3_index", "hour_bucket"])["weighted_score_30d"]
        out["cell_score_7d_ago"] = (
            out.set_index(["h3_index", "hour_bucket"])
            .index.map(lambda ix: past.get(ix, 0.0))
            .values
        )

    # Fill empty buckets: every h3_index gets 6 rows
    all_h3 = out["h3_index"].unique().tolist()
    extra_rows: list[dict[str, Any]] = []
    for h3_idx in all_h3:
        for b in bucket_ids:
            if ((out["h3_index"] == h3_idx) & (out["hour_bucket"] == b)).any():
                continue
            extra_rows.append(
                {
                    "h3_index": h3_idx,
                    "hour_bucket": b,
                    "computed_date": execution_date,
                    "weighted_score_3d": 0.0,
                    "weighted_score_10d": 0.0,
                    "weighted_score_30d": 0.0,
                    "weighted_score_90d": 0.0,
                    "incident_count_30d": 0,
                    "trend_3v10": default_trend,
                    "trend_10v30": default_trend,
                    "trend_30v90": default_trend,
                    "violent_score_30d": 0.0,
                    "gun_incident_count_30d": 0,
                    "high_severity_ratio_30d": 0.0,
                    "night_score_ratio": 0.0,
                    "evening_score_ratio": 0.0,
                    "weekend_score_ratio": 0.0,
                    "neighbor_weighted_score_30d": 0.0,
                    "neighbor_trend_3v10": 0.0,
                    "neighbor_gun_count_30d": 0.0,
                    "cell_score_yesterday": 0.0,
                    "cell_score_7d_ago": 0.0,
                    "neighbor_score_yesterday": 0.0,
                    "risk_score": 0.0,
                    "risk_tier": "LOW",
                }
            )
    if extra_rows:
        out = pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)

    # Broadcast cell-level trends to all buckets (including synthetic ones)
    if cell_trends:
        for h3_idx, (t3, t10, t30) in cell_trends.items():
            mask = out["h3_index"] == h3_idx
            out.loc[mask, "trend_3v10"] = t3
            out.loc[mask, "trend_10v30"] = t10
            out.loc[mask, "trend_30v90"] = t30
    return out


class CrimeNavigateFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for crime_navigate; uses build_navigate_features."""

    def __init__(self, config: Settings | None = None):
        super().__init__(config)
        self._data: pd.DataFrame | None = None

    def get_dataset_name(self) -> str:
        return DATASET

    def get_entity_key(self) -> str:
        return "h3_index"

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        return [
            FeatureDefinition("h3_index", "H3 cell", "string", []),
            FeatureDefinition("hour_bucket", "Hour bucket 0-5", "int", []),
            FeatureDefinition("computed_date", "Computed date", "string", []),
            FeatureDefinition("weighted_score_3d", "Weighted score 3d", "float", []),
            FeatureDefinition("weighted_score_10d", "Weighted score 10d", "float", []),
            FeatureDefinition("weighted_score_30d", "Weighted score 30d", "float", []),
            FeatureDefinition("weighted_score_90d", "Weighted score 90d", "float", []),
            FeatureDefinition("incident_count_30d", "Incident count 30d", "int", []),
            FeatureDefinition("trend_3v10", "Trend 3v10", "float", []),
            FeatureDefinition("trend_10v30", "Trend 10v30", "float", []),
            FeatureDefinition("trend_30v90", "Trend 30v90", "float", []),
            FeatureDefinition("violent_score_30d", "Violent score 30d", "float", []),
            FeatureDefinition("gun_incident_count_30d", "Gun count 30d", "int", []),
            FeatureDefinition("high_severity_ratio_30d", "High severity ratio", "float", []),
            FeatureDefinition("night_score_ratio", "Night ratio", "float", []),
            FeatureDefinition("evening_score_ratio", "Evening ratio", "float", []),
            FeatureDefinition("weekend_score_ratio", "Weekend ratio", "float", []),
            FeatureDefinition("neighbor_weighted_score_30d", "Neighbor score 30d", "float", []),
            FeatureDefinition("neighbor_trend_3v10", "Neighbor trend", "float", []),
            FeatureDefinition("neighbor_gun_count_30d", "Neighbor gun count", "float", []),
            FeatureDefinition("cell_score_yesterday", "Cell score yesterday", "float", []),
            FeatureDefinition("cell_score_7d_ago", "Cell score 7d ago", "float", []),
            FeatureDefinition("neighbor_score_yesterday", "Neighbor score yesterday", "float", []),
            FeatureDefinition("risk_score", "Risk score placeholder", "float", []),
            FeatureDefinition("risk_tier", "Risk tier placeholder", "string", []),
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        execution_date = (
            str(pd.to_datetime(df["occurred_on_date"]).max().date())
            if "occurred_on_date" in df.columns and len(df)
            else ""
        )
        if not execution_date:
            execution_date = datetime.now().strftime("%Y-%m-%d")
        result = build_navigate_features(processed_df=df, execution_date=execution_date)
        self._data = result
        return result

    def run(self, df: pd.DataFrame, execution_date: str) -> FeatureBuildResult:
        start = __import__("time").time()
        self._data = None
        out = self.build_features(df)
        self._data = out
        duration = __import__("time").time() - start
        return FeatureBuildResult(
            dataset=DATASET,
            execution_date=execution_date,
            rows_input=len(df),
            rows_output=len(out),
            features_computed=len(out.columns) if out is not None else 0,
            duration_seconds=duration,
            success=True,
        )

    def get_data(self) -> pd.DataFrame | None:
        return self._data
