"""
Safety Scorer — H3 + Firestore spatial safety lookup for Boston.

Each H3 cell (resolution 9) has pre-computed risk scores stored in the
GCP Firestore ``h3_scores`` collection, keyed by ``{h3_index}_{hour_bucket}``.
The ML training pipeline publishes these scores after every run.

Hour buckets:
    0 → 00:00–03:59
    1 → 04:00–07:59
    2 → 08:00–11:59
    3 → 12:00–15:59
    4 → 16:00–19:59
    5 → 20:00–23:59

Usage
-----
    grid = SafetyGrid(project="bostonpulse")
    score = grid.get_score(lat=42.36, lon=-71.06, hour=23)
"""

from __future__ import annotations


import h3
from google.cloud import firestore


# H3 resolution matching the ML pipeline
H3_RESOLUTION = 9

# Number of hour buckets (24 hours / 4 = 6 buckets)
NUM_HOUR_BUCKETS = 6
HOURS_PER_BUCKET = 4

# Default score when no data found for a cell
DEFAULT_SCORE = 50.0
DEFAULT_TIER = "medium"


def _hour_to_bucket(hour: int) -> int:
    """Convert hour-of-day (0–23) to hour bucket (0–5)."""
    return hour // HOURS_PER_BUCKET


def _make_doc_key(h3_index: str, hour_bucket: int) -> str:
    """Build the Firestore document key ``{h3_index}_{hour_bucket}``."""
    return f"{h3_index}_{hour_bucket}"


class SafetyGrid:
    """Spatial grid of ML-computed risk scores backed by Firestore + H3."""

    def __init__(
        self,
        project: str = "bostonpulse",
        collection: str = "h3_scores",
        *,
        db: firestore.Client | None = None,
    ) -> None:
        """
        Parameters
        ----------
        project : str
            GCP project ID.
        collection : str
            Firestore collection name.
        db : firestore.Client, optional
            Inject a Firestore client (useful for testing).
        """
        self._db = db or firestore.Client(project=project)
        self._collection = collection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup(self, h3_index: str, hour_bucket: int) -> dict | None:
        """Fetch a single document from Firestore by composite key."""
        key = _make_doc_key(h3_index, hour_bucket)
        doc = self._db.collection(self._collection).document(key).get()
        if doc.exists:
            return doc.to_dict()
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_score(
        self,
        lat: float,
        lon: float,
        hour: int = 12,
        day_of_week: int = 0,
    ) -> float:
        """Return a risk score (0–100) for a location and time.

        Parameters
        ----------
        lat, lon : float
            WGS-84 coordinates.
        hour : int
            Hour of the day (0–23).
        day_of_week : int
            Kept for backward compatibility; unused (the ML model
            accounts for temporal patterns via hour buckets).

        Returns
        -------
        float
            Risk score clamped to [0, 100], or 50.0 if no data exists.
        """
        h3_index = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        hour_bucket = _hour_to_bucket(hour)
        data = self._lookup(h3_index, hour_bucket)

        if data is None:
            return DEFAULT_SCORE

        return float(data.get("risk_score", DEFAULT_SCORE))

    def get_score_with_tier(
        self,
        lat: float,
        lon: float,
        hour: int = 12,
        day_of_week: int = 0,
    ) -> dict:
        """Return risk score *and* tier for a location and time.

        Returns
        -------
        dict
            ``{"risk_score": float, "risk_tier": str}``
        """
        h3_index = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        hour_bucket = _hour_to_bucket(hour)
        data = self._lookup(h3_index, hour_bucket)

        if data is None:
            return {"risk_score": DEFAULT_SCORE, "risk_tier": DEFAULT_TIER}

        return {
            "risk_score": float(data.get("risk_score", DEFAULT_SCORE)),
            "risk_tier": data.get("risk_tier", DEFAULT_TIER),
        }

    def get_scores_along_path(
        self,
        coords: list[tuple[float, float]],
        hour: int = 12,
        day_of_week: int = 0,
    ) -> list[float]:
        """Return risk scores for each coordinate in a path."""
        return [
            self.get_score(lat, lon, hour, day_of_week) for lat, lon in coords
        ]
