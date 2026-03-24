"""Unit tests for SafetyGrid (Firestore + H3 backed)."""

from unittest.mock import MagicMock

import h3
import pytest

from app.safety.safety_scorer import (
    DEFAULT_SCORE,
    DEFAULT_TIER,
    H3_RESOLUTION,
    SafetyGrid,
    _hour_to_bucket,
    _make_doc_key,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_firestore_with_docs(docs: dict):
    """Return a mock Firestore client preloaded with documents.

    Parameters
    ----------
    docs : dict
        Mapping of doc key → dict payload.  Keys not present will
        behave as non-existent documents.
    """
    mock_db = MagicMock()

    def _get_doc(key):
        mock_doc_ref = MagicMock()
        mock_snapshot = MagicMock()
        if key in docs:
            mock_snapshot.exists = True
            mock_snapshot.to_dict.return_value = docs[key]
        else:
            mock_snapshot.exists = False
            mock_snapshot.to_dict.return_value = None
        mock_doc_ref.get.return_value = mock_snapshot
        return mock_doc_ref

    mock_collection = MagicMock()
    mock_collection.document.side_effect = _get_doc
    mock_db.collection.return_value = mock_collection
    return mock_db


# Sample coordinates in downtown Boston
SAMPLE_LAT, SAMPLE_LON = 42.3601, -71.0589
SAMPLE_H3 = h3.latlng_to_cell(SAMPLE_LAT, SAMPLE_LON, H3_RESOLUTION)


@pytest.fixture
def mock_docs():
    """Pre-built Firestore documents for tests."""
    return {
        f"{SAMPLE_H3}_0": {
            "h3_index": SAMPLE_H3,
            "hour_bucket": 0,
            "risk_score": 72.5,
            "risk_tier": "high",
            "predicted_danger": 0.8,
            "model_version": "v1",
        },
        f"{SAMPLE_H3}_3": {
            "h3_index": SAMPLE_H3,
            "hour_bucket": 3,
            "risk_score": 35.0,
            "risk_tier": "medium",
            "predicted_danger": 0.3,
            "model_version": "v1",
        },
        f"{SAMPLE_H3}_5": {
            "h3_index": SAMPLE_H3,
            "hour_bucket": 5,
            "risk_score": 88.0,
            "risk_tier": "high",
            "predicted_danger": 0.9,
            "model_version": "v1",
        },
    }


@pytest.fixture
def grid(mock_docs):
    """SafetyGrid wired to a mock Firestore client."""
    mock_db = _mock_firestore_with_docs(mock_docs)
    return SafetyGrid(db=mock_db)


# ── Unit helpers ─────────────────────────────────────────────────────────────


class TestHourBucket:
    def test_midnight(self):
        assert _hour_to_bucket(0) == 0

    def test_3am(self):
        assert _hour_to_bucket(3) == 0

    def test_4am(self):
        assert _hour_to_bucket(4) == 1

    def test_noon(self):
        assert _hour_to_bucket(12) == 3

    def test_11pm(self):
        assert _hour_to_bucket(23) == 5

    def test_all_buckets(self):
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        for hour, bucket in enumerate(expected):
            assert _hour_to_bucket(hour) == bucket, f"hour={hour}"


class TestDocKey:
    def test_format(self):
        assert _make_doc_key("8a2a100d2dfffff", 3) == "8a2a100d2dfffff_3"

    def test_bucket_zero(self):
        assert _make_doc_key("abc", 0) == "abc_0"


class TestH3Resolution:
    def test_valid_h3_index(self):
        idx = h3.latlng_to_cell(42.36, -71.06, H3_RESOLUTION)
        assert h3.get_resolution(idx) == 9


# ── SafetyGrid tests ────────────────────────────────────────────────────────


class TestGetScore:
    def test_returns_score_from_firestore(self, grid):
        # hour=1 → bucket 0, which has risk_score 72.5
        score = grid.get_score(SAMPLE_LAT, SAMPLE_LON, hour=1)
        assert score == 72.5

    def test_different_hour_bucket(self, grid):
        # hour=12 → bucket 3, which has risk_score 35.0
        score = grid.get_score(SAMPLE_LAT, SAMPLE_LON, hour=12)
        assert score == 35.0

    def test_night_hour_bucket(self, grid):
        # hour=22 → bucket 5, which has risk_score 88.0
        score = grid.get_score(SAMPLE_LAT, SAMPLE_LON, hour=22)
        assert score == 88.0

    def test_missing_cell_returns_default(self, grid):
        # Coordinates far from sample → different H3 → no doc
        score = grid.get_score(0.0, 0.0, hour=12)
        assert score == DEFAULT_SCORE

    def test_missing_bucket_returns_default(self, grid):
        # bucket 2 has no doc for this cell
        score = grid.get_score(SAMPLE_LAT, SAMPLE_LON, hour=8)
        assert score == DEFAULT_SCORE

    def test_returns_in_range(self, grid):
        for hour in [1, 12, 22]:
            score = grid.get_score(SAMPLE_LAT, SAMPLE_LON, hour=hour)
            assert 0 <= score <= 100


class TestGetScoreWithTier:
    def test_returns_score_and_tier(self, grid):
        result = grid.get_score_with_tier(SAMPLE_LAT, SAMPLE_LON, hour=1)
        assert result["risk_score"] == 72.5
        assert result["risk_tier"] == "high"

    def test_missing_returns_defaults(self, grid):
        result = grid.get_score_with_tier(0.0, 0.0, hour=12)
        assert result["risk_score"] == DEFAULT_SCORE
        assert result["risk_tier"] == DEFAULT_TIER


class TestGetScoresAlongPath:
    def test_returns_list(self, grid):
        coords = [(SAMPLE_LAT, SAMPLE_LON), (SAMPLE_LAT, SAMPLE_LON)]
        scores = grid.get_scores_along_path(coords, hour=12)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_mixed_known_unknown(self, grid):
        coords = [(SAMPLE_LAT, SAMPLE_LON), (0.0, 0.0)]
        scores = grid.get_scores_along_path(coords, hour=12)
        assert scores[0] == 35.0  # bucket 3
        assert scores[1] == DEFAULT_SCORE
