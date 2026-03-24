"""Unit tests for route ranking logic."""

from unittest.mock import MagicMock

import h3
import pytest

from app.routes.route_ranker import rank_routes, score_route
from app.safety.safety_scorer import H3_RESOLUTION, SafetyGrid


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_firestore_with_docs(docs: dict):
    """Return a mock Firestore client preloaded with documents."""
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


# Test coordinates
COORDS = [
    (42.3500, -71.0700),
    (42.3520, -71.0680),
    (42.3540, -71.0660),
    (42.3560, -71.0640),
    (42.3580, -71.0620),
    (42.3600, -71.0600),
    (42.3620, -71.0580),
]

# Pre-assign varying risk scores per H3 cell (for hour bucket 3 = noon)
RISK_SCORES = [85.0, 70.0, 55.0, 40.0, 65.0, 80.0, 60.0]


def _build_test_docs():
    """Build Firestore docs for all test coordinates at hour bucket 3 (noon)."""
    docs = {}
    for (lat, lon), score in zip(COORDS, RISK_SCORES):
        h3_idx = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        key = f"{h3_idx}_3"  # bucket 3 = hours 12–15
        docs[key] = {
            "h3_index": h3_idx,
            "hour_bucket": 3,
            "risk_score": score,
            "risk_tier": "high" if score >= 66 else ("medium" if score >= 33 else "low"),
            "predicted_danger": score / 100.0,
            "model_version": "test",
        }
    return docs


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_grid():
    """SafetyGrid backed by a mock Firestore with test data."""
    docs = _build_test_docs()
    mock_db = _mock_firestore_with_docs(docs)
    return SafetyGrid(db=mock_db)


def _make_route(coords, distance_m, duration_s):
    return {
        "geometry": coords,
        "distance_m": distance_m,
        "duration_s": duration_s,
    }


# ── Tests ────────────────────────────────────────────────────────────────────


class TestScoreRoute:
    def test_returns_float(self, sample_grid):
        coords = [(42.3600, -71.0600), (42.3580, -71.0620)]
        score = score_route(coords, sample_grid, hour=12, day_of_week=0)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_empty_coords_returns_default(self, sample_grid):
        assert score_route([], sample_grid) == 50.0


class TestRankRoutes:
    def test_three_routes_labeled(self, sample_grid):
        routes = [
            _make_route(
                [(42.3600, -71.0600), (42.3580, -71.0620)],
                distance_m=500,
                duration_s=400,
            ),
            _make_route(
                [(42.3500, -71.0700), (42.3520, -71.0680)],
                distance_m=800,
                duration_s=700,
            ),
            _make_route(
                [(42.3540, -71.0660), (42.3560, -71.0640)],
                distance_m=600,
                duration_s=300,
            ),
        ]
        ranked = rank_routes(routes, sample_grid, hour=12, day_of_week=0)

        labels = [r["rank_label"] for r in ranked]
        assert "safest" in labels
        assert "fastest" in labels
        assert "balanced" in labels

    def test_all_routes_have_safety_score(self, sample_grid):
        routes = [
            _make_route(
                [(42.3600, -71.0600)],
                distance_m=500,
                duration_s=400,
            ),
            _make_route(
                [(42.3500, -71.0700)],
                distance_m=800,
                duration_s=700,
            ),
        ]
        ranked = rank_routes(routes, sample_grid)
        for r in ranked:
            assert "safety_score" in r
            assert 0 <= r["safety_score"] <= 100

    def test_single_route(self, sample_grid):
        routes = [
            _make_route(
                [(42.3600, -71.0600)],
                distance_m=500,
                duration_s=400,
            ),
        ]
        ranked = rank_routes(routes, sample_grid)
        assert len(ranked) == 1
        assert "safest" in ranked[0]["rank_label"]
        assert "fastest" in ranked[0]["rank_label"]
        assert "balanced" in ranked[0]["rank_label"]

    def test_empty_routes(self, sample_grid):
        assert rank_routes([], sample_grid) == []

    def test_safest_has_highest_score(self, sample_grid):
        routes = [
            _make_route(
                [(42.3600, -71.0600), (42.3620, -71.0580)],
                distance_m=500,
                duration_s=400,
            ),
            _make_route(
                [(42.3540, -71.0660), (42.3560, -71.0640)],
                distance_m=600,
                duration_s=300,
            ),
            _make_route(
                [(42.3500, -71.0700), (42.3520, -71.0680)],
                distance_m=900,
                duration_s=800,
            ),
        ]
        ranked = rank_routes(routes, sample_grid, hour=12, day_of_week=0)
        safest = next(r for r in ranked if r["rank_label"] == "safest")
        other_scores = [
            r["safety_score"] for r in ranked if r["rank_label"] != "safest"
        ]
        assert all(safest["safety_score"] >= s for s in other_scores)

    def test_fastest_has_shortest_duration(self, sample_grid):
        routes = [
            _make_route([(42.3600, -71.0600)], 500, 400),
            _make_route([(42.3500, -71.0700)], 600, 200),
            _make_route([(42.3540, -71.0660)], 700, 600),
        ]
        ranked = rank_routes(routes, sample_grid, hour=12, day_of_week=0)
        fastest = next(r for r in ranked if r["rank_label"] == "fastest")
        other_durations = [
            r["duration_s"] for r in ranked if r["rank_label"] != "fastest"
        ]
        assert all(fastest["duration_s"] <= d for d in other_durations)
