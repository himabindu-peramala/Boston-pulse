"""
Route Ranker — score and rank walking routes by safety & speed.

Given a list of decoded routes (from the ORS client) and a loaded
SafetyGrid, this module:
1. Samples safety scores along each route's geometry.
2. Computes an average safety score per route.
3. Labels routes as **safest**, **balanced**, or **fastest**.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.safety.safety_scorer import SafetyGrid


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_route(
    route_coords: list[tuple[float, float]],
    safety_grid: SafetyGrid,
    hour: int = 12,
    day_of_week: int = 0,
    max_sample_points: int = 100,
) -> float:
    """Return the average safety score for a route.

    To avoid excessive computation on very long routes, we uniformly
    sample at most *max_sample_points* from the geometry.

    Parameters
    ----------
    route_coords : list of (lat, lon)
    safety_grid : SafetyGrid instance
    hour : 0–23
    day_of_week : 0=Mon … 6=Sun
    max_sample_points : int

    Returns
    -------
    float  — average safety score (0–100)
    """
    if not route_coords:
        return 50.0

    # Uniform sampling when route has many points
    n = len(route_coords)
    if n > max_sample_points:
        indices = np.linspace(0, n - 1, max_sample_points, dtype=int)
        sampled = [route_coords[i] for i in indices]
    else:
        sampled = route_coords

    scores = safety_grid.get_scores_along_path(sampled, hour, day_of_week)
    return float(np.mean(scores))


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------

def rank_routes(
    routes: list[dict[str, Any]],
    safety_grid: SafetyGrid,
    hour: int = 12,
    day_of_week: int = 0,
    safety_weight: float = 0.6,
    speed_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Score every route and assign rank labels.

    Labels assigned:
    - **safest**  — highest average safety score
    - **fastest** — shortest duration
    - **balanced** — best composite of safety and inverse-duration

    If only one route is returned (ORS sometimes does), it receives
    all three labels joined (``safest / fastest / balanced``).

    Parameters
    ----------
    routes : list of route dicts from ORSClient
    safety_grid : loaded SafetyGrid
    hour, day_of_week : time context
    safety_weight, speed_weight : weights for the balanced score

    Returns
    -------
    list[dict]
        Same route dicts enriched with ``safety_score`` and ``rank_label``.
    """
    if not routes:
        return []

    # 1. Compute safety scores
    for route in routes:
        route["safety_score"] = round(
            score_route(route["geometry"], safety_grid, hour, day_of_week),
            2,
        )

    # 2. Edge case: single route
    if len(routes) == 1:
        routes[0]["rank_label"] = "safest / fastest / balanced"
        return routes

    # 3. Determine safest & fastest
    safest_idx = int(np.argmax([r["safety_score"] for r in routes]))
    fastest_idx = int(np.argmin([r["duration_s"] for r in routes]))

    # 4. Composite score for balanced
    max_duration = max(r["duration_s"] for r in routes) or 1
    composites = []
    for r in routes:
        norm_safety = r["safety_score"] / 100.0
        norm_speed = 1 - (r["duration_s"] / max_duration)  # faster → higher
        composites.append(safety_weight * norm_safety + speed_weight * norm_speed)

    balanced_idx = int(np.argmax(composites))

    # 5. Assign labels (handle ties / overlaps)
    used_indices: set[int] = set()

    # Priority: fastest first (objective metric), then safest, then balanced
    label_map: dict[int, str] = {}

    # Fastest — always the actual shortest duration
    label_map[fastest_idx] = "fastest"
    used_indices.add(fastest_idx)

    # Safest — highest safety score (unless already claimed as fastest)
    if safest_idx not in used_indices:
        label_map[safest_idx] = "safest"
        used_indices.add(safest_idx)
    else:
        # Fastest is also safest — find route with next-highest safety
        safety_order = sorted(
            enumerate(r["safety_score"] for r in routes),
            key=lambda x: x[1],
            reverse=True,
        )
        for idx, _ in safety_order:
            if idx not in used_indices:
                label_map[idx] = "safest"
                used_indices.add(idx)
                break

    # Balanced — best composite (if not already claimed)
    if balanced_idx not in used_indices:
        label_map[balanced_idx] = "balanced"
        used_indices.add(balanced_idx)
    else:
        composite_order = sorted(
            enumerate(composites), key=lambda x: x[1], reverse=True
        )
        for idx, _ in composite_order:
            if idx not in used_indices:
                label_map[idx] = "balanced"
                used_indices.add(idx)
                break

    # Any remaining routes (if ORS returns >3)
    for i in range(len(routes)):
        if i not in label_map:
            label_map[i] = "alternative"

    for i, route in enumerate(routes):
        route["rank_label"] = label_map.get(i, "alternative")

    # 6. Sort: safest → balanced → fastest → alternative
    rank_order = {"safest": 0, "balanced": 1, "fastest": 2, "alternative": 3}
    routes.sort(key=lambda r: rank_order.get(r["rank_label"], 99))

    return routes
