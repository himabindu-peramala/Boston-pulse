"""
ORS Client — wrapper around the OpenRouteService Directions API.

Fetches multiple walking routes between two points using the
``foot-walking`` profile with ``alternative_routes`` enabled.
"""

from __future__ import annotations

import os
from typing import Any

import polyline as pl
import requests
import yaml


class ORSClient:
    """Thin wrapper around the ORS v2 Directions API (POST, JSON response)."""

    DEFAULT_BASE_URL = (
        "https://api.openrouteservice.org/v2/directions/foot-walking/json"
    )

    def __init__(
        self,
        api_key: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ORS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ORS API key is required. Set ORS_API_KEY env var or pass api_key."
            )

        # Load config for alternative-route tuning
        self.base_url = self.DEFAULT_BASE_URL
        self.alt_routes_params = {
            "target_count": 3,
            "share_factor": 0.6,
            "weight_factor": 1.4,
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            ors_cfg = cfg.get("ors", {})
            self.base_url = ors_cfg.get("base_url", self.base_url)
            alt = ors_cfg.get("alternative_routes", {})
            self.alt_routes_params.update(alt)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_walking_routes(
        self,
        source: tuple[float, float],
        destination: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Fetch walking routes from ORS.

        Parameters
        ----------
        source : (lat, lon)
        destination : (lat, lon)

        Returns
        -------
        list[dict]
            Each dict has:
              - ``geometry``: list of (lat, lon) tuples
              - ``distance_m``: total distance in metres
              - ``duration_s``: total duration in seconds
        """
        # ORS expects [lon, lat] order
        body = {
            "coordinates": [
                [source[1], source[0]],
                [destination[1], destination[0]],
            ],
            "alternative_routes": self.alt_routes_params,
            "geometry": True,
            "instructions": False,
        }

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }

        resp = requests.post(self.base_url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        routes: list[dict[str, Any]] = []
        for route in data.get("routes", []):
            summary = route.get("summary", {})
            encoded_geom = route.get("geometry", "")

            # Decode polyline (ORS returns encoded polyline by default)
            decoded = pl.decode(encoded_geom)  # list of (lat, lon)

            routes.append(
                {
                    "geometry": decoded,
                    "distance_m": summary.get("distance", 0),
                    "duration_s": summary.get("duration", 0),
                }
            )

        if not routes:
            raise ValueError("ORS returned no routes for the given coordinates.")

        return routes
