"""
Boston Pulse — Backend API

This module hosts TWO micro-services side-by-side:
  1. Flask app  – Safe Walking Route API   (port 5001, /api/routes/safe-walk …)
  2. FastAPI app – RAG Chatbot API          (port 8000, /api/chat …)

Run whichever you need:
  flask --app app.main run -p 5001
  uvicorn app.main:fastapi_app --port 8000
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flasgger import Swagger
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Load environment variables from backend/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1.  FLASK — Safe Walking Route API                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def create_app(config_path: str | None = None) -> Flask:
    """Application factory."""
    app = Flask(__name__, static_folder="static")
    CORS(app)

    # ── Swagger Configuration ────────────────────────────────────────────
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/",
    }

    swagger_template = {
        "info": {
            "title": "Boston Pulse — Safe Walking Routes API",
            "description": (
                "API for finding safety-ranked walking routes in Boston.\n\n"
                "Uses the OpenRouteService Directions API to fetch multiple walking "
                "routes, scores each against a pre-computed spatial safety grid, "
                "and ranks them as **safest**, **balanced**, or **fastest**."
            ),
            "version": "1.0.0",
            "contact": {"name": "Boston Pulse Team"},
        },
        "basePath": "/",
        "schemes": ["http"],
    }

    Swagger(app, config=swagger_config, template=swagger_template)

    # Resolve paths relative to the backend/ directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if config_path is None:
        config_path = os.path.join(backend_dir, "config", "config.yaml")

    # ------------------------------------------------------------------
    # Lazy-load heavy objects on first request
    # ------------------------------------------------------------------
    _state: dict = {}

    def _get_safety_grid():
        if "grid" not in _state:
            from app.safety.safety_scorer import SafetyGrid

            _state["grid"] = SafetyGrid(
                project="bostonpulse",
                collection="h3_scores",
            )
        return _state["grid"]

    def _get_ors_client():
        if "ors" not in _state:
            from app.routes.ors_client import ORSClient

            _state["ors"] = ORSClient(config_path=config_path)
        return _state["ors"]

    # ------------------------------------------------------------------
    # Boston bounding-box validation
    # ------------------------------------------------------------------
    BOSTON_BOUNDS = {
        "lat_min": 42.2270,
        "lat_max": 42.3970,
        "lon_min": -71.1910,
        "lon_max": -70.9230,
    }

    def _in_boston(lat: float, lon: float) -> bool:
        return (
            BOSTON_BOUNDS["lat_min"] <= lat <= BOSTON_BOUNDS["lat_max"]
            and BOSTON_BOUNDS["lon_min"] <= lon <= BOSTON_BOUNDS["lon_max"]
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/routes/safe-walk", methods=["POST"])
    def safe_walk():
        """Return safety-ranked walking routes between two Boston locations.
        ---
        tags:
          - Routes
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            description: Source/destination coordinates with time context
            schema:
              type: object
              required:
                - source
                - destination
                - date
                - day
                - time
              properties:
                source:
                  type: array
                  items:
                    type: number
                  example: [42.3601, -71.0589]
                  description: "Source coordinates [latitude, longitude]"
                destination:
                  type: array
                  items:
                    type: number
                  example: [42.3505, -71.0648]
                  description: "Destination coordinates [latitude, longitude]"
                date:
                  type: string
                  example: "2026-03-16"
                  description: "Date in YYYY-MM-DD format"
                day:
                  type: string
                  enum: [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
                  example: "Monday"
                  description: "Day of the week"
                time:
                  type: string
                  example: "14:30"
                  description: "Time in HH:MM format"
        responses:
          200:
            description: Safety-ranked walking routes
            schema:
              type: object
              properties:
                routes:
                  type: array
                  items:
                    type: object
                    properties:
                      rank_label:
                        type: string
                        description: "Route ranking — safest, balanced, fastest, or alternative"
                        example: "safest"
                      safety_score:
                        type: number
                        description: "Average safety score along the route (0–100)"
                        example: 78.4
                      distance_m:
                        type: number
                        description: "Total walking distance in metres"
                        example: 1234.5
                      duration_s:
                        type: number
                        description: "Estimated walking duration in seconds"
                        example: 920.0
                      geometry:
                        type: array
                        items:
                          type: array
                          items:
                            type: number
                        description: "List of [lat, lon] coordinates forming the route path"
                meta:
                  type: object
                  properties:
                    date:
                      type: string
                    day:
                      type: string
                    time:
                      type: string
          400:
            description: Invalid request (missing fields, bad coordinates, outside Boston)
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Source coordinates are outside Boston"
          502:
            description: Failed to fetch routes from OpenRouteService
            schema:
              type: object
              properties:
                error:
                  type: string
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # --- validate required fields ---
        for field in ("source", "destination", "date", "day", "time"):
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        source = data["source"]
        destination = data["destination"]

        if (
            not isinstance(source, list)
            or len(source) != 2
            or not isinstance(destination, list)
            or len(destination) != 2
        ):
            return (
                jsonify({"error": "source and destination must be [lat, lon] arrays"}),
                400,
            )

        try:
            src_lat, src_lon = float(source[0]), float(source[1])
            dst_lat, dst_lon = float(destination[0]), float(destination[1])
        except (TypeError, ValueError):
            return jsonify({"error": "Coordinates must be numeric"}), 400

        if not _in_boston(src_lat, src_lon):
            return (
                jsonify({"error": "Source coordinates are outside Boston"}),
                400,
            )
        if not _in_boston(dst_lat, dst_lon):
            return (
                jsonify({"error": "Destination coordinates are outside Boston"}),
                400,
            )

        # --- parse time context ---
        try:
            time_parts = data["time"].split(":")
            hour = int(time_parts[0])
        except (ValueError, IndexError, AttributeError):
            return jsonify({"error": "time must be in HH:MM format"}), 400

        day_str = data.get("day", "Monday")
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        day_of_week = day_map.get(day_str.lower(), 0)

        # --- fetch routes from ORS ---
        try:
            ors = _get_ors_client()
            routes = ors.get_walking_routes(
                source=(src_lat, src_lon),
                destination=(dst_lat, dst_lon),
            )
        except Exception as exc:
            return (
                jsonify({"error": f"Failed to fetch routes from ORS: {str(exc)}"}),
                502,
            )

        # --- rank routes ---
        from app.routes.route_ranker import rank_routes

        grid = _get_safety_grid()

        import yaml

        ranking_cfg: dict = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                ranking_cfg = (yaml.safe_load(f) or {}).get("ranking", {})

        ranked = rank_routes(
            routes,
            grid,
            hour=hour,
            day_of_week=day_of_week,
            safety_weight=ranking_cfg.get("safety_weight", 0.6),
            speed_weight=ranking_cfg.get("speed_weight", 0.4),
        )

        # --- build response ---
        response_routes = []
        for r in ranked:
            response_routes.append(
                {
                    "rank_label": r["rank_label"],
                    "safety_score": r["safety_score"],
                    "distance_m": round(r["distance_m"], 1),
                    "duration_s": round(r["duration_s"], 1),
                    "geometry": r["geometry"],
                }
            )

        return jsonify(
            {
                "routes": response_routes,
                "meta": {
                    "date": data.get("date"),
                    "day": data.get("day"),
                    "time": data.get("time"),
                },
            }
        )

    @app.route("/api/routes/safe-walk", methods=["GET"])
    def safe_walk_get():
        """GET version — fetch safety-ranked walking routes via query params.

        Example: /api/routes/safe-walk?src_lat=42.3601&src_lon=-71.0589&dst_lat=42.3505&dst_lon=-71.0648&date=2026-03-16&day=Monday&time=14:30
        ---
        tags:
          - Routes
        parameters:
          - name: src_lat
            in: query
            type: number
            required: true
            description: Source latitude
            example: 42.3601
          - name: src_lon
            in: query
            type: number
            required: true
            description: Source longitude
            example: -71.0589
          - name: dst_lat
            in: query
            type: number
            required: true
            description: Destination latitude
            example: 42.3505
          - name: dst_lon
            in: query
            type: number
            required: true
            description: Destination longitude
            example: -71.0648
          - name: date
            in: query
            type: string
            required: true
            example: "2026-03-16"
          - name: day
            in: query
            type: string
            required: true
            enum: [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            example: Monday
          - name: time
            in: query
            type: string
            required: true
            description: "Time in HH:MM"
            example: "14:30"
        responses:
          200:
            description: Safety-ranked walking routes
          400:
            description: Missing or invalid parameters
          502:
            description: ORS API error
        """
        try:
            src_lat = float(request.args["src_lat"])
            src_lon = float(request.args["src_lon"])
            dst_lat = float(request.args["dst_lat"])
            dst_lon = float(request.args["dst_lon"])
            date_str = request.args["date"]
            day_str = request.args["day"]
            time_str = request.args["time"]
        except (KeyError, ValueError) as e:
            return jsonify({
                "error": f"Missing or invalid query parameter: {e}",
                "usage": (
                    "/api/routes/safe-walk?"
                    "src_lat=42.3601&src_lon=-71.0589"
                    "&dst_lat=42.3505&dst_lon=-71.0648"
                    "&date=2026-03-16&day=Monday&time=14:30"
                ),
            }), 400

        if not _in_boston(src_lat, src_lon):
            return jsonify({"error": "Source coordinates are outside Boston"}), 400
        if not _in_boston(dst_lat, dst_lon):
            return jsonify({"error": "Destination coordinates are outside Boston"}), 400

        hour = int(time_str.split(":")[0])
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        day_of_week = day_map.get(day_str.lower(), 0)

        try:
            ors = _get_ors_client()
            routes = ors.get_walking_routes(
                source=(src_lat, src_lon), destination=(dst_lat, dst_lon)
            )
        except Exception as exc:
            return jsonify({"error": f"ORS error: {str(exc)}"}), 502

        from app.routes.route_ranker import rank_routes
        import yaml

        grid = _get_safety_grid()
        ranking_cfg: dict = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                ranking_cfg = (yaml.safe_load(f) or {}).get("ranking", {})

        ranked = rank_routes(
            routes, grid, hour=hour, day_of_week=day_of_week,
            safety_weight=ranking_cfg.get("safety_weight", 0.6),
            speed_weight=ranking_cfg.get("speed_weight", 0.4),
        )

        response_routes = [
            {
                "rank_label": r["rank_label"],
                "safety_score": r["safety_score"],
                "distance_m": round(r["distance_m"], 1),
                "duration_s": round(r["duration_s"], 1),
                "geometry": r["geometry"],
            }
            for r in ranked
        ]

        return jsonify({
            "routes": response_routes,
            "meta": {"date": date_str, "day": day_str, "time": time_str},
        })

    @app.route("/api/safety-score", methods=["GET"])
    def safety_score():
        """Get the safety score for a single location.
        ---
        tags:
          - Safety
        parameters:
          - name: lat
            in: query
            type: number
            required: true
            example: 42.3601
          - name: lon
            in: query
            type: number
            required: true
            example: -71.0589
          - name: hour
            in: query
            type: integer
            required: false
            default: 12
            description: "Hour of day (0–23)"
          - name: day_of_week
            in: query
            type: integer
            required: false
            default: 0
            description: "0=Monday … 6=Sunday"
        responses:
          200:
            description: Safety score for the location
            schema:
              type: object
              properties:
                lat:
                  type: number
                lon:
                  type: number
                safety_score:
                  type: number
                hour:
                  type: integer
                day_of_week:
                  type: integer
          400:
            description: Missing or invalid parameters
        """
        try:
            lat = float(request.args["lat"])
            lon = float(request.args["lon"])
        except (KeyError, ValueError):
            return jsonify({
                "error": "lat and lon query params required",
                "usage": "/api/safety-score?lat=42.3601&lon=-71.0589&hour=14&day_of_week=0",
            }), 400

        hour = int(request.args.get("hour", 12))
        day_of_week = int(request.args.get("day_of_week", 0))

        grid = _get_safety_grid()
        score = grid.get_score(lat, lon, hour, day_of_week)

        return jsonify({
            "lat": lat,
            "lon": lon,
            "safety_score": round(score, 2),
            "hour": hour,
            "day_of_week": day_of_week,
        })

    @app.route("/api/safety-grid/info", methods=["GET"])
    def grid_info():
        """Get safety grid metadata.
        ---
        tags:
          - Safety
        responses:
          200:
            description: Grid metadata
            schema:
              type: object
              properties:
                bounds:
                  type: object
                h3_resolution:
                  type: integer
                data_source:
                  type: string
        """
        from app.safety.safety_scorer import H3_RESOLUTION

        return jsonify({
            "bounds": BOSTON_BOUNDS,
            "h3_resolution": H3_RESOLUTION,
            "data_source": "firestore:h3_scores",
        })

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint.
        ---
        tags:
          - System
        responses:
          200:
            description: Service is healthy
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: "ok"
        """
        return jsonify({"status": "ok"})

    @app.route("/")
    def index():
        """Serve the frontend."""
        return send_from_directory(app.static_folder, "index.html")

    return app


# Allow `flask --app app.main run`
app = create_app()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2.  FASTAPI — RAG Chatbot API                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes import health as health_route, chat, ingest

fastapi_app = FastAPI(
    title="Boston Pulse Chatbot API",
    description="RAG-powered civic intelligence chatbot for Boston.",
    version="0.1.0",
    docs_url="/docs",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app.include_router(health_route.router, tags=["Health"])
fastapi_app.include_router(chat.router,         prefix="/api", tags=["Chat"])
fastapi_app.include_router(ingest.router,       prefix="/api", tags=["Ingest"])
