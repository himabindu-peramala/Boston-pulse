# Boston Pulse — Backend

The backend hosts **two APIs** side-by-side:

1. **Chatbot API (FastAPI)** — RAG-based conversational interface for Boston city data using ChromaDB + Gemini.
2. **Safe Walking Route API (Flask)** — Safety-ranked pedestrian navigation using OpenRouteService + Firestore H3 safety grid.

---

## Architecture

### Chatbot API
- **FastAPI** — REST framework
- **ChromaDB** — Vector DB for embeddings
- **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) — Embedding model
- **Gemini 2.5 Flash** — LLM for answer generation
- **Google Cloud Storage** — Source of processed parquet files

### Safe Walking Route API
- **Flask + Flasgger** — REST framework with Swagger UI
- **OpenRouteService** — Walking route generation
- **Google Cloud Firestore** — H3-indexed spatial safety scores
- **H3** — Uber's hexagonal spatial index for safety lookups

---

## Endpoints

### Chatbot (FastAPI — port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Ask a question about Boston city data |
| POST | `/api/ingest` | Trigger re-ingestion of data into ChromaDB |

### Safe Walking Routes (Flask — port 5001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/routes/safe-walk` | Get safety-ranked walking routes (JSON body) |
| GET | `/api/routes/safe-walk` | Get safety-ranked walking routes (query params) |
| GET | `/api/safety-score` | Get safety score for a single lat/lon |
| GET | `/api/safety-grid/info` | Safety grid metadata |
| GET | `/apidocs/` | Swagger UI |

---

## Folder Structure
```
backend/
├── app/
│   ├── core/           # Config and settings (Pydantic, GCS loader)
│   ├── models/         # Pydantic request/response schemas
│   ├── routes/         # API route handlers (chat, health, ingest, route_ranker, ors_client)
│   ├── safety/         # Safety scorer (H3 + Firestore)
│   ├── services/       # Business logic (embedder, retriever, gemini client, chat service)
│   ├── static/         # Frontend assets
│   └── main.py         # Entry point — Flask app + FastAPI app
├── config/             # YAML configs (ORS, ranking weights)
├── evaluation/         # RAG evaluation framework
├── scripts/            # Utility scripts
├── tests/              # Pytest test suite
├── Dockerfile          # Container setup
└── requirements.txt    # Python dependencies
```

---

## Running Locally

### 1. Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
export GEMINI_API_KEY=your_gemini_api_key
export HF_TOKEN=your_huggingface_token
export GCS_BUCKET=boston-pulse-data-pipeline
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
```

### 3. Run the Chatbot API (FastAPI)

```bash
uvicorn app.main:fastapi_app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the Safe Walking Route API (Flask)

```bash
flask --app app.main run -p 5001
```

Swagger UI: [http://localhost:5001/apidocs](http://localhost:5001/apidocs)

> **Note:** Both servers can run simultaneously in separate terminals.

---

## Example Requests

### Chat with the bot
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the most common 311 complaints in Boston?"}'
```

### Ingest data into ChromaDB
```bash
curl -X POST http://localhost:8000/api/ingest
```

### Get safe walking routes (POST)
```bash
curl -X POST http://localhost:5001/api/routes/safe-walk \
  -H "Content-Type: application/json" \
  -d '{
    "source": [42.3601, -71.0589],
    "destination": [42.3505, -71.0648],
    "date": "2026-03-16",
    "day": "Monday",
    "time": "14:30"
  }'
```

### Get safe walking routes (GET)
```bash
curl "http://localhost:5001/api/routes/safe-walk?src_lat=42.3601&src_lon=-71.0589&dst_lat=42.3505&dst_lon=-71.0648&date=2026-03-16&day=Monday&time=14:30"
```

### Get safety score for a location
```bash
curl "http://localhost:5001/api/safety-score?lat=42.3601&lon=-71.0589&hour=14&day_of_week=0"
```

---

## Running Tests
```bash
pytest tests/
```

---

## Deployment

The backend is deployed on Google Cloud Run at:
`https://boston-pulse-chatbot-384523870431.us-central1.run.app`

Deployed with 2Gi memory and 2 CPUs to support the HuggingFace embedding model.
