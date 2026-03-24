# Boston Pulse — Chatbot Backend

RAG-based conversational API for the Boston Pulse civic intelligence platform. Answers natural language questions about Boston city data using ChromaDB vector search and Gemini as the LLM.

## Architecture

- **FastAPI** — REST API framework
- **ChromaDB** — Vector database for storing and retrieving embeddings
- **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) — Embedding model
- **Gemini 2.5 Flash** — LLM for generating answers
- **Google Cloud Storage** — Source of processed Boston dataset parquet files

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Ask a question about Boston city data |
| POST | `/api/ingest` | Trigger re-ingestion of data into ChromaDB |

## Folder Structure
```
backend/
├── app/
│   ├── core/           # Config and settings
│   ├── models/         # Pydantic request/response models
│   ├── routes/         # API route handlers (chat, health, ingest)
│   └── services/       # Business logic (embedder, retriever, gemini client, chat service)
├── config/             # Environment configs
├── scripts/            # Utility scripts
├── tests/              # Pytest test suite
├── Dockerfile          # Container setup
└── requirements.txt    # Python dependencies
```

## Running Locally

1. Clone the repo and navigate to the backend folder:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
export GEMINI_API_KEY=your_gemini_api_key
export HF_TOKEN=your_huggingface_token
export GCS_BUCKET=boston-pulse-data-pipeline
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
```

5. Run the API:
```bash
uvicorn app.main:app --reload --port 8000
```

6. Ingest data into ChromaDB:
```bash
curl -X POST http://localhost:8000/api/ingest
```

7. Ask a question:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the most common 311 complaints in Boston?"}'
```

## Running Tests
```bash
pytest tests/
```

## Deployment

The backend is deployed on Google Cloud Run at:
`https://boston-pulse-chatbot-384523870431.us-central1.run.app`

Deployed with 2Gi memory and 2 CPUs to support the HuggingFace embedding model.
