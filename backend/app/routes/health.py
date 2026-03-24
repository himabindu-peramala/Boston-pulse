"""
Boston Pulse — Health Route
GET /api/health
"""
from fastapi import APIRouter
from datetime import datetime
from app.models.schemas import HealthResponse
from app.services.retriever import get_collection_stats

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — confirms API is alive and shows vector DB stats."""
    stats = get_collection_stats()
    return HealthResponse(
        status="ok",
        service=f"boston-pulse-chatbot-api | chunks in DB: {stats['total_chunks']}",
        timestamp=datetime.utcnow(),
    )
