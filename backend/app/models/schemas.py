"""
Boston Pulse — API Schemas
Pydantic request/response models for chatbot endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# =============================================================================
# GET /api/health
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime


# =============================================================================
# POST /api/chat
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Pass back session_id from previous response to continue conversation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Which restaurants in Roxbury failed health inspections?",
                "session_id": "optional-session-id"
            }
        }


class RetrievedChunk(BaseModel):
    dataset: str        # e.g. "crime", "311", "food_inspections"
    content: str        # the text chunk
    score: float        # similarity score


DATASET_LABELS = {
    "crime": "BPD Crime Reports",
    "311": "Boston 311 Data",
    "food_inspections": "Food Inspection Records",
    "cityscore": "CityScore Data",
    "berdo": "BERDO Energy Reports",
    "street_sweeping": "Street Sweeping Schedule",
}

class ChatResponse(BaseModel):
    text: str
    sources: List[str] = []


# =============================================================================
# POST /api/ingest  (triggers re-ingestion manually if needed)
# =============================================================================

class IngestResponse(BaseModel):
    status: str
    datasets_ingested: List[str]
    total_chunks: int
    message: str
