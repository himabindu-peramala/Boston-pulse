"""
Boston Pulse — Chat Route
POST /api/chat
DELETE /api/chat/history/{session_id}
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat
from app.core.session_store import clear_history

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chatbot endpoint.
    Retrieves relevant Boston city data chunks,
    generates a grounded answer with Gemini,
    and returns the answer with citations.
    """
    try:
        return await handle_chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/history/{session_id}")
async def delete_history(session_id: str):
    """Clear conversation history for a session."""
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}
