"""
Boston Pulse — Chat Service
Full RAG pipeline: retrieve relevant chunks → generate grounded answer.
"""
import logging
import uuid
from typing import List

from app.core.session_store import add_turn, get_history
from app.models.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.services.retriever import retrieve
from app.services.gemini_client import generate_answer

logger = logging.getLogger(__name__)


async def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Full RAG flow:
    1. Get conversation history
    2. Retrieve relevant chunks from ChromaDB
    3. Generate grounded answer with Gemini
    4. Save turn to history
    5. Return answer with sources
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id)

    # Step 1 — Retrieve relevant chunks from vector DB
    chunks: List[RetrievedChunk] = retrieve(request.message)

    if chunks:
        # Step 2 — Build context from retrieved chunks
        context_texts = [c.content for c in chunks]
        # Step 3 — Generate answer grounded in context
        answer = await generate_answer(
            question=request.message,
            context_chunks=context_texts,
            history=history,
        )
    else:
        # No chunks found — tell user honestly
        answer = (
            "I don't have enough Boston city data loaded yet to answer that question. "
            "Please ensure the data ingestion pipeline has been run."
        )

    # Step 4 — Save to session history
    add_turn(session_id, request.message, answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=chunks,
    )
