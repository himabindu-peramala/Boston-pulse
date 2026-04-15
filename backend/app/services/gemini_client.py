"""
Boston Pulse — Gemini LLM Client
Generates answers from retrieved context chunks.
"""
import logging
from typing import List, Optional

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        genai.configure(api_key=settings.gemini_api_key)
        _model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
            ),
            system_instruction=SYSTEM_PROMPT,
        )
    return _model


async def generate_answer(
    question: str,
    context_chunks: List[str],
    history: Optional[List[dict]] = None,
) -> str:
    """
    Generate an answer using retrieved context chunks.
    
    Args:
        question: User's question
        context_chunks: Retrieved text chunks from vector DB
        history: Conversation history in Gemini format
    
    Returns:
        Gemini's grounded answer
    """
    try:
        # Build context string from retrieved chunks
        context = "\n\n".join(
            f"[Source {i+1}]: {chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        prompt = (
            f"Using the following Boston city data as context, "
            f"answer the question. When citing sources, use dataset names like BPD Crime Reports, Boston 311 Data, Food Inspection Records instead of Source 1, Source 2.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        model = _get_model()
        chat = model.start_chat(history=history or [])
        response = await chat.send_message_async(prompt)
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise RuntimeError(f"LLM unavailable: {e}") from e


SYSTEM_PROMPT = """You are Boston Pulse, an AI civic assistant for the City of Boston.
You answer questions about Boston neighborhoods, safety, city services, and civic data.

Rules:
1. Only use the context provided to answer. Never make up statistics.
2. Always cite your source (e.g. "Based on crime incident data..." or "311 records show...").
3. Be concise and practical. Residents need quick, actionable answers.
4. If the context doesn't contain enough info, say so honestly.
5. For safety questions, be balanced — present data without being alarmist.
"""
